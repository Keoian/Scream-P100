// cpu_experts.cpp: see cpu_experts.hpp. Compiled as plain C++ (no CUDA); the AVX2 kernels are target-attributed functions chosen at
// run time, so the file builds with the project's flags on any x86-64 and falls back to the scalar steps elsewhere.
#include "cpu_experts.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define FNX_CPUX_X86 1
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#define FNX_CPUX_SIMD __attribute__((target("avx2,fma,f16c")))
#else
#define FNX_CPUX_X86 0
#endif

namespace fnx {
namespace cpux {
namespace {

constexpr int8_t KV_IQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
constexpr int MAXN = 4096;   // the widest row a chunk decodes on its stack

inline float half_f(uint16_t h) {   // IEEE half to float, every value (ggml's table holds the same)
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {
            int sh = 0;
            uint32_t mm = m;
            while (!(mm & 1024u)) mm <<= 1, ++sh;
            u = s | ((uint32_t) (127 - 15 - sh + 1) << 23) | ((mm & 1023u) << 13);
        }
    } else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 112u) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline uint16_t rd16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
// ggml's get_scale_min_k4: sub-block j's 6-bit scale and minimum of a K-quant super-block
inline void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (uint8_t) ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = (uint8_t) ((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// ---- ggml's dequantize_row_* steps
void ref_q4_0(const uint8_t* s, int n, float* y) {
    for (int b = 0; b < n / 32; ++b, s += 18, y += 32) {
        const float d = half_f(rd16(s));
        for (int j = 0; j < 16; ++j) {
            y[j] = (float) ((s[2 + j] & 0x0F) - 8) * d;
            y[j + 16] = (float) ((s[2 + j] >> 4) - 8) * d;
        }
    }
}
void ref_q5_0(const uint8_t* s, int n, float* y) {
    for (int b = 0; b < n / 32; ++b, s += 22, y += 32) {
        const float d = half_f(rd16(s));
        uint32_t qh;
        std::memcpy(&qh, s + 2, 4);
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh0 = (uint8_t) (((qh >> (j + 0)) << 4) & 0x10), xh1 = (uint8_t) ((qh >> (j + 12)) & 0x10);
            y[j] = (float) (((s[6 + j] & 0x0F) | xh0) - 16) * d;
            y[j + 16] = (float) (((s[6 + j] >> 4) | xh1) - 16) * d;
        }
    }
}
void ref_q5_1(const uint8_t* s, int n, float* y) {
    for (int b = 0; b < n / 32; ++b, s += 24, y += 32) {
        const float d = half_f(rd16(s)), m = half_f(rd16(s + 2));
        uint32_t qh;
        std::memcpy(&qh, s + 4, 4);
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh0 = (uint8_t) (((qh >> (j + 0)) << 4) & 0x10), xh1 = (uint8_t) ((qh >> (j + 12)) & 0x10);
            y[j] = (float) ((s[8 + j] & 0x0F) | xh0) * d + m;
            y[j + 16] = (float) ((s[8 + j] >> 4) | xh1) * d + m;
        }
    }
}
void ref_q8_0(const uint8_t* s, int n, float* y) {
    for (int b = 0; b < n / 32; ++b, s += 34, y += 32) {
        const float d = half_f(rd16(s));
        for (int j = 0; j < 32; ++j) y[j] = (float) (int8_t) s[2 + j] * d;
    }
}
void ref_q4_k(const uint8_t* s, int n, float* y) {
    for (int b = 0; b < n / 256; ++b, s += 144) {
        const float d = half_f(rd16(s)), mn = half_f(rd16(s + 2));
        const uint8_t* sc = s + 4;
        const uint8_t* q = s + 16;
        for (int is = 0; is < 8; is += 2, q += 32) {
            uint8_t c, m;
            scale_min_k4(is, sc, c, m);
            const float d1 = d * c, m1 = mn * m;
            scale_min_k4(is + 1, sc, c, m);
            const float d2 = d * c, m2 = mn * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (float) (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (float) (q[l] >> 4) - m2;
        }
    }
}
void ref_q5_k(const uint8_t* s, int n, float* y) {
    for (int b = 0; b < n / 256; ++b, s += 176) {
        const float d = half_f(rd16(s)), mn = half_f(rd16(s + 2));
        const uint8_t* sc = s + 4;
        const uint8_t* qh = s + 16;
        const uint8_t* ql = s + 48;
        uint8_t u1 = 1, u2 = 2;
        for (int is = 0; is < 8; is += 2, ql += 32, u1 <<= 2, u2 <<= 2) {
            uint8_t c, m;
            scale_min_k4(is, sc, c, m);
            const float d1 = d * c, m1 = mn * m;
            scale_min_k4(is + 1, sc, c, m);
            const float d2 = d * c, m2 = mn * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (float) ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (float) ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        }
    }
}
void ref_q6_k(const uint8_t* s, int n, float* y) {
    for (int b = 0; b < n / 256; ++b, s += 210) {
        const uint8_t* ql = s;
        const uint8_t* qh = s + 128;
        const int8_t* sc = (const int8_t*) s + 192;
        const float d = half_f(rd16(s + 208));
        for (int h = 0; h < 2; ++h, y += 128, ql += 64, qh += 32, sc += 8)
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t) ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t) ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
    }
}
void ref_iq4_nl(const uint8_t* s, int n, float* y) {
    for (int b = 0; b < n / 32; ++b, s += 18, y += 32) {
        const float d = half_f(rd16(s));
        for (int j = 0; j < 16; ++j) {
            y[j] = d * (float) KV_IQ4NL[s[2 + j] & 0xf];
            y[j + 16] = d * (float) KV_IQ4NL[s[2 + j] >> 4];
        }
    }
}

float dot_ref(const float* a, const float* b, int n) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += (double) a[i] * b[i];
    return (float) s;
}

#if FNX_CPUX_X86
// 32 small unsigned values (a byte each) to floats, each times dl plus ml
FNX_CPUX_SIMD inline void put32(__m256i q, __m256 dl, __m256 ml, float* y) {
    const __m128i lo = _mm256_castsi256_si128(q), hi = _mm256_extracti128_si256(q, 1);
    _mm256_storeu_ps(y, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)), dl, ml));
    _mm256_storeu_ps(y + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8))), dl, ml));
    _mm256_storeu_ps(y + 16, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)), dl, ml));
    _mm256_storeu_ps(y + 24, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8))), dl, ml));
}
FNX_CPUX_SIMD void simd_q4_k(const uint8_t* s, int n, float* y) {
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    for (int b = 0; b < n / 256; ++b, s += 144) {
        const float d = half_f(rd16(s)), mn = half_f(rd16(s + 2));
        const uint8_t* sc = s + 4;
        const uint8_t* q = s + 16;
        for (int is = 0; is < 8; is += 2, q += 32, y += 64) {
            uint8_t c1, m1, c2, m2;
            scale_min_k4(is, sc, c1, m1);
            scale_min_k4(is + 1, sc, c2, m2);
            const __m256i v = _mm256_loadu_si256((const __m256i*) q);
            put32(_mm256_and_si256(v, m4), _mm256_set1_ps(d * c1), _mm256_set1_ps(-(mn * m1)), y);
            put32(_mm256_and_si256(_mm256_srli_epi16(v, 4), m4), _mm256_set1_ps(d * c2), _mm256_set1_ps(-(mn * m2)), y + 32);
        }
    }
}
FNX_CPUX_SIMD void simd_q5_k(const uint8_t* s, int n, float* y) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), c16 = _mm256_set1_epi8(16);
    for (int b = 0; b < n / 256; ++b, s += 176) {
        const float d = half_f(rd16(s)), mn = half_f(rd16(s + 2));
        const uint8_t* sc = s + 4;
        const __m256i qh = _mm256_loadu_si256((const __m256i*) (s + 16));
        const uint8_t* q = s + 48;
        for (int is = 0; is < 8; is += 2, q += 32, y += 64) {
            uint8_t c1, m1, c2, m2;
            scale_min_k4(is, sc, c1, m1);
            scale_min_k4(is + 1, sc, c2, m2);
            const __m256i v = _mm256_loadu_si256((const __m256i*) q);
            const __m256i u1 = _mm256_set1_epi8((char) (1 << is)), u2 = _mm256_set1_epi8((char) (2 << is));
            const __m256i h1 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, u1), u1), c16);
            const __m256i h2 = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, u2), u2), c16);
            put32(_mm256_or_si256(_mm256_and_si256(v, m4), h1), _mm256_set1_ps(d * c1), _mm256_set1_ps(-(mn * m1)), y);
            put32(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v, 4), m4), h2), _mm256_set1_ps(d * c2), _mm256_set1_ps(-(mn * m2)), y + 32);
        }
    }
}
FNX_CPUX_SIMD void simd_iq4_nl(const uint8_t* s, int n, float* y) {
    const __m128i kv = _mm_loadu_si128((const __m128i*) KV_IQ4NL), m4 = _mm_set1_epi8(0x0F);
    for (int b = 0; b < n / 32; ++b, s += 18, y += 32) {
        const __m256 d = _mm256_set1_ps(half_f(rd16(s)));
        const __m128i v = _mm_loadu_si128((const __m128i*) (s + 2));
        const __m128i lo = _mm_shuffle_epi8(kv, _mm_and_si128(v, m4)), hi = _mm_shuffle_epi8(kv, _mm_and_si128(_mm_srli_epi16(v, 4), m4));
        _mm256_storeu_ps(y, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), d));
        _mm256_storeu_ps(y + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), d));
        _mm256_storeu_ps(y + 16, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), d));
        _mm256_storeu_ps(y + 24, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), d));
    }
}
FNX_CPUX_SIMD float simd_dot(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    const __m256 s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_shuffle_ps(r, r, 1));
    float f = _mm_cvtss_f32(r);
    for (; i < n; ++i) f += a[i] * b[i];
    return f;
}
FNX_CPUX_SIMD void simd_f16_to_f32(const uint16_t* src, float* dst, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (src + i))));
    for (; i < n; ++i) dst[i] = half_f(src[i]);
}
bool cpu_simd() {
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    return ok;
}
#else
bool cpu_simd() { return false; }
#endif

inline float silu(float g) { return g / (1.0f + std::exp(-g)); }

}  // namespace

bool have_simd() { return cpu_simd(); }

size_t row_bytes(uint32_t type, int n) {
    if (n <= 0 || n > MAXN) return 0;
    switch (type) {
        case T_Q4_0: return n % 32 ? 0 : (size_t) n / 32 * 18;
        case T_Q5_0: return n % 32 ? 0 : (size_t) n / 32 * 22;
        case T_Q5_1: return n % 32 ? 0 : (size_t) n / 32 * 24;
        case T_Q8_0: return n % 32 ? 0 : (size_t) n / 32 * 34;
        case T_IQ4_NL: return n % 32 ? 0 : (size_t) n / 32 * 18;
        case T_Q4_K: return n % 256 ? 0 : (size_t) n / 256 * 144;
        case T_Q5_K: return n % 256 ? 0 : (size_t) n / 256 * 176;
        case T_Q6_K: return n % 256 ? 0 : (size_t) n / 256 * 210;
        default: return 0;
    }
}
bool type_ok(uint32_t type, int n) { return row_bytes(type, n) != 0; }
const char* type_name(uint32_t type) {
    switch (type) {
        case T_Q4_0: return "Q4_0";
        case T_Q5_0: return "Q5_0";
        case T_Q5_1: return "Q5_1";
        case T_Q8_0: return "Q8_0";
        case T_IQ4_NL: return "IQ4_NL";
        case T_Q4_K: return "Q4_K";
        case T_Q5_K: return "Q5_K";
        case T_Q6_K: return "Q6_K";
        default: return "?";
    }
}

void decode_row_ref(uint32_t type, const uint8_t* src, int n, float* dst) {
    switch (type) {
        case T_Q4_0: ref_q4_0(src, n, dst); break;
        case T_Q5_0: ref_q5_0(src, n, dst); break;
        case T_Q5_1: ref_q5_1(src, n, dst); break;
        case T_Q8_0: ref_q8_0(src, n, dst); break;
        case T_IQ4_NL: ref_iq4_nl(src, n, dst); break;
        case T_Q4_K: ref_q4_k(src, n, dst); break;
        case T_Q5_K: ref_q5_k(src, n, dst); break;
        case T_Q6_K: ref_q6_k(src, n, dst); break;
        default: break;
    }
}
void decode_row(uint32_t type, const uint8_t* src, int n, float* dst) {
#if FNX_CPUX_X86
    if (cpu_simd()) {
        if (type == T_Q4_K) return simd_q4_k(src, n, dst);
        if (type == T_Q5_K) return simd_q5_k(src, n, dst);
        if (type == T_IQ4_NL) return simd_iq4_nl(src, n, dst);
    }
#endif
    decode_row_ref(type, src, n, dst);
}
float dot(const float* a, const float* b, int n) {
#if FNX_CPUX_X86
    if (cpu_simd()) return simd_dot(a, b, n);
#endif
    return dot_ref(a, b, n);
}
void f16_to_f32(const uint16_t* src, float* dst, size_t n) {
#if FNX_CPUX_X86
    if (cpu_simd()) return simd_f16_to_f32(src, dst, n);
#endif
    for (size_t i = 0; i < n; ++i) dst[i] = half_f(src[i]);
}

// ---- FNX_CPU_EXACT (cpu_experts.hpp): the oracle, on the integers
namespace {
typedef unsigned __int128 u128;
struct H16 {   // cls 0: finite, (-1)^sign * m * 2^e; 1: infinite; 2: NaN
    int sign, cls;
    uint32_t m;
    int e;
};
inline H16 h16_dec(uint16_t h) {
    H16 v{h >> 15, 0, 0u, 0};
    const uint32_t ex = (h >> 10) & 31u, mt = h & 1023u;
    if (ex == 31) v.cls = mt ? 2 : 1;
    else if (ex == 0) v.m = mt, v.e = -24;
    else v.m = 1024u + mt, v.e = (int) ex - 25;
    return v;
}
inline int bitlen(u128 x) {
    const uint64_t hi = (uint64_t) (x >> 64), lo = (uint64_t) x;
    return hi ? 128 - __builtin_clzll(hi) : lo ? 64 - __builtin_clzll(lo) : 0;
}
// sign * mag * 2^L (exact) to binary16: to nearest, ties to even, overflow to infinity
uint16_t h16_round(int sign, u128 mag, int L) {
    const uint16_t sb = (uint16_t) (sign << 15);
    if (mag == 0) return sb;
    int em = bitlen(mag) - 1 + L;              // the leading bit's exponent
    const int qe = em >= -14 ? em - 10 : -24;  // the exponent of the result's last bit (normal, subnormal)
    const int sh = qe - L;
    u128 q;
    if (sh <= 0) q = mag << (unsigned) -sh;
    else {
        q = mag >> (unsigned) sh;
        const u128 rem = mag - (q << (unsigned) sh), half = (u128) 1 << (unsigned) (sh - 1);
        if (rem > half || (rem == half && (q & 1))) ++q;
    }
    if (em >= -14) {
        if (q == 2048) q = 1024, ++em;
        if (em > 15) return (uint16_t) (sb | 0x7c00u);
        return (uint16_t) (sb | ((unsigned) (em + 15) << 10) | (unsigned) (q - 1024));
    }
    return (uint16_t) (sb | (unsigned) q);   // a subnormal (q 1024: the smallest normal, which has the same bits)
}
inline uint16_t h16_mul_ref(uint16_t a, uint16_t b) { return f16_fma_ref(a, b, 0x8000u); }   // a * b + (-0): the product's own zero sign
inline uint16_t h16_int(int v) { return f16_from_f32_ref((float) v); }
}  // namespace

uint16_t f16_fma_ref(uint16_t a, uint16_t b, uint16_t c) {
    const H16 x = h16_dec(a), y = h16_dec(b), z = h16_dec(c);
    const int ps = x.sign ^ y.sign;
    if (x.cls == 2 || y.cls == 2 || z.cls == 2) return 0x7e00u;
    if (x.cls == 1 || y.cls == 1) {
        if ((x.cls == 0 && x.m == 0) || (y.cls == 0 && y.m == 0)) return 0x7e00u;   // infinity times zero
        if (z.cls == 1 && z.sign != ps) return 0x7e00u;                              // infinity minus infinity
        return (uint16_t) ((ps << 15) | 0x7c00);
    }
    if (z.cls == 1) return c;
    const int pe = x.e + y.e, L = std::min(pe, z.e);
    const u128 P = (u128) ((uint64_t) x.m * y.m) << (unsigned) (pe - L), C = (u128) z.m << (unsigned) (z.e - L);
    if (P == 0 && C == 0) return (uint16_t) ((ps & z.sign) << 15);   // two zeros: -0 only from two -0
    if (ps == z.sign) return h16_round(ps, P + C, L);
    if (P == C) return 0;                                            // an exact cancellation: +0, rounding to nearest
    return P > C ? h16_round(ps, P - C, L) : h16_round(z.sign, C - P, L);
}
uint16_t f16_from_f32_ref(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const int sign = (int) (u >> 31), ex = (int) ((u >> 23) & 255);
    const uint32_t mt = u & 0x7fffffu;
    if (ex == 255) return mt ? 0x7e00u : (uint16_t) ((sign << 15) | 0x7c00);
    return ex ? h16_round(sign, mt | 0x800000u, ex - 150) : h16_round(sign, mt, -149);
}
float f16_to_f32(uint16_t h) { return half_f(h); }

bool exact_ok(uint32_t gu_type, uint32_t dn_type) {
    return (gu_type == T_Q4_K || gu_type == T_Q5_K) && (dn_type == T_IQ4_NL || dn_type == T_Q5_1 || dn_type == T_Q8_0);
}

// the card's gate|up row (p100_moe_rows.cuh k_moe_rows, kq_finish of p100_iq_gemm.cuh) with the oracle's steps: `lanes` parts, part p
// the 256-value groups [p * ngr / lanes, (p + 1) * ngr / lanes); a run's two chains from +0; the parts added as the xor shuffles do
float gu_row_card_ref(uint32_t type, const uint8_t* row, int n, const uint16_t* x, int lanes) {
    const bool k5 = type == T_Q5_K;
    const int bb = k5 ? 176 : 144, nruns = n / 32, ngr = (nruns + 7) / 8;
    float s[8] = {};
    for (int p = 0; p < lanes; ++p)
        for (int g = p * ngr / lanes; g < (p + 1) * ngr / lanes; ++g) {
            const uint8_t* b = row + (size_t) g * bb;
            const float d = half_f(rd16(b)), dm = half_f(rd16(b + 2));
            for (int ib = 0; ib < 8; ++ib) {
                uint8_t sc, m;
                scale_min_k4(ib, b + 4, sc, m);
                const uint16_t A = f16_from_f32_ref(d * (float) sc), B = f16_from_f32_ref(-dm * (float) m);
                uint16_t w[32];
                for (int l = 0; l < 32; ++l) {
                    int q = (b[(k5 ? 48 : 16) + 32 * (ib >> 1) + l] >> (4 * (ib & 1))) & 15;
                    if (k5) q |= ((b[16 + l] >> ib) & 1) << 4;
                    w[l] = f16_fma_ref(h16_int(q), A, B);
                }
                const uint16_t* xr = x + (size_t) g * 256 + ib * 32;
                uint16_t lo = 0, hi = 0;
                for (int t = 0; t < 16; ++t) {
                    lo = f16_fma_ref(w[2 * t], xr[2 * t], lo);
                    hi = f16_fma_ref(w[2 * t + 1], xr[2 * t + 1], hi);
                }
                s[p] = s[p] + (half_f(lo) + half_f(hi));
            }
        }
    for (int o = 1; o < lanes; o <<= 1)
        for (int p = 0; p + o < lanes; p += 2 * o) s[p] = s[p] + s[p + o];
    return s[0];
}
// the card's down row (one lane): IQ4_NL fp16(v d), Q8_0 fp16(q d), Q5_1 fp16(q d + m)
float dn_row_card_ref(uint32_t type, const uint8_t* row, int n, const uint16_t* h) {
    const int bb = type == T_Q5_1 ? 24 : type == T_Q8_0 ? 34 : 18;
    float s = 0.f;
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* blk = row + (size_t) b * bb;
        const uint16_t D = rd16(blk);
        uint16_t w[32];
        for (int l = 0; l < 32; ++l) {
            if (type == T_Q8_0) w[l] = h16_mul_ref(h16_int((int8_t) blk[2 + l]), D);
            else if (type == T_IQ4_NL) w[l] = h16_mul_ref(h16_int(KV_IQ4NL[(l < 16 ? blk[2 + l] : blk[2 + l - 16] >> 4) & 15]), D);
            else {
                uint32_t qh;
                std::memcpy(&qh, blk + 4, 4);
                const int q = ((l < 16 ? blk[8 + l] : blk[8 + l - 16] >> 4) & 15) | (int) (((qh >> l) & 1u) << 4);
                w[l] = f16_fma_ref(h16_int(q), D, rd16(blk + 2));
            }
        }
        const uint16_t* hr = h + (size_t) b * 32;
        uint16_t lo = 0, hi = 0;
        for (int t = 0; t < 16; ++t) {
            lo = f16_fma_ref(w[2 * t], hr[2 * t], lo);
            hi = f16_fma_ref(w[2 * t + 1], hr[2 * t + 1], hi);
        }
        s = s + (half_f(lo) + half_f(hi));
    }
    return s;
}

#if FNX_CPUX_X86
// ---- the products straight from the blocks (no decoded row in between): a sub-block's 32 values share a scale and a minimum, so
//      sum x (d q - m) = d * sum(x q) - m * sum(x), with sum(x) over each 32 columns taken once a token (xs)
FNX_CPUX_SIMD inline float hsum(__m256 s) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_shuffle_ps(r, r, 1));
    return _mm_cvtss_f32(r);
}
FNX_CPUX_SIMD inline float half_s(const uint8_t* p) { return _cvtsh_ss(rd16(p)); }
FNX_CPUX_SIMD inline __m256 dot32(__m256 w0, __m256 w1, __m256 w2, __m256 w3, const float* x) {
    return _mm256_fmadd_ps(w3, _mm256_loadu_ps(x + 24),
                           _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + 16), _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + 8), _mm256_mul_ps(w0, _mm256_loadu_ps(x)))));
}
// one row of n values in Q4_K (K5 false) or Q5_K blocks times NT rows of x; xs[k]: row k's sums of each 32 columns
template <int NT, bool K5>
FNX_CPUX_SIMD inline void row_kq(const uint8_t* s, int n, const float* const* x, const float* const* xs, float* out) {
    const __m256i m15 = _mm256_set1_epi32(15), one = _mm256_set1_epi32(1);
    __m256 acc[NT][2];
    float macc[NT];
    for (int k = 0; k < NT; ++k) acc[k][0] = acc[k][1] = _mm256_setzero_ps(), macc[k] = 0.f;
    const int bb = K5 ? 176 : 144;
    for (int b = 0; b < n / 256; ++b, s += bb) {
        const float d = half_s(s), mn = half_s(s + 2);
        const uint8_t* sc = s + 4;
        const uint8_t* q = s + (K5 ? 48 : 16);
        __m256i f[4];
        if (K5)
            for (int j = 0; j < 4; ++j) f[j] = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (s + 16 + 8 * j)));
        for (int is = 0; is < 8; is += 2, q += 32) {
            uint8_t c1, m1, c2, m2;
            scale_min_k4(is, sc, c1, m1);
            scale_min_k4(is + 1, sc, c2, m2);
            __m256 lo[4], hi[4];
            for (int j = 0; j < 4; ++j) {
                const __m256i e = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (q + 8 * j)));
                __m256i l = _mm256_and_si256(e, m15), h = _mm256_srli_epi32(e, 4);
                if (K5) {
                    const __m128i c0 = _mm_cvtsi32_si128(is), c1s = _mm_cvtsi32_si128(is + 1);
                    l = _mm256_or_si256(l, _mm256_slli_epi32(_mm256_and_si256(_mm256_srl_epi32(f[j], c0), one), 4));
                    h = _mm256_or_si256(h, _mm256_slli_epi32(_mm256_and_si256(_mm256_srl_epi32(f[j], c1s), one), 4));
                }
                lo[j] = _mm256_cvtepi32_ps(l), hi[j] = _mm256_cvtepi32_ps(h);
            }
            const __m256 d1 = _mm256_set1_ps(d * c1), d2 = _mm256_set1_ps(d * c2);
            const float n1 = mn * m1, n2 = mn * m2;
            const int col = b * 256 + is * 32, blk = col / 32;
            for (int k = 0; k < NT; ++k) {
                const __m256 tl = dot32(lo[0], lo[1], lo[2], lo[3], x[k] + col), th = dot32(hi[0], hi[1], hi[2], hi[3], x[k] + col + 32);
                acc[k][(is >> 1) & 1] = _mm256_add_ps(acc[k][(is >> 1) & 1], _mm256_fmadd_ps(d1, tl, _mm256_mul_ps(d2, th)));
                macc[k] += n1 * xs[k][blk] + n2 * xs[k][blk + 1];
            }
        }
    }
    for (int k = 0; k < NT; ++k) out[k] = hsum(_mm256_add_ps(acc[k][0], acc[k][1])) - macc[k];
}
// one row of n values in IQ4_NL blocks times NT rows of h
template <int NT>
FNX_CPUX_SIMD inline void row_iq4nl(const uint8_t* s, int n, const float* const* h, float* out) {
    const __m128i kv = _mm_loadu_si128((const __m128i*) KV_IQ4NL), m4 = _mm_set1_epi8(0x0F);
    __m256 acc[NT][2];
    for (int k = 0; k < NT; ++k) acc[k][0] = acc[k][1] = _mm256_setzero_ps();
    for (int b = 0; b < n / 32; ++b, s += 18) {
        const __m256 d = _mm256_set1_ps(half_s(s));
        const __m128i v = _mm_loadu_si128((const __m128i*) (s + 2));
        const __m128i lo = _mm_shuffle_epi8(kv, _mm_and_si128(v, m4)), hi = _mm_shuffle_epi8(kv, _mm_and_si128(_mm_srli_epi16(v, 4), m4));
        const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
        const __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
        for (int k = 0; k < NT; ++k) acc[k][b & 1] = _mm256_fmadd_ps(d, dot32(w0, w1, w2, w3, h[k] + b * 32), acc[k][b & 1]);
    }
    for (int k = 0; k < NT; ++k) out[k] = hsum(_mm256_add_ps(acc[k][0], acc[k][1]));
}
FNX_CPUX_SIMD void block_sums(const float* x, int n, float* out) {   // each 32 columns' sum
    for (int b = 0; b < n / 32; ++b, x += 32)
        out[b] = hsum(_mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(x + 8)), _mm256_add_ps(_mm256_loadu_ps(x + 16), _mm256_loadu_ps(x + 24))));
}
// Q5_1's fifth bits of columns c0 .. c0 + 7 (bits c0 .. c0 + 7 of qh), each at bit 4 of its lane
FNX_CPUX_SIMD inline __m256i q5_hi(uint32_t qh, unsigned c0) {
    return _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int) (qh >> c0)), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)), _mm256_set1_epi32(1)), 4);
}
// one row of n values in Q5_1 blocks (d, m, 32 high bits, 16 bytes of nibble pairs: value = d q + m, q 0..31) times NT rows of h;
// hb[k]: row k's sums of each 32 values
template <int NT>
FNX_CPUX_SIMD inline void row_q5_1(const uint8_t* s, int n, const float* const* h, const float* const* hb, float* out) {
    const __m256i m15 = _mm256_set1_epi32(15);
    __m256 acc[NT][2];
    float macc[NT];
    for (int k = 0; k < NT; ++k) acc[k][0] = acc[k][1] = _mm256_setzero_ps(), macc[k] = 0.f;
    for (int b = 0; b < n / 32; ++b, s += 24) {
        const float d = half_s(s), m = half_s(s + 2);
        uint32_t qh;
        std::memcpy(&qh, s + 4, 4);
        const __m256i e0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (s + 8))), e1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) (s + 16)));
        const __m256 w0 = _mm256_cvtepi32_ps(_mm256_or_si256(_mm256_and_si256(e0, m15), q5_hi(qh, 0))), w1 = _mm256_cvtepi32_ps(_mm256_or_si256(_mm256_and_si256(e1, m15), q5_hi(qh, 8)));
        const __m256 w2 = _mm256_cvtepi32_ps(_mm256_or_si256(_mm256_srli_epi32(e0, 4), q5_hi(qh, 16))), w3 = _mm256_cvtepi32_ps(_mm256_or_si256(_mm256_srli_epi32(e1, 4), q5_hi(qh, 24)));
        const __m256 dv = _mm256_set1_ps(d);
        for (int k = 0; k < NT; ++k) {
            acc[k][b & 1] = _mm256_fmadd_ps(dv, dot32(w0, w1, w2, w3, h[k] + b * 32), acc[k][b & 1]);
            macc[k] += m * hb[k][b];
        }
    }
    for (int k = 0; k < NT; ++k) out[k] = hsum(_mm256_add_ps(acc[k][0], acc[k][1])) + macc[k];
}
// one row of n values in Q8_0 blocks (d, 32 int8) times NT rows of h
template <int NT>
FNX_CPUX_SIMD inline void row_q8_0(const uint8_t* s, int n, const float* const* h, float* out) {
    __m256 acc[NT][2];
    for (int k = 0; k < NT; ++k) acc[k][0] = acc[k][1] = _mm256_setzero_ps();
    for (int b = 0; b < n / 32; ++b, s += 34) {
        const __m256 d = _mm256_set1_ps(half_s(s));
        const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (s + 2)))), w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (s + 10))));
        const __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (s + 18)))), w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) (s + 26))));
        for (int k = 0; k < NT; ++k) acc[k][b & 1] = _mm256_fmadd_ps(d, dot32(w0, w1, w2, w3, h[k] + b * 32), acc[k][b & 1]);
    }
    for (int k = 0; k < NT; ++k) out[k] = hsum(_mm256_add_ps(acc[k][0], acc[k][1]));
}
template <bool K5>
FNX_CPUX_SIMD void gu_chunk_kq(const Work& w, const Expert& e, int j0, int j1) {
    const size_t rb = row_bytes(K5 ? T_Q5_K : T_Q4_K, w.N);
    alignas(32) float xs[16][MAXN / 32];
    const float* x[16];
    const float* sp[16];
    for (int k = 0; k < e.n; ++k) {
        x[k] = w.x + (size_t) e.rows[k] * w.x_stride;
        sp[k] = xs[k];
        block_sums(x[k], w.N, xs[k]);
    }
    for (int j = j0; j < j1; ++j) {
        const uint8_t* g = e.gu + (size_t) j * rb;
        const uint8_t* u = e.gu + (size_t) (w.ffs + j) * rb;
        float gv[2], uv[2];
        int k = 0;
        for (; k + 2 <= e.n; k += 2) {
            row_kq<2, K5>(g, w.N, x + k, sp + k, gv);
            row_kq<2, K5>(u, w.N, x + k, sp + k, uv);
            e.h[(size_t) k * w.ffs + j] = silu(gv[0]) * uv[0];
            e.h[(size_t) (k + 1) * w.ffs + j] = silu(gv[1]) * uv[1];
        }
        if (k < e.n) {
            row_kq<1, K5>(g, w.N, x + k, sp + k, gv);
            row_kq<1, K5>(u, w.N, x + k, sp + k, uv);
            e.h[(size_t) k * w.ffs + j] = silu(gv[0]) * uv[0];
        }
    }
}
// down rows [i0, i1) of a 32-value-block type with a kernel: TY 20 IQ4_NL, 7 Q5_1, 8 Q8_0
template <int TY>
FNX_CPUX_SIMD void down_chunk_k(const Work& w, const Expert& e, int i0, int i1) {
    const size_t rb = row_bytes((uint32_t) TY, w.ffs);
    alignas(32) float hs[16][MAXN / 32];
    const float* h[16];
    const float* hb[16];
    for (int k = 0; k < e.n; ++k) {
        h[k] = e.h + (size_t) k * w.ffs;
        hb[k] = hs[k];
        if constexpr (TY == 7) block_sums(h[k], w.ffs, hs[k]);
    }
    for (int i = i0; i < i1; ++i) {
        const uint8_t* d = e.dn + (size_t) i * rb;
        float o[2];
        int k = 0;
        for (; k + 2 <= e.n; k += 2) {
            if constexpr (TY == 7) row_q5_1<2>(d, w.ffs, h + k, hb + k, o);
            else if constexpr (TY == 8) row_q8_0<2>(d, w.ffs, h + k, o);
            else row_iq4nl<2>(d, w.ffs, h + k, o);
            e.out[(size_t) k * w.out_stride + i] = o[0];
            e.out[(size_t) (k + 1) * w.out_stride + i] = o[1];
        }
        if (k < e.n) {
            if constexpr (TY == 7) row_q5_1<1>(d, w.ffs, h + k, hb + k, o);
            else if constexpr (TY == 8) row_q8_0<1>(d, w.ffs, h + k, o);
            else row_iq4nl<1>(d, w.ffs, h + k, o);
            e.out[(size_t) k * w.out_stride + i] = o[0];
        }
    }
}
// ---- FNX_CPU_EXACT's kernels: 16 rows at a time, a row a lane of two vectors, each lane's two chains the card's two chains of that row
FNX_CPUX_SIMD inline __m256 rn16(__m256 v) { return _mm256_cvtph_ps(_mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT)); }
FNX_CPUX_SIMD inline float rn16s(float f) { return _cvtsh_ss(_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT)); }
// fp16(w * x + c) of fp16 values held in floats, w * x exact in fp32 (as every product of two fp16 values): the fp32 sum s, its error e
// (TwoSum, each step exact), s rounded to odd where e is not 0 (moved one ulp toward the exact sum when its last bit is 0), then to fp16:
// the correctly rounded result, as the card's HFMA2 gives it. The subtractions as FMAs with 1.0 (the same exact result once rounded):
// Haswell adds on one port only, FMAs on two
FNX_CPUX_SIMD inline __m256 fma16(__m256 w, __m256 x, __m256 c) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 p = _mm256_mul_ps(w, x);
    const __m256 s = _mm256_add_ps(p, c);
    const __m256 bb = _mm256_fmsub_ps(s, one, p);
    const __m256 e = _mm256_add_ps(_mm256_fmsub_ps(p, one, _mm256_fmsub_ps(s, one, bb)), _mm256_fmsub_ps(c, one, bb));
    const __m256i si = _mm256_castps_si256(s), i1 = _mm256_set1_epi32(1);
    const __m256i inexact = _mm256_castps_si256(_mm256_cmp_ps(e, _mm256_setzero_ps(), _CMP_NEQ_OQ));   // not for NaN (an infinite s)
    const __m256i even = _mm256_cmpeq_epi32(_mm256_and_si256(si, i1), _mm256_setzero_si256());
    const __m256i away = _mm256_or_si256(_mm256_srai_epi32(_mm256_xor_si256(si, _mm256_castps_si256(e)), 31), i1);   // +1 away from 0, else -1
    return rn16(_mm256_castsi256_ps(_mm256_add_epi32(si, _mm256_and_si256(away, _mm256_and_si256(inexact, even)))));
}
// the fast form: fp16(fp32(w * x + c)), wrong only where that fp32 sum s was rounded (inexact) onto a point exactly halfway between two
// fp16 values. `tie` gets every such lane (and a few more), off the chain's dependency path: every fp16 midpoint (normal, subnormal,
// 65520) has at most 12 significant bits, so its float's low 12 bits are 0; s = p + c is exact when s - p == c and s - c == p (for the
// larger operand that subtraction is Fast2Sum's, exact). A run with a flagged lane is done again with fma16 (about one in twenty).
FNX_CPUX_SIMD inline __m256 fma16f(__m256 w, __m256 x, __m256 c, __m256& tie) {
    const __m256 p = _mm256_mul_ps(w, x), s = _mm256_add_ps(p, c);
    const __m256 few = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(s), _mm256_set1_epi32(0xfff)), _mm256_setzero_si256()));
    const __m256 inexact = _mm256_or_ps(_mm256_cmp_ps(_mm256_sub_ps(s, p), c, _CMP_NEQ_OQ), _mm256_cmp_ps(_mm256_sub_ps(s, c), p, _CMP_NEQ_OQ));
    tie = _mm256_or_ps(tie, _mm256_and_ps(few, inexact));
    return rn16(s);
}
// lanes where q a + b is exact in fp32 for every integer q < 32: a or b zero, or both normal fp16 values whose exponents differ by at
// most 7. (q a is a multiple of a's last bit below 2^(ea + 6), b of its own below 2^(eb + 1): the sum spans at most
// max(ea + 6, eb + 1) - min(ea, eb) + 11 <= 24 bits.) There fp16(fp32(q a + b)) is the card's one rounding
FNX_CPUX_SIMD inline bool decode_plain(__m256 a, __m256 b) {
    const __m256i m = _mm256_set1_epi32(0x7fffffff), ai = _mm256_and_si256(_mm256_castps_si256(a), m), bi = _mm256_and_si256(_mm256_castps_si256(b), m);
    const __m256i ea = _mm256_srli_epi32(ai, 23), eb = _mm256_srli_epi32(bi, 23), d = _mm256_sub_epi32(ea, eb), lim = _mm256_set1_epi32(112);
    const __m256i normal = _mm256_and_si256(_mm256_cmpgt_epi32(ea, lim), _mm256_cmpgt_epi32(eb, lim));   // fp32 exponent 113 = 2^-14
    const __m256i close = _mm256_and_si256(_mm256_cmpgt_epi32(_mm256_set1_epi32(8), d), _mm256_cmpgt_epi32(d, _mm256_set1_epi32(-8)));
    const __m256i zero = _mm256_or_si256(_mm256_cmpeq_epi32(ai, _mm256_setzero_si256()), _mm256_cmpeq_epi32(bi, _mm256_setzero_si256()));
    return _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_or_si256(zero, _mm256_and_si256(normal, close)))) == 0xff;
}
// 8 vectors' lanes transposed (rows to columns)
FNX_CPUX_SIMD inline void tr8_ps(__m256* v) {
    const __m256 t0 = _mm256_unpacklo_ps(v[0], v[1]), t1 = _mm256_unpackhi_ps(v[0], v[1]), t2 = _mm256_unpacklo_ps(v[2], v[3]), t3 = _mm256_unpackhi_ps(v[2], v[3]);
    const __m256 t4 = _mm256_unpacklo_ps(v[4], v[5]), t5 = _mm256_unpackhi_ps(v[4], v[5]), t6 = _mm256_unpacklo_ps(v[6], v[7]), t7 = _mm256_unpackhi_ps(v[6], v[7]);
    const __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44), u1 = _mm256_shuffle_ps(t0, t2, 0xEE), u2 = _mm256_shuffle_ps(t1, t3, 0x44), u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44), u5 = _mm256_shuffle_ps(t4, t6, 0xEE), u6 = _mm256_shuffle_ps(t5, t7, 0x44), u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    v[0] = _mm256_permute2f128_ps(u0, u4, 0x20), v[1] = _mm256_permute2f128_ps(u1, u5, 0x20), v[2] = _mm256_permute2f128_ps(u2, u6, 0x20), v[3] = _mm256_permute2f128_ps(u3, u7, 0x20);
    v[4] = _mm256_permute2f128_ps(u0, u4, 0x31), v[5] = _mm256_permute2f128_ps(u1, u5, 0x31), v[6] = _mm256_permute2f128_ps(u2, u6, 0x31), v[7] = _mm256_permute2f128_ps(u3, u7, 0x31);
}
// 8 rows of 32 bytes to 32 groups of 8 bytes, a group one byte position of the 8 rows (rows in order); tr_off: each position's group
FNX_CPUX_SIMD inline void tr8x32(const __m256i* v, uint8_t* out) {
    const __m256i a0 = _mm256_unpacklo_epi8(v[0], v[1]), a1 = _mm256_unpackhi_epi8(v[0], v[1]);
    const __m256i a2 = _mm256_unpacklo_epi8(v[2], v[3]), a3 = _mm256_unpackhi_epi8(v[2], v[3]);
    const __m256i a4 = _mm256_unpacklo_epi8(v[4], v[5]), a5 = _mm256_unpackhi_epi8(v[4], v[5]);
    const __m256i a6 = _mm256_unpacklo_epi8(v[6], v[7]), a7 = _mm256_unpackhi_epi8(v[6], v[7]);
    const __m256i b0 = _mm256_unpacklo_epi16(a0, a2), b1 = _mm256_unpackhi_epi16(a0, a2), b2 = _mm256_unpacklo_epi16(a1, a3), b3 = _mm256_unpackhi_epi16(a1, a3);
    const __m256i b4 = _mm256_unpacklo_epi16(a4, a6), b5 = _mm256_unpackhi_epi16(a4, a6), b6 = _mm256_unpacklo_epi16(a5, a7), b7 = _mm256_unpackhi_epi16(a5, a7);
    __m256i* o = reinterpret_cast<__m256i*>(out);
    _mm256_store_si256(o + 0, _mm256_unpacklo_epi32(b0, b4));
    _mm256_store_si256(o + 1, _mm256_unpackhi_epi32(b0, b4));
    _mm256_store_si256(o + 2, _mm256_unpacklo_epi32(b1, b5));
    _mm256_store_si256(o + 3, _mm256_unpackhi_epi32(b1, b5));
    _mm256_store_si256(o + 4, _mm256_unpacklo_epi32(b2, b6));
    _mm256_store_si256(o + 5, _mm256_unpackhi_epi32(b2, b6));
    _mm256_store_si256(o + 6, _mm256_unpacklo_epi32(b3, b7));
    _mm256_store_si256(o + 7, _mm256_unpackhi_epi32(b3, b7));
}
FNX_CPUX_SIMD __attribute__((noinline)) void tr_probe(uint8_t* map) {   // the transpose run on the positions' own numbers
    alignas(32) uint8_t id[32], out[256];
    for (int l = 0; l < 32; ++l) id[l] = (uint8_t) l;
    __m256i v[8];
    for (auto& x : v) x = _mm256_load_si256(reinterpret_cast<const __m256i*>(id));
    tr8x32(v, out);
    for (int g = 0; g < 32; ++g) map[out[8 * g]] = (uint8_t) (8 * g);
}
const uint8_t* tr_off() {
    static const std::array<uint8_t, 32> m = [] {
        std::array<uint8_t, 32> a{};
        tr_probe(a.data());
        return a;
    }();
    return m.data();
}
FNX_CPUX_SIMD inline __m256 codes8(const uint8_t* g, bool sgn) {   // a group's 8 bytes as floats
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(g));
    return _mm256_cvtepi32_ps(sgn ? _mm256_cvtepi8_epi32(b) : _mm256_cvtepu8_epi32(b));
}
// Q5_1's 32 fifth bits: byte l is 16 where bit l of qh is set
FNX_CPUX_SIMD inline __m256i bits16(uint32_t qh) {
    const __m256i sh = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bm = _mm256_set1_epi64x(0x8040201008040201LL);
    const __m256i b = _mm256_shuffle_epi8(_mm256_set1_epi32((int) qh), sh);
    return _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(b, bm), bm), _mm256_set1_epi8(16));
}
// a run's 32 weights of 16 rows (W[l][half]) times one token's 32 values: the card's chains (even values, odd values, from +0), folded.
// The fast form first; when any lane met a midpoint, the run again with fma16
FNX_CPUX_SIMD inline void chain32(const __m256 (*W)[2], const float* x, __m256* acc) {
    __m256 lo0 = _mm256_setzero_ps(), lo1 = lo0, hi0 = lo0, hi1 = lo0, tie = lo0;
    for (int t = 0; t < 16; ++t) {
        const __m256 xe = _mm256_broadcast_ss(x + 2 * t), xo = _mm256_broadcast_ss(x + 2 * t + 1);
        lo0 = fma16f(W[2 * t][0], xe, lo0, tie);
        lo1 = fma16f(W[2 * t][1], xe, lo1, tie);
        hi0 = fma16f(W[2 * t + 1][0], xo, hi0, tie);
        hi1 = fma16f(W[2 * t + 1][1], xo, hi1, tie);
    }
    if (__builtin_expect(_mm256_movemask_ps(tie) != 0, 0)) {
        lo0 = lo1 = hi0 = hi1 = _mm256_setzero_ps();
        for (int t = 0; t < 16; ++t) {
            const __m256 xe = _mm256_broadcast_ss(x + 2 * t), xo = _mm256_broadcast_ss(x + 2 * t + 1);
            lo0 = fma16(W[2 * t][0], xe, lo0);
            lo1 = fma16(W[2 * t][1], xe, lo1);
            hi0 = fma16(W[2 * t + 1][0], xo, hi0);
            hi1 = fma16(W[2 * t + 1][1], xo, hi1);
        }
    }
    acc[0] = _mm256_add_ps(acc[0], _mm256_add_ps(lo0, hi0));
    acc[1] = _mm256_add_ps(acc[1], _mm256_add_ps(lo1, hi1));
}
// weights fp16(q a + b) of a run, 16 rows: one rounding of the fp32 sum where that sum is exact for every q (decode_plain), else fma16
FNX_CPUX_SIMD inline void decode_fma(const uint8_t (*tb)[256], const uint8_t* off, const __m256* a, const __m256* b, __m256 (*W)[2]) {
    for (int hh = 0; hh < 2; ++hh) {
        if (decode_plain(a[hh], b[hh])) {
            for (int l = 0; l < 32; ++l) W[l][hh] = rn16(_mm256_fmadd_ps(codes8(tb[hh] + off[l], false), a[hh], b[hh]));
            continue;
        }
        __m256 tie = _mm256_setzero_ps();   // the fast form and its test, the exact form where it flags
        for (int l = 0; l < 32; ++l) W[l][hh] = fma16f(codes8(tb[hh] + off[l], false), a[hh], b[hh], tie);
        if (__builtin_expect(_mm256_movemask_ps(tie) != 0, 0))
            for (int l = 0; l < 32; ++l) W[l][hh] = fma16(codes8(tb[hh] + off[l], false), a[hh], b[hh]);
    }
}
// 16 gate|up rows (Q4_K, or Q5_K with K5) of n values times nt tokens: res[k * 16 + r]
template <bool K5>
FNX_CPUX_SIMD void gu16_exact(const uint8_t* const* rp, int n, int lanes, int nt, const float* const* x, float* res) {
    const int bb = K5 ? 176 : 144, nruns = n / 32, ngr = (nruns + 7) / 8;
    const uint8_t* off = tr_off();
    const __m256i m4 = _mm256_set1_epi8(0x0F), one8 = _mm256_set1_epi8(1);
    alignas(32) uint8_t tb[2][256];
    __m256 W[32][2], part[8][16][2], As[2][8], Bs[2][8];
    for (int p = 0; p < lanes; ++p) {
        __m256 acc[16][2];
        for (int k = 0; k < nt; ++k) acc[k][0] = acc[k][1] = _mm256_setzero_ps();
        for (int g = p * ngr / lanes; g < (p + 1) * ngr / lanes; ++g) {
            // the 8 runs' a = fp16(d sc) and b = fp16(-dmin m) of each row (ggml's unpacking of the 12 scale bytes), then by run
            for (int hh = 0; hh < 2; ++hh) {
                for (int r = 0; r < 8; ++r) {
                    const uint8_t* blk = rp[8 * hh + r] + (size_t) g * bb;
                    uint32_t u[4];
                    std::memcpy(u, blk + 4, 12);
                    u[3] = ((u[2] >> 4) & 0x0f0f0f0fu) | (((u[1] >> 6) & 0x03030303u) << 4);
                    const uint32_t ux = u[1] & 0x3f3f3f3fu;
                    u[1] = (u[2] & 0x0f0f0f0fu) | (((u[0] >> 6) & 0x03030303u) << 4);
                    u[2] = ux;
                    u[0] &= 0x3f3f3f3fu;
                    const __m256 sc = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(u))));
                    const __m256 mn = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(u + 2))));
                    As[hh][r] = rn16(_mm256_mul_ps(_mm256_set1_ps(half_s(blk)), sc));   // the products exact, one rounding each
                    Bs[hh][r] = rn16(_mm256_mul_ps(_mm256_set1_ps(-half_s(blk + 2)), mn));
                }
                tr8_ps(As[hh]);
                tr8_ps(Bs[hh]);
            }
            for (int ib = 0; ib < 8; ++ib) {
                const __m128i cq = _mm_cvtsi32_si128(4 * (ib & 1)), ch = _mm_cvtsi32_si128(ib);
                for (int hh = 0; hh < 2; ++hh) {
                    __m256i v[8];
                    for (int r = 0; r < 8; ++r) {
                        const uint8_t* blk = rp[8 * hh + r] + (size_t) g * bb;
                        __m256i c = _mm256_and_si256(_mm256_srl_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + (K5 ? 48 : 16) + 32 * (ib >> 1))), cq), m4);
                        if (K5)
                            c = _mm256_or_si256(c, _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 16)), ch), one8), 4));
                        v[r] = c;
                    }
                    tr8x32(v, tb[hh]);
                }
                const __m256 a[2] = {As[0][ib], As[1][ib]}, b[2] = {Bs[0][ib], Bs[1][ib]};
                decode_fma(tb, off, a, b, W);
                const int col = g * 256 + ib * 32;
                for (int k = 0; k < nt; ++k) chain32(W, x[k] + col, acc[k]);
            }
        }
        for (int k = 0; k < nt; ++k) part[p][k][0] = acc[k][0], part[p][k][1] = acc[k][1];
    }
    for (int o = 1; o < lanes; o <<= 1)   // the warp's xor shuffles: lane 0 ends with ((s0 + s1) + (s2 + s3)) + ...
        for (int p = 0; p + o < lanes; p += 2 * o)
            for (int k = 0; k < nt; ++k)
                part[p][k][0] = _mm256_add_ps(part[p][k][0], part[p + o][k][0]), part[p][k][1] = _mm256_add_ps(part[p][k][1], part[p + o][k][1]);
    for (int k = 0; k < nt; ++k) _mm256_storeu_ps(res + 16 * k, part[0][k][0]), _mm256_storeu_ps(res + 16 * k + 8, part[0][k][1]);
}
// 16 down rows (TY 20 IQ4_NL, 7 Q5_1, 8 Q8_0) of n values times nt rows of h: res[k * 16 + r]
template <int TY>
FNX_CPUX_SIMD void dn16_exact(const uint8_t* const* rp, int n, int nt, const float* const* h, float* res) {
    const int bb = TY == 7 ? 24 : TY == 8 ? 34 : 18;
    const uint8_t* off = tr_off();
    const __m256i m4 = _mm256_set1_epi8(0x0F), kv = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(KV_IQ4NL)));
    alignas(32) float D[16], M[16];
    alignas(32) uint8_t tb[2][256];
    __m256 W[32][2], acc[16][2];
    for (int k = 0; k < nt; ++k) acc[k][0] = acc[k][1] = _mm256_setzero_ps();
    for (int b = 0; b < n / 32; ++b) {
        for (int hh = 0; hh < 2; ++hh) {
            __m256i v[8];
            for (int r = 0; r < 8; ++r) {
                const int rr = 8 * hh + r;
                const uint8_t* blk = rp[rr] + (size_t) b * bb;
                D[rr] = half_s(blk);
                if constexpr (TY == 8) v[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 2));
                else {
                    const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + (TY == 7 ? 8 : 2)));
                    __m256i c = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q), m4);   // values 0-15: low nibbles, 16-31: high
                    if constexpr (TY == 20) c = _mm256_shuffle_epi8(kv, c);
                    else {
                        uint32_t qh;
                        std::memcpy(&qh, blk + 4, 4);
                        c = _mm256_or_si256(c, bits16(qh));
                        M[rr] = half_s(blk + 2);
                    }
                    v[r] = c;
                }
            }
            tr8x32(v, tb[hh]);
        }
        const __m256 d[2] = {_mm256_load_ps(D), _mm256_load_ps(D + 8)};
        if constexpr (TY == 7) {   // fp16(q d + m)
            const __m256 m[2] = {_mm256_load_ps(M), _mm256_load_ps(M + 8)};
            decode_fma(tb, off, d, m, W);
        } else
            for (int l = 0; l < 32; ++l) {   // fp16(v d): the product exact, one rounding
                W[l][0] = rn16(_mm256_mul_ps(codes8(tb[0] + off[l], true), d[0]));
                W[l][1] = rn16(_mm256_mul_ps(codes8(tb[1] + off[l], true), d[1]));
            }
        for (int k = 0; k < nt; ++k) chain32(W, h[k] + b * 32, acc[k]);
    }
    for (int k = 0; k < nt; ++k) _mm256_storeu_ps(res + 16 * k, acc[k][0]), _mm256_storeu_ps(res + 16 * k + 8, acc[k][1]);
}
FNX_CPUX_SIMD void gu_chunk_exact_simd(const Work& w, const Expert& e, int j0, int j1) {
    const size_t rb = row_bytes(w.gu_type, w.N);
    const float* x[16];
    for (int k = 0; k < e.n; ++k) x[k] = w.x + (size_t) e.rows[k] * w.x_stride;
    alignas(32) float res[16 * 16];
    for (int part = 0; part < 2; ++part)   // the gate rows, then the up rows
        for (int j = j0; j < j1; j += 16) {
            const uint8_t* rp[16];
            for (int r = 0; r < 16; ++r) rp[r] = e.gu + (size_t) (part * w.ffs + std::min(j + r, j1 - 1)) * rb;
            if (w.gu_type == T_Q5_K) gu16_exact<true>(rp, w.N, w.gu_lanes, e.n, x, res);
            else gu16_exact<false>(rp, w.N, w.gu_lanes, e.n, x, res);
            for (int k = 0; k < e.n; ++k)
                for (int r = 0; r < 16 && j + r < j1; ++r) e.gu_out[(size_t) k * 2 * w.ffs + (size_t) part * w.ffs + j + r] = res[16 * k + r];
        }
}
FNX_CPUX_SIMD void down_chunk_exact_simd(const Work& w, const Expert& e, int i0, int i1) {
    const size_t rb = row_bytes(w.dn_type, w.ffs);
    const float* h[16];
    for (int k = 0; k < e.n; ++k) h[k] = e.h + (size_t) k * w.ffs;
    alignas(32) float res[16 * 16];
    for (int i = i0; i < i1; i += 16) {
        const uint8_t* rp[16];
        for (int r = 0; r < 16; ++r) rp[r] = e.dn + (size_t) std::min(i + r, i1 - 1) * rb;
        if (w.dn_type == T_Q5_1) dn16_exact<7>(rp, w.ffs, e.n, h, res);
        else if (w.dn_type == T_Q8_0) dn16_exact<8>(rp, w.ffs, e.n, h, res);
        else dn16_exact<20>(rp, w.ffs, e.n, h, res);
        for (int k = 0; k < e.n; ++k)
            for (int r = 0; r < 16 && i + r < i1; ++r) e.out[(size_t) k * w.out_stride + i + r] = res[16 * k + r];
    }
}
#endif

namespace {
// without AVX2: the oracle's steps, a row at a time (slow; exact all the same)
void gu_chunk_exact_ref(const Work& w, const Expert& e, int j0, int j1) {
    const size_t rb = row_bytes(w.gu_type, w.N);
    std::vector<uint16_t> x16((size_t) e.n * w.N);
    for (int k = 0; k < e.n; ++k)
        for (int c = 0; c < w.N; ++c) x16[(size_t) k * w.N + c] = f16_from_f32_ref(w.x[(size_t) e.rows[k] * w.x_stride + c]);
    for (int part = 0; part < 2; ++part)
        for (int j = j0; j < j1; ++j)
            for (int k = 0; k < e.n; ++k)
                e.gu_out[(size_t) k * 2 * w.ffs + (size_t) part * w.ffs + j] =
                    gu_row_card_ref(w.gu_type, e.gu + (size_t) (part * w.ffs + j) * rb, w.N, x16.data() + (size_t) k * w.N, w.gu_lanes);
}
void down_chunk_exact_ref(const Work& w, const Expert& e, int i0, int i1) {
    const size_t rb = row_bytes(w.dn_type, w.ffs);
    std::vector<uint16_t> h16((size_t) e.n * w.ffs);
    for (size_t i = 0; i < h16.size(); ++i) h16[i] = f16_from_f32_ref(e.h[i]);
    for (int i = i0; i < i1; ++i)
        for (int k = 0; k < e.n; ++k)
            e.out[(size_t) k * w.out_stride + i] = dn_row_card_ref(w.dn_type, e.dn + (size_t) i * rb, w.ffs, h16.data() + (size_t) k * w.ffs);
}
}  // namespace

bool fma16_x8(const float* w, const float* x, const float* c, float* out) {
#if FNX_CPUX_X86
    if (!cpu_simd()) return false;
    struct S {
        FNX_CPUX_SIMD static void run(const float* w, const float* x, const float* c, float* out) {
            _mm256_storeu_ps(out, fma16(_mm256_loadu_ps(w), _mm256_loadu_ps(x), _mm256_loadu_ps(c)));
        }
    };
    S::run(w, x, c, out);
    return true;
#else
    (void) w, (void) x, (void) c, (void) out;
    return false;
#endif
}

void gu_chunk(const Work& w, const Expert& e, int j0, int j1) {
    if (w.exact) {
#if FNX_CPUX_X86
        if (cpu_simd()) return gu_chunk_exact_simd(w, e, j0, j1);
#endif
        return gu_chunk_exact_ref(w, e, j0, j1);
    }
#if FNX_CPUX_X86
    if (cpu_simd() && w.gu_type == T_Q4_K) return gu_chunk_kq<false>(w, e, j0, j1);
    if (cpu_simd() && w.gu_type == T_Q5_K) return gu_chunk_kq<true>(w, e, j0, j1);
#endif
    alignas(64) float g[MAXN], u[MAXN];
    const size_t rb = row_bytes(w.gu_type, w.N);
    for (int j = j0; j < j1; ++j) {
        decode_row(w.gu_type, e.gu + (size_t) j * rb, w.N, g);
        decode_row(w.gu_type, e.gu + (size_t) (w.ffs + j) * rb, w.N, u);
        for (int k = 0; k < e.n; ++k) {
            const float* x = w.x + (size_t) e.rows[k] * w.x_stride;
            e.h[(size_t) k * w.ffs + j] = silu(dot(g, x, w.N)) * dot(u, x, w.N);
        }
    }
}
void down_chunk(const Work& w, const Expert& e, int i0, int i1) {
    if (w.exact) {
#if FNX_CPUX_X86
        if (cpu_simd()) return down_chunk_exact_simd(w, e, i0, i1);
#endif
        return down_chunk_exact_ref(w, e, i0, i1);
    }
#if FNX_CPUX_X86
    if (cpu_simd() && w.dn_type == T_IQ4_NL) return down_chunk_k<20>(w, e, i0, i1);
    if (cpu_simd() && w.dn_type == T_Q5_1) return down_chunk_k<7>(w, e, i0, i1);
    if (cpu_simd() && w.dn_type == T_Q8_0) return down_chunk_k<8>(w, e, i0, i1);
#endif
    alignas(64) float d[MAXN];
    const size_t rb = row_bytes(w.dn_type, w.ffs);
    for (int i = i0; i < i1; ++i) {
        decode_row(w.dn_type, e.dn + (size_t) i * rb, w.ffs, d);
        for (int k = 0; k < e.n; ++k) e.out[(size_t) k * w.out_stride + i] = dot(d, e.h + (size_t) k * w.ffs, w.ffs);
    }
}

// ---- the pool
namespace {
constexpr int GU_CHUNK = 32, DN_CHUNK = 256;
inline void relax() {
#if FNX_CPUX_X86
    _mm_pause();
#endif
}
}  // namespace

struct Pool::Impl {
    std::vector<std::thread> th;
    std::atomic<bool> stop{false};
    // a round: its number in the ticket's high half, the next chunk in its low half. A round's fields below are written before its
    // ticket and stay until every chunk of it is done, so whoever takes a chunk with the ticket reads that round's fields.
    // limit: the round's number and its count of chunks in ONE word, written before the ticket. The count was a word of its own at
    // first: a worker that had just finished a round's last chunk (or came late to it) then read the old round's spent ticket, and
    // after it the NEXT round's larger count, which the host thread had stored in between; if its compare-and-swap also came before
    // the new ticket, it took a chunk beyond the old round, computed that chunk of the new round and counted it done there, and the
    // new round ended one chunk early: its rows read while a worker was still writing them. The rows' round (a short chunk a row,
    // windows of more than 4 rows) before the gate|up round is where it happened: about one such list in six thousand on one card of
    // the two here, none on the other
    std::atomic<uint64_t> ticket{0}, limit{0};
    std::atomic<int> done{0};
    std::atomic<unsigned long long> late{0};   // workers that met a spent ticket and the next round's count: turned away
    const Work* w = nullptr;
    const Pool::Rows* rows = nullptr;
    int phase = 0, per = 0;   // phase 0: the window's rows; 1: gate|up chunks; 2: down chunks
    uint32_t round = 0;
    std::atomic<int> asleep{0};
    std::mutex mu;
    std::condition_variable cv;

    void chunk(int i) const {
        if (phase == 0) {   // row i: halves to floats
            f16_to_f32(rows->x16 + (size_t) i * w->N, rows->xf + (size_t) i * w->N, (size_t) w->N);
            return;
        }
        const Expert& e = w->ex[i / per];
        const int c = i % per;
        if (phase == 1) gu_chunk(*w, e, c * GU_CHUNK, std::min(w->ffs, (c + 1) * GU_CHUNK));
        else down_chunk(*w, e, c * DN_CHUNK, std::min(w->N, (c + 1) * DN_CHUNK));
    }
    // take chunks of round r until none is left (or the round is over); true when it did any
    bool take(uint32_t r) {
        bool any = false;
        uint64_t t = ticket.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (t >> 32) != r) return any;
            const uint64_t lim = limit.load(std::memory_order_acquire);
            const int i = (int) (uint32_t) t;
            if ((uint32_t) (lim >> 32) != r) {   // round r is over, the next one is being set up
                if (i < (int) (uint32_t) lim) late.fetch_add(1, std::memory_order_relaxed);
                return any;
            }
            if (i >= (int) (uint32_t) lim) return any;
            if (!ticket.compare_exchange_weak(t, t + 1, std::memory_order_acq_rel, std::memory_order_acquire)) continue;
            chunk(i);
            done.fetch_add(1, std::memory_order_release);
            any = true;
            t = ticket.load(std::memory_order_acquire);
        }
    }
    void loop(int cpu) {
#if FNX_CPUX_X86
        if (cpu >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            pthread_setaffinity_np(pthread_self(), sizeof set, &set);
        }
#else
        (void) cpu;
#endif
        uint32_t seen = 0;
        auto last = std::chrono::steady_clock::now();
        while (!stop.load(std::memory_order_relaxed)) {
            const uint32_t r = (uint32_t) (ticket.load(std::memory_order_acquire) >> 32);
            if (r != seen) {
                seen = r;
                take(r);
                last = std::chrono::steady_clock::now();
                continue;
            }
            const auto idle = std::chrono::steady_clock::now() - last;
            if (idle > std::chrono::seconds(1)) {   // block until the next round
                std::unique_lock<std::mutex> g(mu);
                asleep.fetch_add(1, std::memory_order_acq_rel);
                cv.wait(g, [&] { return stop.load() || (uint32_t) (ticket.load(std::memory_order_acquire) >> 32) != seen; });
                asleep.fetch_sub(1, std::memory_order_acq_rel);
                last = std::chrono::steady_clock::now();
            } else if (idle > std::chrono::milliseconds(20)) std::this_thread::sleep_for(std::chrono::microseconds(100));
            else relax();
        }
    }
    void phase_run(const Work& work, int ph, const Pool::Rows* r = nullptr) {
        w = &work;
        rows = r;
        phase = ph;
        per = ph == 1 ? (work.ffs + GU_CHUNK - 1) / GU_CHUNK : ph == 2 ? (work.N + DN_CHUNK - 1) / DN_CHUNK : 1;
        const int n = ph == 0 ? r->T : work.m * per;
        if (n <= 0) return;
        done.store(0, std::memory_order_release);
        ++round;
        limit.store((uint64_t) round << 32 | (uint32_t) n, std::memory_order_release);
        ticket.store((uint64_t) round << 32, std::memory_order_release);
        if (asleep.load(std::memory_order_acquire) > 0) {
            std::lock_guard<std::mutex> g(mu);
            cv.notify_all();
        }
        take(round);
        while (done.load(std::memory_order_acquire) < n) relax();
    }
};

Pool::Pool(int threads, const std::vector<int>& cpus) : p_(new Impl) {
    for (int i = 0; i < threads; ++i) p_->th.emplace_back([this, i, cpus] { p_->loop(cpus.empty() ? -1 : cpus[(size_t) i % cpus.size()]); });
}
Pool::~Pool() {
    p_->stop.store(true);
    {
        std::lock_guard<std::mutex> g(p_->mu);
        p_->cv.notify_all();
    }
    for (auto& t : p_->th) t.join();
}
int Pool::threads() const { return (int) p_->th.size(); }
unsigned long long Pool::late() { return p_->late.exchange(0, std::memory_order_relaxed); }
void Pool::run_gu(Work& w, const Rows* rows) {
    if (w.m <= 0) return;
    if (rows && rows->x16) {
        if (rows->T <= 4) {   // a few rows: here, a round of the pool costs more than they do
            p_->w = &w, p_->rows = rows, p_->phase = 0;
            for (int t = 0; t < rows->T; ++t) p_->chunk(t);
        } else p_->phase_run(w, 0, rows);
        w.x = rows->xf, w.x_stride = (size_t) w.N;
    }
    p_->phase_run(w, 1);
}
void Pool::run_down(Work& w) {
    if (w.m <= 0) return;
    p_->phase_run(w, 2);
}
void Pool::run(Work& w, const Rows* rows) {
    run_gu(w, rows);
    run_down(w);
}

std::vector<int> node_cores(int node) {
    std::vector<int> out;
    std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    std::string s;
    if (!f || !std::getline(f, s)) return out;
    std::vector<int> cpus;
    size_t i = 0;
    while (i < s.size()) {
        const size_t j = s.find(',', i);
        const std::string part = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str()), b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
        for (int c = a; c <= b; ++c) cpus.push_back(c);
        if (j == std::string::npos) break;
        i = j + 1;
    }
    std::set<std::string> seen;   // a core by its siblings' list: the first CPU that names it stands for it
    for (int c : cpus) {
        std::ifstream t("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
        std::string sib;
        if (!t || !std::getline(t, sib)) sib = std::to_string(c);
        if (seen.insert(sib).second) out.push_back(c);
    }
    return out;
}

}  // namespace cpux
}  // namespace fnx
