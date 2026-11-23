// cpu_experts.cpp: see cpu_experts.hpp. Compiled as plain C++ (no CUDA); the AVX2 kernels are target-attributed functions chosen at
// run time, so the file builds with the project's flags on any x86-64 and falls back to the scalar steps elsewhere.
#include "cpu_experts.hpp"

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
#endif

void gu_chunk(const Work& w, const Expert& e, int j0, int j1) {
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
void Pool::run(Work& w, const Rows* rows) {
    if (w.m <= 0) return;
    if (rows && rows->x16) {
        if (rows->T <= 4) {   // a few rows: here, a round of the pool costs more than they do
            p_->w = &w, p_->rows = rows, p_->phase = 0;
            for (int t = 0; t < rows->T; ++t) p_->chunk(t);
        } else p_->phase_run(w, 0, rows);
        w.x = rows->xf, w.x_stride = (size_t) w.N;
    }
    p_->phase_run(w, 1);
    p_->phase_run(w, 2);
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
