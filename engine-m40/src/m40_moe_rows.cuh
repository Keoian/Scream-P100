// m40_moe_rows.cuh: a decode window's routed experts on Maxwell (sm_52, ai-streznik-m40), a row a lane. Maxwell has no f16 arithmetic,
// converts at a quarter of its FMA rate, and its shared memory serves 128 bytes a clock to the whole SM, so (m40_rows_test, 2026-11-06:
// the first version, f32 tables of 16-byte entries and the rows read from global memory in the inner loop, made 18-90 GB/s):
//   - the job's window rows are f32 and staged, a group's worth at a time, in shared memory, where every lane reads the same address;
//   - the grids stay ggml's bytes (8 or 4 bytes an entry, in shared memory), a byte becomes a float with a byte permute into 2^23 and
//     one add, and a sign is a shift and an xor of the value's bit (no sign tables);
//   - a run's grid products are summed unscaled per token and its scale goes on once (twice for the types with a scale a half run).
// Types: the M40's GSQ IQ3_XXS (gate|up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S; down IQ4_NL and Strata's Q2_0) and the draft layer's Q4_0.
#pragma once

#include "env.hpp"

namespace fnx {
namespace {

using namespace strata::kernels::p100f::iqg;   // h2f, the group sizes (kGrBytes, kRunsGr, kSU); g16, g8, group_load from p100_moe_rows.cuh

constexpr int kM40T = 128;   // threads (rows) a block

// the type's table in shared memory, in 16-byte units: ggml's grids as they are (8 bytes or 4 an entry), IQ4_NL's 16 values as floats
template <int TY> struct M40TabU { static constexpr int n = 0; };
template <> struct M40TabU<16> { static constexpr int n = 256 * 8 / 16; };
template <> struct M40TabU<17> { static constexpr int n = 512 * 8 / 16; };
template <> struct M40TabU<22> { static constexpr int n = 1024 * 8 / 16; };
template <> struct M40TabU<18> { static constexpr int n = 256 * 4 / 16; };
template <> struct M40TabU<21> { static constexpr int n = 512 * 4 / 16; };
template <> struct M40TabU<20> { static constexpr int n = 4; };

template <int TY, int NT = kM40T>
__device__ __forceinline__ void m40_stage(uint4* tab, const int tid) {
    auto cp = [&](const void* from, int n) {   // 8 bytes a load: ggml's tables are 8-byte aligned, not 16
        for (int i = tid; i < 2 * n; i += NT) reinterpret_cast<uint2*>(tab)[i] = __ldg(reinterpret_cast<const uint2*>(from) + i);
    };
    if constexpr (TY == 16) cp(iq2xxs_grid, M40TabU<16>::n);
    else if constexpr (TY == 17) cp(iq2xs_grid, M40TabU<17>::n);
    else if constexpr (TY == 22) cp(iq2s_grid, M40TabU<22>::n);
    else if constexpr (TY == 18) cp(iq3xxs_grid, M40TabU<18>::n);
    else if constexpr (TY == 21) cp(iq3s_grid, M40TabU<21>::n);
    else if constexpr (TY == 20) {
        if (tid < 16) reinterpret_cast<float*>(tab)[tid] = float(kvalues_iq4nl[tid]);
    }
}

// byte k of w as a float: the byte into the mantissa of 2^23, then 2^23 off (a permute and an add; I2F is quarter rate)
__device__ __forceinline__ float m40_byte(const uint32_t w, const int k) { return __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7440u | k)) - 8388608.0f; }
// a small integer as a float the same way
__device__ __forceinline__ float m40_i(const uint32_t u) { return __uint_as_float(0x4B000000u | u) - 8388608.0f; }
// 8 grid bytes (g.x: values 0-3, g.y: 4-7) with value j negative where bit j of s8 is set
__device__ __forceinline__ void m40_g8(const uint2 g, const uint32_t s8, float (&v)[8]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = __uint_as_float(__float_as_uint(m40_byte(g.x, i)) ^ ((s8 << (31 - i)) & 0x80000000u));
        v[4 + i] = __uint_as_float(__float_as_uint(m40_byte(g.y, i)) ^ ((s8 << (27 - i)) & 0x80000000u));
    }
}
// ksigns_iq2xs: 7 sign bits and an eighth that makes their count even
__device__ __forceinline__ uint32_t m40_s7(const uint32_t s7) { return s7 | ((__popc(s7) & 1u) << 7); }

// the staged rows (xs: [token][XS] floats) at value `at` of the chunk times v, into each token's partial sum
template <int TT, int XS>
__device__ __forceinline__ void m40_mac(const float (&v)[8], const float* xs, const int at, const int n, float (&p)[TT]) {
#pragma unroll
    for (int k = 0; k < TT; ++k) {
        if (k >= n) continue;
        const float4 a = *reinterpret_cast<const float4*>(xs + k * XS + at), b = *reinterpret_cast<const float4*>(xs + k * XS + at + 4);
        float q0 = p[k], q1 = 0.0f;   // two chains: half the dependent adds
        q0 = fmaf(v[0], a.x, q0); q1 = fmaf(v[1], a.y, q1); q0 = fmaf(v[2], a.z, q0); q1 = fmaf(v[3], a.w, q1);
        q0 = fmaf(v[4], b.x, q0); q1 = fmaf(v[5], b.y, q1); q0 = fmaf(v[6], b.z, q0); q1 = fmaf(v[7], b.w, q1);
        p[k] = q0 + q1;
    }
}
template <int TT>
__device__ __forceinline__ void m40_scale(const float d, float (&p)[TT], float (&s)[TT], const int n) {
#pragma unroll
    for (int k = 0; k < TT; ++k)
        if (k < n) s[k] = fmaf(d, p[k], s[k]), p[k] = 0.0f;
}

// run IB of the group W (its first byte at word 0): ggml's dequantize_row_* in f32, piece by piece of 8 values; xs holds the chunk
template <int TY, int TT, int XS, int IB>
__device__ __forceinline__ void m40_run(const uint32_t (&W)[kGrW<TY>], const uint4* tab, const float* xs, const int n, float (&s)[TT]) {
    float p[TT];
#pragma unroll
    for (int k = 0; k < TT; ++k) p[k] = 0.0f;
    constexpr int AT = 32 * IB;   // the run's first value in the chunk
    float v[8];
    if constexpr (TY == 16) {   // IQ2_XXS: 4 grid bytes, then 4 x 7 sign bits and the scale's 4
        const uint2* grid = reinterpret_cast<const uint2*>(tab);
        const uint32_t gi = g16<2 + 8 * IB>(W) | (g16<4 + 8 * IB>(W) << 16), aux = g16<6 + 8 * IB>(W) | (g16<8 + 8 * IB>(W) << 16);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            m40_g8(grid[(gi >> (8 * j)) & 255], m40_s7((aux >> (7 * j)) & 127), v);
            m40_mac<TT, XS>(v, xs, AT + 8 * j, n, p);
        }
        m40_scale<TT>(h2f(uint16_t(g16<0>(W))) * (0.5f + float(aux >> 28)) * 0.25f, p, s, n);
    } else if constexpr (TY == 17) {   // IQ2_XS: 4 x (9-bit grid index, 7 sign bits), a scale nibble a half run
        const uint2* grid = reinterpret_cast<const uint2*>(tab);
        const uint32_t q01 = g16<2 + 8 * IB>(W) | (g16<4 + 8 * IB>(W) << 16), q23 = g16<6 + 8 * IB>(W) | (g16<8 + 8 * IB>(W) << 16);
        const uint32_t sc = g8<66 + IB>(W);
        const float d = h2f(uint16_t(g16<0>(W))) * 0.25f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = ((j < 2 ? q01 : q23) >> (16 * (j & 1))) & 0xffff;
            m40_g8(grid[q & 511], m40_s7(q >> 9), v);
            m40_mac<TT, XS>(v, xs, AT + 8 * j, n, p);
            if (j == 1) m40_scale<TT>(d * (0.5f + float(sc & 15)), p, s, n);
        }
        m40_scale<TT>(d * (0.5f + float(sc >> 4)), p, s, n);
    } else if constexpr (TY == 22) {   // IQ2_S: grid low bytes, sign bytes, 2 high bits a piece in qh, a scale nibble a half run
        const uint2* grid = reinterpret_cast<const uint2*>(tab);
        const uint32_t lo = g16<2 + 4 * IB>(W) | (g16<4 + 4 * IB>(W) << 16), sb = g16<34 + 4 * IB>(W) | (g16<36 + 4 * IB>(W) << 16);
        const uint32_t qh = g8<66 + IB>(W), sc = g8<74 + IB>(W);
        const float d = h2f(uint16_t(g16<0>(W))) * 0.25f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            m40_g8(grid[((lo >> (8 * j)) & 255) | ((qh << (8 - 2 * j)) & 0x300u)], (sb >> (8 * j)) & 255, v);
            m40_mac<TT, XS>(v, xs, AT + 8 * j, n, p);
            if (j == 1) m40_scale<TT>(d * (0.5f + float(sc & 15)), p, s, n);
        }
        m40_scale<TT>(d * (0.5f + float(sc >> 4)), p, s, n);
    } else if constexpr (TY == 18) {   // IQ3_XXS: 8 grid bytes (4 values each), then the signs and scale word
        const uint32_t* grid = reinterpret_cast<const uint32_t*>(tab);
        const uint32_t w0 = g16<2 + 8 * IB>(W), w1 = g16<4 + 8 * IB>(W), w2 = g16<6 + 8 * IB>(W), w3 = g16<8 + 8 * IB>(W);
        const uint32_t aux = g16<66 + 4 * IB>(W) | (g16<68 + 4 * IB>(W) << 16);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t wj = j == 0 ? w0 : j == 1 ? w1 : j == 2 ? w2 : w3;
            m40_g8(make_uint2(grid[wj & 255], grid[wj >> 8]), m40_s7((aux >> (7 * j)) & 127), v);
            m40_mac<TT, XS>(v, xs, AT + 8 * j, n, p);
        }
        m40_scale<TT>(h2f(uint16_t(g16<0>(W))) * (0.5f + float(aux >> 28)) * 0.5f, p, s, n);
    } else if constexpr (TY == 21) {   // IQ3_S: 8 grid bytes and a ninth bit each in qh, sign bytes, a scale nibble a run
        const uint32_t* grid = reinterpret_cast<const uint32_t*>(tab);
        const uint32_t w0 = g16<2 + 8 * IB>(W), w1 = g16<4 + 8 * IB>(W), w2 = g16<6 + 8 * IB>(W), w3 = g16<8 + 8 * IB>(W);
        const uint32_t sb = g16<74 + 4 * IB>(W) | (g16<76 + 4 * IB>(W) << 16), qh = g8<66 + IB>(W);
        const uint32_t sc = (g8<106 + IB / 2>(W) >> (4 * (IB & 1))) & 15;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t wj = j == 0 ? w0 : j == 1 ? w1 : j == 2 ? w2 : w3;
            const uint32_t i1 = (wj & 255) | ((qh << (8 - 2 * j)) & 256), i2 = (wj >> 8) | ((qh << (7 - 2 * j)) & 256);
            m40_g8(make_uint2(grid[i1], grid[i2]), (sb >> (8 * j)) & 255, v);
            m40_mac<TT, XS>(v, xs, AT + 8 * j, n, p);
        }
        m40_scale<TT>(h2f(uint16_t(g16<0>(W))) * float(1 + 2 * sc), p, s, n);
    } else if constexpr (TY == 20 || TY == 2) {   // IQ4_NL (kvalues) or Q4_0 (nibble - 8): blocks of 32, value i low nibble of qs[i], 16 + i high
        constexpr int B = 18 * IB;
        const uint32_t q0 = g16<B + 2>(W) | (g16<B + 4>(W) << 16), q1 = g16<B + 6>(W) | (g16<B + 8>(W) << 16);
        const uint32_t q2 = g16<B + 10>(W) | (g16<B + 12>(W) << 16), q3 = g16<B + 14>(W) | (g16<B + 16>(W) << 16);
        const float* kv = reinterpret_cast<const float*>(tab);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t a = j & 1 ? q2 : q0, b = j & 1 ? q3 : q1, sh = j < 2 ? 0 : 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t na = (a >> (8 * i + sh)) & 15, nb = (b >> (8 * i + sh)) & 15;
                if constexpr (TY == 20) v[i] = kv[na], v[4 + i] = kv[nb];
                else v[i] = m40_i(na) - 8.0f, v[4 + i] = m40_i(nb) - 8.0f;
            }
            m40_mac<TT, XS>(v, xs, AT + 8 * j, n, p);
        }
        m40_scale<TT>(h2f(uint16_t(g16<B>(W))), p, s, n);
    } else {   // Q2_0: blocks of 64, 2 bits a value (code - 1); run IB is half IB & 1 of block IB >> 1
        constexpr int B = 18 * (IB >> 1), H = 8 * (IB & 1);
        const uint32_t ql = g16<B + 2 + H>(W) | (g16<B + 4 + H>(W) << 16), qh = g16<B + 6 + H>(W) | (g16<B + 8 + H>(W) << 16);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t c = ((j < 2 ? ql : qh) >> (16 * (j & 1))) & 0xffff;
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = m40_i((c >> (2 * i)) & 3) - 1.0f;
            m40_mac<TT, XS>(v, xs, AT + 8 * j, n, p);
        }
        m40_scale<TT>(h2f(uint16_t(g16<B>(W))), p, s, n);
    }
}

// FNX_M40_KS=5: gate|up rows split by columns for small windows (few jobs, so few blocks): a 2560-column row's 10 groups in 5 blocks of 2
// (blockIdx.z), each writing its partial sums to slice z of Y, a.slice floats apart; k_m40_swiglu adds them in a fixed order, so the sums
// stay the same from run to run. Off by default: m40_rows_test (2026-11-06) had it faster at T = 4 for IQ2_XXS / IQ2_XS / IQ3_XXS (259 -> 182
// us for 10 jobs) but slower at T = 1 (IQ3_S 122 -> 202 us): the cost is the decode a weight, not the blocks
inline int m40_ks(int T, int K) {
    static const int ks = (int) env_int("FNX_M40_KS", 1);
    return ks == 5 && T <= 6 && K == 2560 ? 5 : 1;
}

template <int TY, int TT, int XS>
__device__ __forceinline__ void m40_group(const uint32_t (&W)[kGrW<TY>], const uint4* tab, const float* xs, const int n, const int cw, float (&s)[TT]) {
    if constexpr (kRunsGr<TY> == 8) {
        m40_run<TY, TT, XS, 0>(W, tab, xs, n, s);
        m40_run<TY, TT, XS, 1>(W, tab, xs, n, s);
        m40_run<TY, TT, XS, 2>(W, tab, xs, n, s);
        m40_run<TY, TT, XS, 3>(W, tab, xs, n, s);
        m40_run<TY, TT, XS, 4>(W, tab, xs, n, s);
        m40_run<TY, TT, XS, 5>(W, tab, xs, n, s);
        m40_run<TY, TT, XS, 6>(W, tab, xs, n, s);
        m40_run<TY, TT, XS, 7>(W, tab, xs, n, s);
    } else {
        m40_run<TY, TT, XS, 0>(W, tab, xs, n, s);
        m40_run<TY, TT, XS, 1>(W, tab, xs, n, s);
        if (cw > 64) m40_run<TY, TT, XS, 2>(W, tab, xs, n, s);
        if (cw > 96) m40_run<TY, TT, XS, 3>(W, tab, xs, n, s);
    }
}

// a row a lane, kM40T rows a block; the K columns a group (one IQ super-block, or 4 IQ4_NL / Q4_0 blocks, or 2 Q2_0 blocks) at a time:
// the job's rows for those columns go to shared memory first, then each lane loads its row's group and runs it. KS > 1: this block's two
// groups of the row (m40_ks), both loaded at once, into slice blockIdx.z of Y
template <int TY, int TT, int KS>
__global__ void __launch_bounds__(kM40T) k_m40_rows(RowArgs a) {
    constexpr int RUNS = kRunsGr<TY>, GB = kGrBytes<TY>, CH = 32 * RUNS, XS = KS > 1 ? 2 * CH : CH;
    __shared__ __align__(16) float xs[TT * XS];
    __shared__ int ys[TT];
    extern __shared__ uint4 tab[];
    const int tid = int(threadIdx.x), ji = int(blockIdx.y);
    if (ji >= *a.njobs) return;
    const Job* const J = a.jobs + ji;
    int we_i = __ldg(&J->w);
    if (a.phase == 1 && we_i < 0) return;
    if (a.phase == 2) {
        if (we_i >= 0) return;
        we_i = ~we_i;
    }
    m40_stage<TY>(tab, tid);
    const int n = min(TT, __ldg(&J->n));
    if (tid < n) ys[tid] = __ldg(&J->y[tid]);
    const int row = int(blockIdx.x) * kM40T + tid, N = a.N, K = a.K;
    const float* const X = reinterpret_cast<const float*>(a.X);
    const char* base = a.ep ? a.ep[we_i] : a.wa + size_t(we_i) * a.sa;
    const uint8_t* rowp = reinterpret_cast<const uint8_t*>(base) + size_t(row < N ? row : 0) * a.row_bytes;
    float s[TT];
#pragma unroll
    for (int k = 0; k < TT; ++k) s[k] = 0.0f;
    if constexpr (KS > 1) {
        const int g0 = 2 * int(blockIdx.z), c0 = g0 * CH;
        uint32_t W0[kGrW<TY>], W1[kGrW<TY>];
        if (row < N) {   // both groups' bytes in flight while the rows are staged
            group_load<TY>(rowp + g0 * GB, int(a.row_bytes) - g0 * GB, W0);
            group_load<TY>(rowp + (g0 + 1) * GB, int(a.row_bytes) - (g0 + 1) * GB, W1);
        }
        __syncthreads();   // the tables and ys
        for (int i = tid; i < n * (XS / 4); i += kM40T) {
            const int k = i / (XS / 4), c = i % (XS / 4);
            reinterpret_cast<float4*>(xs)[k * (XS / 4) + c] = __ldg(reinterpret_cast<const float4*>(X + size_t(ys[k]) * K + c0) + c);
        }
        __syncthreads();
        if (row < N) {
            m40_group<TY, TT, XS>(W0, tab, xs, n, CH, s);
            m40_group<TY, TT, XS>(W1, tab, xs + CH, n, CH, s);
#pragma unroll
            for (int k = 0; k < TT; ++k)
                if (k < n) a.Y[size_t(blockIdx.z) * a.slice + size_t(__ldg(&J->out[k])) * N + row] = s[k];
        }
        return;
    }
    const int nch = (K + CH - 1) / CH;
    for (int ch = 0; ch < nch; ++ch) {
        const int c0 = ch * CH, cw = min(CH, K - c0);
        __syncthreads();   // the last chunk's rows are done with (the first time: the tables and ys are in place)
        for (int i = tid; i < n * (cw / 4); i += kM40T) {
            const int k = i / (cw / 4), c = i % (cw / 4);
            reinterpret_cast<float4*>(xs)[k * (XS / 4) + c] = __ldg(reinterpret_cast<const float4*>(X + size_t(ys[k]) * K + c0) + c);
        }
        __syncthreads();
        if (row < N) {
            uint32_t W[kGrW<TY>];
            group_load<TY>(rowp + ch * GB, int(a.row_bytes) - ch * GB, W);
            m40_group<TY, TT, XS>(W, tab, xs, n, cw, s);
        }
    }
    if (row < N) {
#pragma unroll
        for (int k = 0; k < TT; ++k)
            if (k < n) a.Y[size_t(__ldg(&J->out[k])) * N + row] = s[k];
    }
}

// FNX_M40_LPR=5 or 10 (1: off): gate|up of a small window (T <= 4; 10 lanes only to T = 2, for shared memory) with L lanes a row, lane l
// the row's groups l, l + L, ...: with a row a lane only 8-13 warps an SM run at T = 1-2 (640 rows of 10-16 jobs) and the kernel waits on
// its loads (m40_rows_test, 2026-11-06: IQ3_XXS 113 us for what its instructions do in ~35; v4's second set of registers made it worse).
// A block is kM40LR rows; the rows' chunks of a round sit CH + 4 floats apart (the L lanes' float4 reads in different banks). The lanes'
// sums meet in shared memory and lane 0 adds them in lane order, so a row's sum does not change from run to run
constexpr int kM40LR = 32;
inline int m40_lpr(int T, int K, int CH) {
    static const int lpr = (int) env_int("FNX_M40_LPR", 1);
    if (lpr <= 1 || T > 4 || K % CH) return 1;
    if (lpr >= 10 && T <= 2 && (K / CH) % 10 == 0) return 10;
    return (K / CH) % 5 == 0 ? 5 : 1;
}

template <int TY, int TT, int L>
__global__ void __launch_bounds__(kM40LR * L) k_m40_rows_l(RowArgs a) {
    constexpr int RUNS = kRunsGr<TY>, GB = kGrBytes<TY>, CH = 32 * RUNS, CS = CH + 4, XS = L * CS, NT = kM40LR * L;
    __shared__ __align__(16) float xs[TT * XS];
    __shared__ float red[kM40LR * L * TT];
    __shared__ int ys[TT];
    extern __shared__ uint4 tab[];
    const int tid = int(threadIdx.x), ji = int(blockIdx.y), r = tid / L, l = tid - r * L;
    if (ji >= *a.njobs) return;
    const Job* const J = a.jobs + ji;
    int we_i = __ldg(&J->w);
    if (a.phase == 1 && we_i < 0) return;
    if (a.phase == 2) {
        if (we_i >= 0) return;
        we_i = ~we_i;
    }
    m40_stage<TY, NT>(tab, tid);
    const int n = min(TT, __ldg(&J->n));
    if (tid < n) ys[tid] = __ldg(&J->y[tid]);
    const int row = int(blockIdx.x) * kM40LR + r, N = a.N, K = a.K, nch = K / CH;
    const float* const X = reinterpret_cast<const float*>(a.X);
    const char* base = a.ep ? a.ep[we_i] : a.wa + size_t(we_i) * a.sa;
    const uint8_t* rowp = reinterpret_cast<const uint8_t*>(base) + size_t(row < N ? row : 0) * a.row_bytes;
    float s[TT];
#pragma unroll
    for (int k = 0; k < TT; ++k) s[k] = 0.0f;
    for (int c0 = 0; c0 < nch; c0 += L) {   // a round: chunks c0 .. c0 + L - 1, lane l's the chunk c0 + l
        const int nc = min(L, nch - c0), per = nc * (CH / 4);
        __syncthreads();   // the last round's rows are done with (the first time: the table and ys are in place)
        for (int i = tid; i < n * per; i += NT) {
            const int k = i / per, ci = (i % per) / (CH / 4), c = i % (CH / 4);
            reinterpret_cast<float4*>(xs + k * XS + ci * CS)[c] = __ldg(reinterpret_cast<const float4*>(X + size_t(ys[k]) * K + (c0 + ci) * CH) + c);
        }
        __syncthreads();
        if (row < N && l < nc) {
            uint32_t W[kGrW<TY>];
            group_load<TY>(rowp + (c0 + l) * GB, int(a.row_bytes) - (c0 + l) * GB, W);
            m40_group<TY, TT, XS>(W, tab, xs + l * CS, n, CH, s);
        }
    }
#pragma unroll
    for (int k = 0; k < TT; ++k) red[(r * L + l) * TT + k] = s[k];
    __syncthreads();
    if (l == 0 && row < N) {
#pragma unroll
        for (int k = 0; k < TT; ++k) {
            if (k >= n) continue;
            float t = red[r * L * TT + k];
#pragma unroll
            for (int j = 1; j < L; ++j) t += red[(r * L + j) * TT + k];
            a.Y[size_t(__ldg(&J->out[k])) * N + row] = t;
        }
    }
}

template <int TY>
void m40_rows_launch(RowArgs a, int T, int njobs_cap, cudaStream_t s) {
    constexpr size_t smem = size_t(M40TabU<TY>::n) * 16;
    if (a.K % 32) throw std::runtime_error("m40 moe rows: columns not whole runs");
    const int ks = kRunsGr<TY> == 8 ? m40_ks(T, a.K) : 1;
    a.slice = size_t(njobs_cap) * a.N;
    if constexpr (kRunsGr<TY> == 8) {
        const int lpr = ks == 1 ? m40_lpr(T, a.K, 32 * kRunsGr<TY>) : 1;
        if (lpr > 1) {
            const dim3 g((a.N + kM40LR - 1) / kM40LR, njobs_cap);
            if (lpr == 10) {
                if (T <= 1) k_m40_rows_l<TY, 1, 10><<<g, kM40LR * 10, smem, s>>>(a);
                else k_m40_rows_l<TY, 2, 10><<<g, kM40LR * 10, smem, s>>>(a);
            } else if (T <= 1) k_m40_rows_l<TY, 1, 5><<<g, kM40LR * 5, smem, s>>>(a);
            else if (T == 2) k_m40_rows_l<TY, 2, 5><<<g, kM40LR * 5, smem, s>>>(a);
            else if (T == 3) k_m40_rows_l<TY, 3, 5><<<g, kM40LR * 5, smem, s>>>(a);
            else k_m40_rows_l<TY, 4, 5><<<g, kM40LR * 5, smem, s>>>(a);
            return;
        }
    }
    const dim3 grid((a.N + kM40T - 1) / kM40T, njobs_cap, ks);
    if (ks == 5) {
        if (T <= 1) k_m40_rows<TY, 1, 5><<<grid, kM40T, smem, s>>>(a);
        else if (T == 2) k_m40_rows<TY, 2, 5><<<grid, kM40T, smem, s>>>(a);
        else if (T == 3) k_m40_rows<TY, 3, 5><<<grid, kM40T, smem, s>>>(a);
        else if (T == 4) k_m40_rows<TY, 4, 5><<<grid, kM40T, smem, s>>>(a);
        else k_m40_rows<TY, 6, 5><<<grid, kM40T, smem, s>>>(a);
        return;
    }
    if (T <= 1) k_m40_rows<TY, 1, 1><<<grid, kM40T, smem, s>>>(a);
    else if (T == 2) k_m40_rows<TY, 2, 1><<<grid, kM40T, smem, s>>>(a);
    else if (T == 3) k_m40_rows<TY, 3, 1><<<grid, kM40T, smem, s>>>(a);
    else if (T == 4) k_m40_rows<TY, 4, 1><<<grid, kM40T, smem, s>>>(a);
    else if (T <= 6) k_m40_rows<TY, 6, 1><<<grid, kM40T, smem, s>>>(a);
    else if (T <= 8) k_m40_rows<TY, 8, 1><<<grid, kM40T, smem, s>>>(a);
    else if (T <= 12) k_m40_rows<TY, 12, 1><<<grid, kM40T, smem, s>>>(a);
    else k_m40_rows<TY, 16, 1><<<grid, kM40T, smem, s>>>(a);
}

// the decode path's f32 inputs (the M40 build): the FFN input as it is, and silu(gate) * up of each pair (all, or only after a list's
// copies); gate|up's ks partial slices (m40_ks: P = T TOPK pairs, columns N) added first, in order
__global__ void __launch_bounds__(256) k_m40_swiglu(const float* __restrict__ gu, int ffs, int P, float* __restrict__ h, const int* __restrict__ list, int ks) {
    if (list && list[0] == 0) return;
    const size_t sl = size_t(P) * 2 * ffs;
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) P * ffs; e += (size_t) gridDim.x * blockDim.x) {
        const size_t r = e / ffs, i = e % ffs;
        float g = gu[r * 2 * ffs + i], u = gu[r * 2 * ffs + ffs + i];
        for (int k = 1; k < ks; ++k) g += gu[k * sl + r * 2 * ffs + i], u += gu[k * sl + r * 2 * ffs + ffs + i];
        h[e] = g / (1.0f + __expf(-g)) * u;
    }
}

}  // namespace
}  // namespace fnx
