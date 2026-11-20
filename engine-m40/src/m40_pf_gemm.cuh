// m40_pf_gemm.cuh (2026-11-15): the prompt path's routed experts on Maxwell (sm_52, ai-streznik-m40), FNX_M40_PF=1. k_gemm4_job (p100_moe.cu)
// is Strata's P100 kernel with f16 tiles; on Maxwell each pair of halves goes to f32 before its FMAs, and a conversion runs at a quarter
// of the FMA rate: 112 FMA-cycles of conversions to 96 of FMAs a step, about 28 % of the cards' f32 peak in the engine (profile of
// 2026-11-06: gate|up and down 42 % of a long read). An f32-tile variant (m40_gemm.cuh, 6 Nov) dropped the conversions but read twice the
// shared bytes a FMA with the same 6 x 8 thread tile and 8 warps an SM, and was no faster. Here:
//   - a block is 128 rows of the expert by BN tokens of a job (96 or 48), a thread 8 rows by BN / 8 tokens (96 or 48 sums): 5 (or 3.5)
//     shared loads of 16 bytes for 96 (48) FMAs, so the FMAs, not shared memory, set the pace;
//   - a step is one run of 32 values: each thread decodes its row's run straight to f32 weights (ggml's dequantize_row_* values: the
//     scale times the grid value is exact in f32) into As[k][row], and the job's tokens' f16 values to f32 into Bs[k][token];
//   - the next step's token values, and every RG steps the row's next group, are loaded into registers while this step computes.
// Sums in f32, in the order of k: the same result from run to run; not the bits of k_gemm4_job (its weights are f16, rounded).
// Types: the M40 model's gate|up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, down IQ4_NL, Q2_0, and Q4_0; any other stays on k_gemm4_job.
#pragma once
#define FNX_M40_PF_GEMM 1

#include "env.hpp"

namespace fnx {
namespace {

constexpr int kPfT = 128, kPfM = 128, kPfK = 32;

// run IB of the group W as 32 f32 weights (value 8 j + i of the run in w[8 j + i])
template <int TY, int IB>
__device__ __forceinline__ void m40_dq(const uint32_t (&W)[kGrW<TY>], const uint4* tab, float (&w)[32]) {
    float v[8];
    auto put = [&](const int j, const float sc) {
#pragma unroll
        for (int i = 0; i < 8; ++i) w[8 * j + i] = v[i] * sc;
    };
    if constexpr (TY == 16) {   // IQ2_XXS
        const uint2* grid = reinterpret_cast<const uint2*>(tab);
        const uint32_t gi = g16<2 + 8 * IB>(W) | (g16<4 + 8 * IB>(W) << 16), aux = g16<6 + 8 * IB>(W) | (g16<8 + 8 * IB>(W) << 16);
        const float sc = h2f(uint16_t(g16<0>(W))) * (0.5f + float(aux >> 28)) * 0.25f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            m40_g8(grid[(gi >> (8 * j)) & 255], m40_s7((aux >> (7 * j)) & 127), v);
            put(j, sc);
        }
    } else if constexpr (TY == 17) {   // IQ2_XS
        const uint2* grid = reinterpret_cast<const uint2*>(tab);
        const uint32_t q01 = g16<2 + 8 * IB>(W) | (g16<4 + 8 * IB>(W) << 16), q23 = g16<6 + 8 * IB>(W) | (g16<8 + 8 * IB>(W) << 16);
        const uint32_t sc = g8<66 + IB>(W);
        const float d = h2f(uint16_t(g16<0>(W))) * 0.25f, s0 = d * (0.5f + float(sc & 15)), s1 = d * (0.5f + float(sc >> 4));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = ((j < 2 ? q01 : q23) >> (16 * (j & 1))) & 0xffff;
            m40_g8(grid[q & 511], m40_s7(q >> 9), v);
            put(j, j < 2 ? s0 : s1);
        }
    } else if constexpr (TY == 22) {   // IQ2_S
        const uint2* grid = reinterpret_cast<const uint2*>(tab);
        const uint32_t lo = g16<2 + 4 * IB>(W) | (g16<4 + 4 * IB>(W) << 16), sb = g16<34 + 4 * IB>(W) | (g16<36 + 4 * IB>(W) << 16);
        const uint32_t qh = g8<66 + IB>(W), sc = g8<74 + IB>(W);
        const float d = h2f(uint16_t(g16<0>(W))) * 0.25f, s0 = d * (0.5f + float(sc & 15)), s1 = d * (0.5f + float(sc >> 4));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            m40_g8(grid[((lo >> (8 * j)) & 255) | ((qh << (8 - 2 * j)) & 0x300u)], (sb >> (8 * j)) & 255, v);
            put(j, j < 2 ? s0 : s1);
        }
    } else if constexpr (TY == 18) {   // IQ3_XXS
        const uint32_t* grid = reinterpret_cast<const uint32_t*>(tab);
        const uint32_t w0 = g16<2 + 8 * IB>(W), w1 = g16<4 + 8 * IB>(W), w2 = g16<6 + 8 * IB>(W), w3 = g16<8 + 8 * IB>(W);
        const uint32_t aux = g16<66 + 4 * IB>(W) | (g16<68 + 4 * IB>(W) << 16);
        const float sc = h2f(uint16_t(g16<0>(W))) * (0.5f + float(aux >> 28)) * 0.5f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t wj = j == 0 ? w0 : j == 1 ? w1 : j == 2 ? w2 : w3;
            m40_g8(make_uint2(grid[wj & 255], grid[wj >> 8]), m40_s7((aux >> (7 * j)) & 127), v);
            put(j, sc);
        }
    } else if constexpr (TY == 21) {   // IQ3_S
        const uint32_t* grid = reinterpret_cast<const uint32_t*>(tab);
        const uint32_t w0 = g16<2 + 8 * IB>(W), w1 = g16<4 + 8 * IB>(W), w2 = g16<6 + 8 * IB>(W), w3 = g16<8 + 8 * IB>(W);
        const uint32_t sb = g16<74 + 4 * IB>(W) | (g16<76 + 4 * IB>(W) << 16), qh = g8<66 + IB>(W);
        const uint32_t s4 = (g8<106 + IB / 2>(W) >> (4 * (IB & 1))) & 15;
        const float sc = h2f(uint16_t(g16<0>(W))) * float(1 + 2 * s4);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t wj = j == 0 ? w0 : j == 1 ? w1 : j == 2 ? w2 : w3;
            const uint32_t i1 = (wj & 255) | ((qh << (8 - 2 * j)) & 256), i2 = (wj >> 8) | ((qh << (7 - 2 * j)) & 256);
            m40_g8(make_uint2(grid[i1], grid[i2]), (sb >> (8 * j)) & 255, v);
            put(j, sc);
        }
    } else if constexpr (TY == 20 || TY == 2) {   // IQ4_NL, Q4_0: block IB of 32
        constexpr int B = 18 * IB;
        const uint32_t q0 = g16<B + 2>(W) | (g16<B + 4>(W) << 16), q1 = g16<B + 6>(W) | (g16<B + 8>(W) << 16);
        const uint32_t q2 = g16<B + 10>(W) | (g16<B + 12>(W) << 16), q3 = g16<B + 14>(W) | (g16<B + 16>(W) << 16);
        const float* kv = reinterpret_cast<const float*>(tab);
        const float sc = h2f(uint16_t(g16<B>(W)));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t a = j & 1 ? q2 : q0, b = j & 1 ? q3 : q1, sh = j < 2 ? 0 : 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t na = (a >> (8 * i + sh)) & 15, nb = (b >> (8 * i + sh)) & 15;
                if constexpr (TY == 20) v[i] = kv[na], v[4 + i] = kv[nb];
                else v[i] = m40_i(na) - 8.0f, v[4 + i] = m40_i(nb) - 8.0f;
            }
            put(j, sc);
        }
    } else {   // Q2_0: half IB & 1 of block IB >> 1 (64 values)
        constexpr int B = 18 * (IB >> 1), H = 8 * (IB & 1);
        const uint32_t ql = g16<B + 2 + H>(W) | (g16<B + 4 + H>(W) << 16), qh = g16<B + 6 + H>(W) | (g16<B + 8 + H>(W) << 16);
        const float sc = h2f(uint16_t(g16<B>(W)));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t c = ((j < 2 ? ql : qh) >> (16 * (j & 1))) & 0xffff;
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = m40_i((c >> (2 * i)) & 3) - 1.0f;
            put(j, sc);
        }
    }
}

template <int TY>
__device__ __forceinline__ void m40_dq_run(const int ib, const uint32_t (&W)[kGrW<TY>], const uint4* tab, float (&w)[32]) {
    switch (ib) {
        case 0: m40_dq<TY, 0>(W, tab, w); break;
        case 1: m40_dq<TY, 1>(W, tab, w); break;
        case 2: m40_dq<TY, 2>(W, tab, w); break;
        case 3: m40_dq<TY, 3>(W, tab, w); break;
        default:
            if constexpr (kRunsGr<TY> == 8) {
                switch (ib) {
                    case 4: m40_dq<TY, 4>(W, tab, w); break;
                    case 5: m40_dq<TY, 5>(W, tab, w); break;
                    case 6: m40_dq<TY, 6>(W, tab, w); break;
                    default: m40_dq<TY, 7>(W, tab, w); break;
                }
            }
    }
}

// the raw 16-byte loads of a group (group_load's first half), and their shift into words from the group's first byte (its second)
template <int TY>
__device__ __forceinline__ void m40_gr_issue(const uint8_t* p, const int avail, uint4 (&V)[kSU<TY>], int& off) {
    const uintptr_t ad = reinterpret_cast<uintptr_t>(p);
    const uint4* al = reinterpret_cast<const uint4*>(ad & ~uintptr_t(15));
    off = int(ad & 15);
    const int bytes = off + min(kGrBytes<TY>, avail);
#pragma unroll
    for (int q = 0; q < kSU<TY>; ++q) V[q] = q * 16 < bytes ? __ldg(al + q) : make_uint4(0u, 0u, 0u, 0u);
}
template <int TY>
__device__ __forceinline__ void m40_gr_words(const uint4 (&V4)[kSU<TY>], const int off, uint32_t (&W)[kGrW<TY>]) {
    constexpr int NV = kSU<TY> * 4;
    uint32_t V[NV];
#pragma unroll
    for (int q = 0; q < kSU<TY>; ++q) V[4 * q] = V4[q].x, V[4 * q + 1] = V4[q].y, V[4 * q + 2] = V4[q].z, V[4 * q + 3] = V4[q].w;
    const bool s8 = (off & 8) != 0, s4 = (off & 4) != 0, s2 = (off & 2) != 0;
#pragma unroll
    for (int i = 0; i < NV - 2; ++i) V[i] = s8 ? V[i + 2] : V[i];
#pragma unroll
    for (int i = 0; i < NV - 1; ++i) V[i] = s4 ? V[i + 1] : V[i];
#pragma unroll
    for (int i = 0; i < kGrW<TY>; ++i) W[i] = s2 ? (V[i] >> 16) | (V[i + 1] << 16) : V[i];
}

struct PfArgs {
    const char* wa;
    size_t sa, row_bytes;
    const char* const* ep;
    const BigJob* jobs;
    const int* njobs;
    const __half* X;
    float* Y;
    int N, K, lo, hi;   // jobs of lo < n <= hi tokens
};

// a block: rows [128 blockIdx.x, + 128) of job blockIdx.y's expert, the job's first BN tokens (a job holds PF_JOB <= 96)
template <int TY, int BN>
__global__ void __launch_bounds__(kPfT, 2) k_m40_pf(PfArgs a) {
    constexpr int TN = BN / 8, RG = kRunsGr<TY>, NXQ = (BN * kPfK / 8 + kPfT - 1) / kPfT;   // tokens a thread; steps a group; X loads
    static_assert(BN == 96 || BN == 48 || BN == 16, "k_m40_pf: 96, 48 or 16 tokens a block");
    __shared__ __align__(16) float As[kPfK][kPfM];
    __shared__ __align__(16) float Bs[kPfK][BN];
    __shared__ __align__(16) uint4 tab[M40TabU<TY>::n > 0 ? M40TabU<TY>::n : 1];
    __shared__ int sy[BN], so[BN];
    __shared__ int sw, sn;
    const int tid = int(threadIdx.x), ji = int(blockIdx.y);
    if (ji >= *a.njobs) return;
    const BigJob* jp = a.jobs + ji;
    const int n = jp->n;
    if (n <= a.lo || n > a.hi) return;
    if (tid == 0) sw = jp->w, sn = n;
    for (int t = tid; t < BN; t += kPfT) {
        sy[t] = t < n ? jp->y[t] : -1;
        so[t] = t < n ? jp->out[t] : -1;
    }
    m40_stage<TY, kPfT>(tab, tid);
    __syncthreads();
    const int N = a.N, K = a.K, row0 = int(blockIdx.x) * kPfM, my_row = row0 + tid;
    const int rx = tid & 15, ty = tid >> 4;   // rows 4 rx + {0..3} and 64 + 4 rx + {0..3}; tokens by ty (below)
    const uint8_t* const we = reinterpret_cast<const uint8_t*>(a.ep ? a.ep[sw] : a.wa + (size_t) sw * a.sa);
    const uint8_t* const wrow = we + (size_t) min(my_row, N - 1) * a.row_bytes;
    const int nsteps = K / kPfK;
    // the step's tokens' values: load q of a thread is uint4 i = tid + kPfT q of the [BN tokens][4 uint4] tile, token i % BN, part i / BN
    uint4 xr[NXQ];
    auto x_issue = [&](const int s) {
#pragma unroll
        for (int q = 0; q < NXQ; ++q) {
            const int i = tid + kPfT * q, t = i % BN, c = i / BN;
            xr[q] = make_uint4(0u, 0u, 0u, 0u);
            if (i < BN * 4 && sy[t] >= 0) xr[q] = __ldg(reinterpret_cast<const uint4*>(a.X + (size_t) sy[t] * K + (size_t) s * kPfK) + c);
        }
    };
    auto x_store = [&] {
#pragma unroll
        for (int q = 0; q < NXQ; ++q) {
            const int i = tid + kPfT * q, t = i % BN, c = i / BN;
            if (i >= BN * 4) continue;
            const uint32_t u[4] = {xr[q].x, xr[q].y, xr[q].z, xr[q].w};
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&u[h]));
                Bs[8 * c + 2 * h][t] = f.x;
                Bs[8 * c + 2 * h + 1][t] = f.y;
            }
        }
    };
    uint4 gv[kSU<TY>];
    int goff = 0;
    uint32_t W[kGrW<TY>];
    auto g_issue = [&](const int s) {   // the group of step s
        const int g = s / RG;
        m40_gr_issue<TY>(wrow + (size_t) g * kGrBytes<TY>, int(a.row_bytes) - g * kGrBytes<TY>, gv, goff);
    };
    float acc[8][TN];
#pragma unroll
    for (int r = 0; r < 8; ++r)
#pragma unroll
        for (int t = 0; t < TN; ++t) acc[r][t] = 0.0f;
    x_issue(0);
    g_issue(0);
    for (int s = 0; s < nsteps; ++s) {
        const int ib = s % RG;
        // this step's tiles: the tokens' values, and the row's run as weights
        x_store();
        if (ib == 0) m40_gr_words<TY>(gv, goff, W);
        {
            float w[32];
            m40_dq_run<TY>(ib, W, tab, w);
#pragma unroll
            for (int k = 0; k < kPfK; ++k) As[k][tid] = w[k];
        }
        __syncthreads();
        // the next step's loads while this one computes
        if (s + 1 < nsteps) {
            x_issue(s + 1);
            if ((s + 1) % RG == 0) g_issue(s + 1);
        }
        // not unrolled whole: 32 x 101 instructions are more than the instruction cache holds
#ifndef M40_PF_UNR
#define M40_PF_UNR 2   // m40_pf_test 15 Nov, IQ3_S full jobs: 2 3.90, 4 3.83, 8 3.75, all 32 1.2 TFLOP/s
#endif
#define M40_PF_PRAGMA(x) _Pragma(#x)
#define M40_PF_UNROLL(n) M40_PF_PRAGMA(unroll n)
        M40_PF_UNROLL(M40_PF_UNR)
        for (int k = 0; k < kPfK; ++k) {
            const float4 a0 = *reinterpret_cast<const float4*>(&As[k][4 * rx]), a1 = *reinterpret_cast<const float4*>(&As[k][64 + 4 * rx]);
            const float ar[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float br[TN];
            if constexpr (BN == 96) {   // tokens 4 ty + {0..3}, 32 + 4 ty + {0..3}, 64 + 4 ty + {0..3}
#pragma unroll
                for (int q = 0; q < 3; ++q) {
                    const float4 b = *reinterpret_cast<const float4*>(&Bs[k][32 * q + 4 * ty]);
                    br[4 * q] = b.x, br[4 * q + 1] = b.y, br[4 * q + 2] = b.z, br[4 * q + 3] = b.w;
                }
            } else if constexpr (BN == 48) {   // tokens 6 ty + {0..5}
#pragma unroll
                for (int q = 0; q < 3; ++q) {
                    const float2 b = *reinterpret_cast<const float2*>(&Bs[k][6 * ty + 2 * q]);
                    br[2 * q] = b.x, br[2 * q + 1] = b.y;
                }
            } else {   // tokens 2 ty + {0, 1}
                const float2 b = *reinterpret_cast<const float2*>(&Bs[k][2 * ty]);
                br[0] = b.x, br[1] = b.y;
            }
#pragma unroll
            for (int r = 0; r < 8; ++r)
#pragma unroll
                for (int t = 0; t < TN; ++t) acc[r][t] = fmaf(ar[r], br[t], acc[r][t]);
        }
        __syncthreads();
    }
    // out: token t's rows 4 rx + {0..3} and 64 + 4 rx + {0..3}
#pragma unroll
    for (int t = 0; t < TN; ++t) {
        const int tk = BN == 96 ? 32 * (t / 4) + 4 * ty + (t & 3) : TN * ty + t;
        if (tk >= sn) continue;
        float* yr = a.Y + (size_t) so[tk] * N + row0;
        if (row0 + 4 * rx + 3 < N) *reinterpret_cast<float4*>(yr + 4 * rx) = make_float4(acc[0][t], acc[1][t], acc[2][t], acc[3][t]);
        if (row0 + 64 + 4 * rx + 3 < N) *reinterpret_cast<float4*>(yr + 64 + 4 * rx) = make_float4(acc[4][t], acc[5][t], acc[6][t], acc[7][t]);
    }
}

template <int TY>
void m40_pf_launch(const PfArgs& a0, int njobs_cap, cudaStream_t s) {
    const dim3 grid((a0.N + kPfM - 1) / kPfM, njobs_cap);
    PfArgs a = a0;
    a.lo = 48, a.hi = 96;
    k_m40_pf<TY, 96><<<grid, kPfT, 0, s>>>(a);
    // jobs of up to this many tokens on the 16-token block (0: none; 16 at most)
    static const int small = std::max(0, std::min(16, (int) env_int("FNX_M40_PF16", 16)));
    a.lo = small, a.hi = 48;
    k_m40_pf<TY, 48><<<grid, kPfT, 0, s>>>(a);
    if (small > 0) {
        a.lo = 0, a.hi = small;
        k_m40_pf<TY, 16><<<grid, kPfT, 0, s>>>(a);
    }
}

}  // namespace

// the prompt path's routed experts of the M40's types on k_m40_pf; false: not handled (k_gemm4_job runs)
bool m40_pf_run(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* x, float* out, cudaStream_t s) {
    if (A.n_in % kPfK || A.n_out % 4) return false;
    PfArgs a{};
    a.wa = (const char*) A.p;
    a.sa = sa;
    a.row_bytes = A.row_bytes;
    a.ep = A.ep;
    a.jobs = jobs;
    a.njobs = njobs;
    a.X = reinterpret_cast<const __half*>(x);
    a.Y = out;
    a.N = A.n_out;
    a.K = A.n_in;
    switch (A.type) {
        case T_IQ2_XXS: m40_pf_launch<16>(a, njobs_cap, s); break;
        case T_IQ2_XS: m40_pf_launch<17>(a, njobs_cap, s); break;
        case T_IQ2_S: m40_pf_launch<22>(a, njobs_cap, s); break;
        case T_IQ3_XXS: m40_pf_launch<18>(a, njobs_cap, s); break;
        case T_IQ3_S: m40_pf_launch<21>(a, njobs_cap, s); break;
        case T_IQ4_NL: m40_pf_launch<20>(a, njobs_cap, s); break;
        case T_Q4_0: m40_pf_launch<2>(a, njobs_cap, s); break;
        case T_Q2_0: m40_pf_launch<42>(a, njobs_cap, s); break;
        default: return false;
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("m40 pf: ") + cudaGetErrorString(e));
    return true;
}
// FNX_M40_PF=1: p100_moe's jobs on k_m40_pf (unset or 0: k_gemm4_job, as before)
bool m40_pf_moe(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* x, float* out, cudaStream_t s) {
    static const bool on = env_int("FNX_M40_PF", 0) != 0;
    return on && m40_pf_run(A, sa, jobs, njobs, njobs_cap, x, out, s);
}

}  // namespace fnx
