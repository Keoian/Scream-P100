// The prompt path's routed experts on GP100: Strata's FP16 expert kernel (p100_iq_gemm.cuh) over the split engine's jobs.
// An expert's tokens come from its BigJob instead of a contiguous range, and its matrix from moe_expert (a VRAM slot or the mapped host copy).
#include "kernels.hpp"
#include "p100_over.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"


#include "p100_iq_gemm.cuh"

namespace fnx {
namespace {

using namespace strata::kernels::p100f::iqg;

__device__ __forceinline__ const char* moe_expert_(const char* wa, size_t sa, const char* const* ep, int e) {
    return ep ? ep[e] : wa + (size_t) e * sa;
}

struct JobArgs {
    Tabs t;
    const char* wa;
    size_t sa, row_bytes;
    const char* const* ep;
    const BigJob* jobs;
    const int* njobs;
    const uint4* X;
    float* Y;
    int ldx_u, N, K;
};

// gemm4_kernel over one tile of one job's tokens: block (rows group, 2 job + tile)
template <int TY>
__launch_bounds__(kThreads)
__global__ void k_gemm4_job(JobArgs a) {
    extern __shared__ uint4 sm[];
    constexpr int TL = kTL, RL = kRL, NT = kThreads, KU = k4KU, TB = k4TB, SU = kSU<TY>;
    uint4* const sw = sm;                 // [k4NB rows][KU]
    uint4* const sx = sm + k4NB * KU;     // [TB tokens][KU]
    uint4* const sg = sx + TB * KU;       // [k4NB rows][SU]: a row's IQ block
    uint4* const tab = sg + k4NB * SU;    // the tables
    int* const sj = reinterpret_cast<int*>(tab + TabU<TY>::n);   // the job
    const int tid = int(threadIdx.x), ji = int(blockIdx.y) / 2, tile = int(blockIdx.y) & 1;
    if (ji >= *a.njobs) return;
    const int* const srcj = reinterpret_cast<const int*>(a.jobs + ji);
    if (tile * TB >= srcj[1]) return;
    for (int k = tid; k < int(sizeof(BigJob) / 4); k += NT) sj[k] = srcj[k];
    const int t00 = tile * TB, n00 = int(blockIdx.x) * k4NB, N = a.N;
    const int ni = tid & 15, ti = tid >> 4;
    const int n0 = n00 + ni * RL, t0 = t00 + ti * TL;
    const int my_row = n00 + tid;
    const uint4* const X = a.X;
    const int ldx_u = a.ldx_u;
    const std::size_t row_bytes = a.row_bytes;
    const Tabs t = a.t;
#if FNX_ARCH >= 53
    half2 h[TL][RL];
#endif
    float s[TL][RL];
#pragma unroll
    for (int i = 0; i < TL; ++i)
#pragma unroll
        for (int r = 0; r < RL; ++r) {
#if FNX_ARCH >= 53
            h[i][r] = as_half2(0u);
#endif
            s[i][r] = 0.0f;
        }
    stage_tabs<TY>(t, tab, tid, NT);
    __syncthreads();
    const BigJob& jb = *reinterpret_cast<const BigJob*>(sj);
    const int tend = jb.n;
    const bool act = t0 < tend && n0 < N;
    const uint8_t* const we = reinterpret_cast<const uint8_t*>(moe_expert_(a.wa, a.sa, a.ep, jb.w));
    const int nu = a.K / 8;
    int off = 0;
    for (int u0 = 0; u0 < nu; u0 += KU) {
        const int run0 = u0 >> 2;
        if ((run0 & (kRunsGr<TY> - 1)) == 0 && my_row < N) {
            const uintptr_t ad = reinterpret_cast<uintptr_t>(we + std::size_t(my_row) * row_bytes + std::size_t(run0 / kRunsGr<TY>) * kGrBytes<TY>);
            const uint4* al = reinterpret_cast<const uint4*>(ad & ~uintptr_t(15));
            const uintptr_t row_end = reinterpret_cast<uintptr_t>(we + std::size_t(my_row + 1) * row_bytes);   // the last row may end a card's allocation
            off = int(ad & 15);
            uint4 v[SU];
#pragma unroll
            for (int q = 0; q < SU; ++q) v[q] = reinterpret_cast<uintptr_t>(al + q) < row_end ? __ldg(al + q) : make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int q = 0; q < SU; ++q) sg[tid * SU + q] = v[q];
        }
        if (my_row < N) {
            const uint8_t* gp = reinterpret_cast<const uint8_t*>(sg + tid * SU) + off;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                uint32_t rw[RunWords<TY>::n];
                run_load_s<TY>(gp, run0 + j, rw);
                uint4 o[4];
                finish_s<TY>(tab, t, rw, o);
#pragma unroll
                for (int q = 0; q < 4; ++q) sw[tid * KU + 4 * j + q] = o[q];
            }
        }
        for (int j0 = tid; j0 < TB * KU; j0 += 4 * NT) {
            uint4 v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int i = j0 + q * NT, r = i / KU, c = i % KU;
                if (i < TB * KU && t00 + r < tend) v[q] = __ldg(X + std::size_t(jb.y[t00 + r]) * ldx_u + u0 + c);
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int i = j0 + q * NT, r = i / KU;
                if (i < TB * KU && t00 + r < tend) sx[i] = v[q];
            }
        }
        __syncthreads();
        if (act) {
            const int wb = ni * RL * KU, xb = ti * TL * KU;
            int cc = tid & (KU - 1);
            for (int c = 0; c < KU; ++c) {
                uint4 wr[RL], xr[TL];
#pragma unroll
                for (int r = 0; r < RL; ++r) wr[r] = sw[wb + r * KU + cc];
#pragma unroll
                for (int i = 0; i < TL; ++i) xr[i] = sx[xb + i * KU + cc];
                cc = (cc + 1) & (KU - 1);
#if FNX_ARCH < 53
                // Maxwell: no f16 arithmetic. Each pair of halves to f32 once (a weight's for every token, a token's for every row),
                // f32 FMAs into the f32 sums
#define M40_MACJ(F) \
                { \
                    float2 xf[TL], wf[RL]; \
                    _Pragma("unroll") for (int i = 0; i < TL; ++i) xf[i] = __half22float2(as_half2(xr[i].F)); \
                    _Pragma("unroll") for (int r = 0; r < RL; ++r) wf[r] = __half22float2(as_half2(wr[r].F)); \
                    _Pragma("unroll") for (int i = 0; i < TL; ++i) \
                        _Pragma("unroll") for (int r = 0; r < RL; ++r) s[i][r] = fmaf(wf[r].y, xf[i].y, fmaf(wf[r].x, xf[i].x, s[i][r])); \
                }
                M40_MACJ(x) M40_MACJ(y) M40_MACJ(z) M40_MACJ(w)
#undef M40_MACJ
#else
#define P100_MACJ(F) \
                _Pragma("unroll") \
                for (int i = 0; i < TL; ++i) \
                    _Pragma("unroll") \
                    for (int r = 0; r < RL; ++r) h[i][r] = __hfma2(as_half2(wr[r].F), as_half2(xr[i].F), h[i][r]);
                P100_MACJ(x) P100_MACJ(y) P100_MACJ(z) P100_MACJ(w)
#undef P100_MACJ
#endif
            }
#if FNX_ARCH >= 53
#pragma unroll
            for (int i = 0; i < TL; ++i)
#pragma unroll
                for (int r = 0; r < RL; ++r) fold_one(h[i][r], s[i][r]);
#endif
        }
        __syncthreads();
    }
    if (act) {
#pragma unroll
        for (int i = 0; i < TL; ++i)
            if (t0 + i < tend) {
                float* yr = a.Y + std::size_t(jb.out[t0 + i]) * N + n0;
#pragma unroll
                for (int r = 0; r < RL; ++r) yr[r] = s[i][r];
            }
    }
}

template <int TY>
void job_launch(const JobArgs& a, int njobs_cap, cudaStream_t s) {
    constexpr std::size_t smem = (std::size_t(k4NB) * k4KU + std::size_t(k4TB) * k4KU + std::size_t(k4NB) * kSU<TY> + TabU<TY>::n) * 16 + sizeof(BigJob);
    static_assert(smem <= 48 * 1024, "p100 moe: shared memory past Pascal's limit");
    static_assert(PF_JOB <= 2 * k4TB, "p100 moe: a job fills at most two tiles");
    k_gemm4_job<TY><<<dim3((a.N + k4NB - 1) / k4NB, njobs_cap * 2), kThreads, smem, s>>>(a);
}

}  // namespace

#if FNX_ARCH < 53
bool m40_pf_moe(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* x, float* out, cudaStream_t s);   // m40_pf_gemm.cuh
#endif
void p100_moe(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* x, float* out, void* st) {
    cudaStream_t s = (cudaStream_t) st;
#if FNX_ARCH < 53
    if (m40_pf_moe(A, sa, jobs, njobs, njobs_cap, x, out, s)) return;   // FNX_M40_PF=1
#endif
    if (A.n_in % 64) throw std::runtime_error("p100 moe: columns not in units of 64");
    JobArgs a{};
    a.t = tabs(s);
    a.wa = (const char*) A.p;
    a.sa = sa;
    a.row_bytes = A.row_bytes;
    a.ep = A.ep;
    a.jobs = jobs;
    a.njobs = njobs;
    a.X = reinterpret_cast<const uint4*>(x);
    a.Y = out;
    a.ldx_u = A.n_in / 8;
    a.N = A.n_out;
    a.K = A.n_in;
    switch (A.type) {
        case T_IQ3_XXS: job_launch<18>(a, njobs_cap, s); break;
        case T_IQ3_S: job_launch<21>(a, njobs_cap, s); break;
        case T_IQ2_S: job_launch<22>(a, njobs_cap, s); break;
        case T_IQ4_XS: job_launch<23>(a, njobs_cap, s); break;
        case T_IQ4_NL: job_launch<20>(a, njobs_cap, s); break;
        case T_Q4_0: job_launch<2>(a, njobs_cap, s); break;
        case T_IQ2_XXS: job_launch<16>(a, njobs_cap, s); break;   // the M40's GSQ IQ3_XXS: 9 layers' gate|up
        case T_IQ2_XS: job_launch<17>(a, njobs_cap, s); break;    // 10 layers' gate|up
        case T_Q2_0: job_launch<42>(a, njobs_cap, s); break;      // 30 layers' down, read as they are (FNX_Q2_0_NATIVE)
        default: throw std::runtime_error("p100 moe: expert type " + std::to_string(A.type) + " not handled");
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("p100 moe: ") + cudaGetErrorString(e));
}

}  // namespace fnx

#include "p100_moe_rows.cuh"
#include "p100_attn.cuh"
#include "p100_dma.cuh"
#if FNX_ARCH < 53
#include "m40_pf_gemm.cuh"
#endif
