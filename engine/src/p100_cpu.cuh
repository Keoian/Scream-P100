// p100_cpu.cuh: FNX_CPU_EXPERTS=1. A decode window's routed experts that are not in VRAM computed on the CPU, from the host copies the
// engine keeps, while the card's row kernels compute the ones that are (the copy engine, p100_dma.cuh, copied them into slots first:
// most of a window's time on a card that holds a third of the experts, and bound by the card's PCIe link).
// The plan lists the layer's misses without giving them slots (cache_plan, RouteTail::cpu); k_cpu_publish marks their jobs (the row
// kernels' phase 1 skips them) and writes the list, the window's picks and its rows (f16) to a mailbox in mapped host memory; the card's
// host thread and its pool (cpu_experts.cpp, the card's own socket) compute this card's half of each listed expert for the tokens that
// picked it, in fp32 from the file's blocks, and send the rows back with one copy and a word in VRAM (cuStreamWriteValue32), as the
// copier does; k_cpu_wait spins on that word after the card's own experts, k_cpu_scatter puts the rows where phase 2 would have.
// FNX_CPU_CHECK=1: the copy engine and phase 2 run as without the switch and the CPU's rows are only compared with theirs.
// Included by p100_moe.cu, after p100_dma.cuh (its includes and dma_pin).
#pragma once
#include "cpu_experts.hpp"

namespace fnx {
namespace {

constexpr size_t CPU_HDR = 4096;   // the staging buffer's head: n_rows, seq, then each row's place in `part` (its pair, t * TOPK + k)
static_assert(N % 16 == 0 && (8 + DMA_MAX) * sizeof(int) <= CPU_HDR, "p100_cpu.cuh: a row in 16-byte units, the head's rows in its 4096 bytes");
struct CpuBox {   // mapped host memory: the card writes the list, the picks and the rows, then seq
    volatile unsigned seq;
    int n, T, layer;
    unsigned sum;                // the sum of every 32-bit word of the list, the picks and the rows: the host thread checks it
    unsigned pad[3];             // the rows stay on a 16-byte boundary (the kernels copy them 16 bytes a step)
    int list[DMA_MAX];           // the listed experts (layer * NEXP + expert)
    int topi[MAXT * TOPK];       // the window's picks, pair t * TOPK + k
    uint16_t x[MAXT * N];        // the window's FFN input rows, f16
};
struct CpuDev {   // VRAM
    unsigned seq;    // the card's count of lists
    unsigned want;   // the last list's number, 0 when it was empty
    unsigned flag;   // the host thread's: the last list whose rows have landed
    unsigned err;    // 1: a wait timed out
    float chk[4];    // FNX_CPU_CHECK: the largest |CPU - card| of a value, the largest |card| value, the rows compared, the worst row's ratio
    unsigned cnt;    // k_cpu_post's blocks that have written their row
    unsigned gflag;  // FNX_CPU_EXACT: the host thread's: the last list whose gate|up rows have landed
    unsigned hcnt;   // FNX_CPU_EXACT: k_cpu_swiglu's blocks that are done
    unsigned bitdiff;   // FNX_CPU_CHECK: values whose bits differ from the card's own
    unsigned gdiff, hdiff;   // FNX_CPU_CHECK with FNX_CPU_EXACT: the CPU's gate|up sums, and the SwiGLU of them, that are not the card's bits
    unsigned xsum, hsum;     // FNX_CPU_POST_ROWS, FNX_CPU_HB_DIRECT (the many-block posts, kept to measure them): the blocks' sums
};
// FNX_CPU_EXACT: the card's SwiGLU of the CPU's gate|up rows goes back through mapped host memory: the list's number, the sum of the
// rows' 32-bit words and the number of rows, then the rows' h (f16)
constexpr size_t CPU_HB_HDR = 64;
static_assert(offsetof(CpuBox, x) % 16 == 0, "p100_cpu.cuh: the mailbox's rows on a 16-byte boundary");
static_assert(offsetof(CpuDev, seq) == 0 && offsetof(CpuDev, want) == sizeof(unsigned), "the router's last block writes CpuDev's seq and want as words 0 and 1 (moe.cuh)");
struct CpuStat {
    std::atomic<unsigned long long> lists{0}, experts{0}, rows{0}, ns_work{0}, ns_send{0}, by_t[MAXT + 1], ex_t[MAXT + 1], ns_t[MAXT + 1], worst_ns{0};
    std::atomic<unsigned long long> ns_hand{0}, h_timeouts{0};   // FNX_CPU_EXACT: gate|up sent to the card's h back; waits given up
    std::atomic<unsigned long long> ns_gu_t[MAXT + 1] = {}, ns_dn_t[MAXT + 1] = {}, ns_hand_t[MAXT + 1] = {};   // FNX_CPU_EXACT, by the window's rows: a list's gate|up, down, hand-off
    // ... and the pool in those lists (cpux::Pool::stats): its threads' own time, its rounds' time, the wait for the first worker
    std::atomic<unsigned long long> pool_busy_t[MAXT + 1] = {}, pool_wall_t[MAXT + 1] = {}, pool_first_t[MAXT + 1] = {}, pool_rounds_t[MAXT + 1] = {};
    std::atomic<unsigned long long> ns_box_t[MAXT + 1] = {};   // from a list's number seen to its mailbox read and checked
    // what the card wrote to mapped host memory against its sum: lists read again until they matched, and those that never did
    std::atomic<unsigned long long> box_again{0}, box_bad{0}, h_again{0}, h_bad{0};
    std::atomic<unsigned long long> pool_late{0};   // cpux::Pool::late
    // FNX_CPU_REUSE=1 (measuring only): listed experts, and those that had been on a list within the last 250, 500, 1000, 2000, 4000
    // listed experts of this card: what a cache of that many experts' decoded weights could have served at most
    std::atomic<unsigned long long> reuse_n{0}, reuse_hit[5] = {};
    // FNX_CPU_DNPROBE=1 (measuring only), by the window's rows: lists probed, from a list's arrival to its experts' down weights
    // landed on the card, and to the CPU's own down rows done
    std::atomic<unsigned long long> dnp_n[MAXT + 1] = {}, dnp_copy_ns[MAXT + 1] = {}, dnp_cpu_ns[MAXT + 1] = {};
};
struct CpuCard {
    int dev = 0, node = -1;
    CpuBox* box = nullptr;      // host address
    CpuBox* box_d = nullptr;    // its device address
    CpuDev* d = nullptr;
    char* stage_h = nullptr;    // page-locked: the head, then the rows
    char* stage_d = nullptr;
    cudaStream_t cs = nullptr;
    const char* const* host_gu = nullptr;   // [NL * NEXP], the card's host copies (Model: CardW::ec_host_gu)
    const char* const* host_dn = nullptr;
    uint32_t gu_type[NL] = {}, dn_type[NL] = {};
    int ffs = 0, threads = 0;
    std::vector<int> cpus;      // the pool's cores: this card's share of its socket's
    std::thread th;
    std::atomic<bool> stop{false};
    CpuStat st;
    std::vector<float> x;       // the window's rows as floats
    std::vector<float> h;       // the experts' gate|up results: a row a pair
    // FNX_CPU_EXACT: the card's arithmetic on the CPU (cpu_experts.hpp); the CPU's gate|up rows to the card, its SwiGLU back
    bool exact = false;
    int gu_lanes = 2;
    char* gstage_h = nullptr;   // page-locked and mapped: the head (n_rows, seq, the rows' sum), then each row's gate sums and up sums
    char* gstage_m = nullptr;   // its address on the card (k_cpu_gpull reads it from there)
    char* gstage_d = nullptr;   // the card's copy of it
    bool gcopy = false;         // FNX_CPU_GCOPY (measuring only): the rows by a copy on the card's copy queue, as the first version did
    // FNX_CPU_DNPROBE=1 (measuring only; the rows are computed as ever): a list's experts' down weights are also copied to a scratch
    // buffer on the card, on a stream of their own, and the time they land is noted: what "only the down weights copied, the card
    // doing SwiGLU and down" would wait for, against the CPU's own down
    bool dnprobe = false;
    cudaStream_t ds = nullptr;
    char* dn_scratch = nullptr;
    unsigned char* hb = nullptr;     // mapped: the list's number, the rows' sum and count, then the rows' h (f16)
    unsigned char* hb_d = nullptr;
    char* hdev = nullptr;            // VRAM: the rows' h as k_cpu_swiglu wrote it; k_cpu_hpost copies it to hb
    bool post_rows = false, hb_direct = false;   // FNX_CPU_POST_ROWS, FNX_CPU_HB_DIRECT: the many-block posts (measuring only)
};

// rows r0..r1-1 of the window (16 bytes a step), and with row 0 the list and the picks, into the mailbox: this thread's share, and the
// sum of the 32-bit words it wrote
__device__ __forceinline__ unsigned box_copy(const int* __restrict__ list, const int* __restrict__ topi, int T, const uint16_t* __restrict__ xh, CpuBox* box, int k, int tid,
                                             int nth, int r0, int r1) {
    unsigned sum = 0;
    const uint4* src = reinterpret_cast<const uint4*>(xh);
    uint4* dst = reinterpret_cast<uint4*>(box->x);
    for (int i = r0 * (N / 8) + tid; i < r1 * (N / 8); i += nth) {
        const uint4 v = src[i];
        dst[i] = v;
        sum += v.x + v.y + v.z + v.w;
    }
    if (r0 == 0) {
        for (int i = tid; i < k; i += nth) {
            const int v = list[1 + i];
            box->list[i] = v;
            sum += (unsigned) v;
        }
        for (int i = tid; i < T * TOPK; i += nth) {
            const int v = topi[i];
            box->topi[i] = v;
            sum += (unsigned) v;
        }
    }
    return sum;
}

__global__ void __launch_bounds__(256) k_cpu_publish(Job* gu, Job* dn, const int* __restrict__ njobs, int layer, const int* __restrict__ list,
                                                     const int* __restrict__ topi, int T, const uint16_t* __restrict__ xh, CpuBox* box, CpuDev* d, int mark) {
    const int k = min(list[0], DMA_MAX), tid = threadIdx.x;
    if (k == 0) {
        if (tid == 0) d->want = 0;
        return;
    }
    if (mark) {   // as k_jobs_split: the listed experts' jobs get w = ~w in both lists
        const int nj = *njobs, base = layer * NEXP;
        for (int j = tid; j < nj; j += 256) {
            const int g = base + gu[j].w;
            for (int i = 1; i <= k; ++i)
                if (list[i] == g) {
                    gu[j].w = ~gu[j].w;
                    dn[j].w = ~dn[j].w;
                    break;
                }
        }
    }
    __shared__ unsigned ss[256];
    ss[tid] = box_copy(list, topi, T, xh, box, k, tid, 256, 0, T);
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        unsigned sum = 0;
        for (int i = 0; i < 256; ++i) sum += ss[i];
        box->n = k;
        box->T = T;
        box->layer = layer;
        box->sum = sum;
        const unsigned s = ++d->seq;
        d->want = s;
        __threadfence_system();
        box->seq = s;
        __threadfence_system();
    }
}

// the window's rows, the plan's list and the picks to the mailbox, then the list's number (the router's last block set it in d->want): on
// the third stream, beside the card's own experts. One block, as k_cpu_publish: every word of the mailbox and, after the threads'
// fences, its sum and its number come from one SM; the host thread checks the sum (cpu_loop) before it computes from the mailbox, so
// "the CPU read what the card wrote" is a counted line of the report, not an assumption. (A block a row with the number from the last
// block to finish, as this was first written, is k_cpu_post_rows: FNX_CPU_POST_ROWS=1, to measure it. It has not been seen to arrive
// torn: 0 of 113,604 lists a card. The rows that were not the card's came from cpux::Pool, see its `limit`.)
__global__ void __launch_bounds__(256) k_cpu_post(int layer, const int* __restrict__ list, const int* __restrict__ topi, int T, const uint16_t* __restrict__ xh, CpuBox* box,
                                                  CpuDev* d) {
    const int k = min(list[0], DMA_MAX), tid = threadIdx.x;
    if (k == 0) return;
    __shared__ unsigned ss[256];
    ss[tid] = box_copy(list, topi, T, xh, box, k, tid, 256, 0, T);
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        unsigned sum = 0;
        for (int i = 0; i < 256; ++i) sum += ss[i];
        box->n = k;
        box->T = T;
        box->layer = layer;
        box->sum = sum;
        __threadfence_system();
        box->seq = d->want;
        __threadfence_system();
    }
}
// FNX_CPU_POST_ROWS=1 (measuring only): a block a row, the blocks' sums added in VRAM, the list's number by the last block to finish
__global__ void __launch_bounds__(256) k_cpu_post_rows(int layer, const int* __restrict__ list, const int* __restrict__ topi, int T, const uint16_t* __restrict__ xh,
                                                       CpuBox* box, CpuDev* d) {
    const int k = min(list[0], DMA_MAX), tid = threadIdx.x, r = blockIdx.x;
    if (k == 0) return;
    __shared__ unsigned ss[256];
    ss[tid] = box_copy(list, topi, T, xh, box, k, tid, 256, r, r + 1);
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        unsigned sum = 0;
        for (int i = 0; i < 256; ++i) sum += ss[i];
        atomicAdd(&d->xsum, sum);
        if (atomicAdd(&d->cnt, 1u) == (unsigned) (T - 1)) {
            d->cnt = 0u;
            box->n = k;
            box->T = T;
            box->layer = layer;
            box->sum = atomicExch(&d->xsum, 0u);
            __threadfence_system();
            box->seq = d->want;
            __threadfence_system();
        }
    }
}

__global__ void __launch_bounds__(32) k_cpu_wait(CpuDev* d) {
    if (threadIdx.x != 0) return;
    const unsigned want = *(volatile unsigned*) &d->want;
    if (want == 0) return;
    const long long t0 = clock64();
    while ((int) (*(volatile unsigned*) &d->flag - want) < 0) {
        if (clock64() - t0 > 4000000000LL) {   // ~3 s: the host thread is gone
            atomicOr(&d->err, 1u);
            break;
        }
    }
    __threadfence();
}

// FNX_CPU_EXACT: wait for the CPU's gate|up rows (the host thread's copy, then gflag), as k_cpu_wait does for its final rows
__global__ void __launch_bounds__(32) k_cpu_wait_g(CpuDev* d) {
    if (threadIdx.x != 0) return;
    const unsigned want = *(volatile unsigned*) &d->want;
    if (want == 0) return;
    const long long t0 = clock64();
    while ((int) (*(volatile unsigned*) &d->gflag - want) < 0) {
        if (clock64() - t0 > 4000000000LL) {   // ~3 s: the host thread is gone
            atomicOr(&d->err, 2u);
            break;
        }
    }
    __threadfence();
}
// FNX_CPU_EXACT: the CPU's gate|up rows fetched by the card out of mapped host memory, where the host thread wrote them, their count,
// the sum of their 32-bit words and then the list's number. One block: it waits for the number, copies the rows into VRAM 16 bytes a
// step and checks the sum (again while it does not match: words still on their way), then writes the head k_cpu_swiglu reads. Not a
// copy on the card's copy queue (k_cpu_wait_g's way, FNX_CPU_GCOPY=1): beside FNX_CPU_SHARE's copies that queue is long, and the rows
// waited behind it (0.7 ms a 16-row list on a card on x8, measured)
__global__ void __launch_bounds__(256) k_cpu_gpull(CpuDev* d, const char* __restrict__ gm, char* __restrict__ gd, int ffs) {
    const unsigned want = *(volatile unsigned*) &d->want;
    if (want == 0) return;
    const int tid = threadIdx.x;
    __shared__ unsigned ss[256];
    __shared__ int s_nr, s_ok;
    const volatile int* hm = reinterpret_cast<const volatile int*>(gm);
    if (tid == 0) {
        const long long t0 = clock64();
        int ok = 1;
        while ((int) ((unsigned) hm[1] - want) < 0)
            if (clock64() - t0 > 4000000000LL) {   // ~3 s: the host thread is gone
                atomicOr(&d->err, 2u);
                ok = 0;
                break;
            }
        s_nr = ok ? min(max((int) hm[0], 0), DMA_MAX) : 0;
        s_ok = 0;
    }
    __syncthreads();
    const int nr = s_nr, n16 = nr * (2 * ffs / 4);
    const uint4* src = reinterpret_cast<const uint4*>(gm + CPU_HDR);
    uint4* dst = reinterpret_cast<uint4*>(gd + CPU_HDR);
    for (int pass = 0; pass < 256; ++pass) {
        unsigned sum = 0;
        for (int i = tid; i < n16; i += 256) {
            const uint4 v = src[i];
            dst[i] = v;
            sum += v.x + v.y + v.z + v.w;
        }
        ss[tid] = sum;
        __syncthreads();
        if (tid == 0) {
            unsigned t = 0;
            for (int i = 0; i < 256; ++i) t += ss[i];
            s_ok = t == (unsigned) hm[2];
        }
        __syncthreads();
        if (s_ok) break;
    }
    if (tid == 0) {
        int* hd = reinterpret_cast<int*>(gd);
        hd[0] = s_ok ? nr : 0;
        hd[1] = (int) want;
        if (!s_ok && nr) atomicOr(&d->err, 4u);   // the rows never matched their sum: none taken (the window goes on without the CPU's rows)
        __threadfence();
    }
}
// FNX_CPU_EXACT: the card's SwiGLU of the CPU's gate|up rows, the very expression of k_swiglu_after (p100_moe_rows.cuh; its __expf and
// division under --use_fast_math are the card's own and cannot be done on a CPU), as f16 into VRAM
__global__ void __launch_bounds__(256) k_cpu_swiglu(const CpuDev* d, const char* __restrict__ stage, int ffs, __half* __restrict__ h) {
    const unsigned want = *(volatile unsigned*) &d->want;
    if (want == 0) return;
    const int* hd = reinterpret_cast<const int*>(stage);
    const int nr = (unsigned) hd[1] == want ? hd[0] : 0;
    const float* gu = reinterpret_cast<const float*>(stage + CPU_HDR);
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) nr * ffs; e += (size_t) gridDim.x * blockDim.x) {
        const size_t r = e / ffs, i = e % ffs;
        const float g = gu[r * 2 * ffs + i];
        h[e] = __float2half(silu_over(g) * gu[r * 2 * ffs + ffs + i]);
    }
}
// ... and those rows to the host thread through mapped memory, 16 bytes a step, with the sum of their 32-bit words and their count,
// then the list's number: one block, and the host thread checks the sum, as for the mailbox. (FNX_CPU_HB_DIRECT=1 is the first version,
// k_cpu_swiglu_direct: h straight into mapped memory from 40 blocks, the number from the last to finish; it too has not been seen to
// arrive torn, 0 of 113,604 lists a card)
__global__ void __launch_bounds__(256) k_cpu_hpost(const CpuDev* d, const char* __restrict__ stage, int ffs, const __half* __restrict__ h, unsigned char* hb) {
    const unsigned want = *(volatile unsigned*) &d->want;
    if (want == 0) return;
    const int tid = threadIdx.x;
    const int* hd = reinterpret_cast<const int*>(stage);
    const int nr = (unsigned) hd[1] == want ? hd[0] : 0;
    const uint4* src = reinterpret_cast<const uint4*>(h);
    uint4* dst = reinterpret_cast<uint4*>(hb + CPU_HB_HDR);
    unsigned sum = 0;
    for (int i = tid; i < nr * (ffs / 8); i += 256) {
        const uint4 v = src[i];
        dst[i] = v;
        sum += v.x + v.y + v.z + v.w;
    }
    __shared__ unsigned ss[256];
    ss[tid] = sum;
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        sum = 0;
        for (int i = 0; i < 256; ++i) sum += ss[i];
        unsigned* w = reinterpret_cast<unsigned*>(hb);
        w[1] = sum;
        w[2] = (unsigned) nr;
        __threadfence_system();
        *reinterpret_cast<volatile unsigned*>(hb) = want;
        __threadfence_system();
    }
}
// The hand-off in one block: k_cpu_gpull's wait and fetch, k_cpu_swiglu's expression into VRAM, k_cpu_hpost's post. This block is
// launched with the list and is running when the CPU's rows come; as three launches in a row, each of the later two waited for
// the row kernels beside it to give up a slot (the hand-off took 70-180 us a list beside them, 36-38 us on an idle card)
__global__ void __launch_bounds__(256) k_cpu_swiglu1(CpuDev* d, const char* __restrict__ gm, char* __restrict__ gd, int ffs, __half* __restrict__ h, unsigned char* hb) {
    const unsigned want = *(volatile unsigned*) &d->want;
    if (want == 0) return;
    const int tid = threadIdx.x;
    __shared__ unsigned ss[256];
    __shared__ int s_nr, s_ok;
    const volatile int* hm = reinterpret_cast<const volatile int*>(gm);
    if (tid == 0) {
        const long long t0 = clock64();
        int ok = 1;
        while ((int) ((unsigned) hm[1] - want) < 0)
            if (clock64() - t0 > 4000000000LL) {   // ~3 s: the host thread is gone
                atomicOr(&d->err, 2u);
                ok = 0;
                break;
            }
        s_nr = ok ? min(max((int) hm[0], 0), DMA_MAX) : 0;
        s_ok = 0;
    }
    __syncthreads();
    const int n16 = s_nr * (2 * ffs / 4);
    {
        const uint4* src = reinterpret_cast<const uint4*>(gm + CPU_HDR);
        uint4* dst = reinterpret_cast<uint4*>(gd + CPU_HDR);
        for (int pass = 0; pass < 256; ++pass) {
            unsigned sum = 0;
            for (int i = tid; i < n16; i += 256) {
                const uint4 v = src[i];
                dst[i] = v;
                sum += v.x + v.y + v.z + v.w;
            }
            ss[tid] = sum;
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                unsigned t = 0;
                for (int i = 0; i < 256; ++i) t += ss[i];
                s_ok = t == (unsigned) hm[2];
            }
            __syncthreads();
            if (s_ok) break;
        }
    }
    const int nr = s_ok ? s_nr : 0;
    if (tid == 0) {
        int* hd = reinterpret_cast<int*>(gd);
        hd[0] = nr;
        hd[1] = (int) want;
        if (!s_ok && s_nr) atomicOr(&d->err, 4u);
    }
    const float* gu = reinterpret_cast<const float*>(gd + CPU_HDR);
    for (size_t e = (size_t) tid; e < (size_t) nr * ffs; e += 256) {   // k_cpu_swiglu's (k_swiglu_after's) expression
        const size_t r = e / ffs, i = e % ffs;
        const float g = gu[r * 2 * ffs + i];
        h[e] = __float2half(silu_over(g) * gu[r * 2 * ffs + ffs + i]);
    }
    __threadfence();
    __syncthreads();
    const uint4* src = reinterpret_cast<const uint4*>(h);
    uint4* dst = reinterpret_cast<uint4*>(hb + CPU_HB_HDR);
    unsigned sum = 0;
    for (int i = tid; i < nr * (ffs / 8); i += 256) {
        const uint4 v = src[i];
        dst[i] = v;
        sum += v.x + v.y + v.z + v.w;
    }
    ss[tid] = sum;
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        sum = 0;
        for (int i = 0; i < 256; ++i) sum += ss[i];
        unsigned* w = reinterpret_cast<unsigned*>(hb);
        w[1] = sum;
        w[2] = (unsigned) nr;
        __threadfence_system();
        *reinterpret_cast<volatile unsigned*>(hb) = want;
        __threadfence_system();
    }
}
// FNX_CPU_HB_DIRECT=1 (measuring only): the first version, h straight into mapped memory from every block, the number from the last
__global__ void __launch_bounds__(256) k_cpu_swiglu_direct(CpuDev* d, const char* __restrict__ stage, int ffs, unsigned char* hb) {
    const unsigned want = *(volatile unsigned*) &d->want;
    if (want == 0) return;
    const int* hd = reinterpret_cast<const int*>(stage);
    const int nr = (unsigned) hd[1] == want ? hd[0] : 0;
    const float* gu = reinterpret_cast<const float*>(stage + CPU_HDR);
    __half* h = reinterpret_cast<__half*>(hb + CPU_HB_HDR);
    unsigned sum = 0;
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) nr * ffs; e += (size_t) gridDim.x * blockDim.x) {
        const size_t r = e / ffs, i = e % ffs;
        const float g = gu[r * 2 * ffs + i];
        const __half v = __float2half(silu_over(g) * gu[r * 2 * ffs + ffs + i]);
        h[e] = v;
        sum += (unsigned) __half_as_ushort(v) << (16 * (e & 1));   // as the 32-bit words' sum
    }
    __shared__ unsigned ss[256];
    ss[threadIdx.x] = sum;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        sum = 0;
        for (int i = 0; i < 256; ++i) sum += ss[i];
        atomicAdd(&d->hsum, sum);
        if (atomicAdd(&d->hcnt, 1u) == gridDim.x - 1) {
            d->hcnt = 0u;
            unsigned* w = reinterpret_cast<unsigned*>(hb);
            w[1] = atomicExch(&d->hsum, 0u);
            w[2] = (unsigned) nr;
            __threadfence_system();
            *reinterpret_cast<volatile unsigned*>(hb) = want;
            __threadfence_system();
        }
    }
}
// FNX_CPU_CHECK with FNX_CPU_EXACT, after the take: a block a row of the CPU's list: its gate|up sums against the card's own of that
// pair (gu, [P][2 * ffs]) and the SwiGLU the card made of them against the card's own (h16, [P][ffs]); values whose bits differ counted
__global__ void __launch_bounds__(256) k_cpu_check_gh(CpuDev* d, const char* __restrict__ stage, const char* __restrict__ gstage, const __half* __restrict__ hc, int ffs,
                                                      const float* __restrict__ gu, const __half* __restrict__ h16) {
    if (d->want == 0 || (d->err & 1u)) return;
    const int* hd = reinterpret_cast<const int*>(stage);
    const int* ghd = reinterpret_cast<const int*>(gstage);
    const int r = blockIdx.x, tid = threadIdx.x;
    if (r >= hd[0] || (unsigned) hd[1] != d->want || (unsigned) ghd[1] != d->want || ghd[0] != hd[0]) return;
    const int pair = hd[8 + r];
    const float* a = reinterpret_cast<const float*>(gstage + CPU_HDR) + (size_t) r * 2 * ffs;
    const float* b = gu + (size_t) pair * 2 * ffs;
    unsigned ng = 0, nh = 0;
    for (int i = tid; i < 2 * ffs; i += 256) ng += __float_as_uint(a[i]) != __float_as_uint(b[i]);
    for (int i = tid; i < ffs; i += 256) nh += __half_as_ushort(hc[(size_t) r * ffs + i]) != __half_as_ushort(h16[(size_t) pair * ffs + i]);
    __shared__ unsigned sg[256], sh[256];
    sg[tid] = ng, sh[tid] = nh;
    __syncthreads();
    if (tid == 0) {
        for (int i = 1; i < 256; ++i) ng += sg[i], nh += sh[i];
        if (ng) atomicAdd(&d->gdiff, ng);
        if (nh) atomicAdd(&d->hdiff, nh);
    }
}

// a block a row of the staging buffer: into its pair's row of part (check: compared with it instead)
__global__ void __launch_bounds__(256) k_cpu_scatter(const CpuDev* d, const char* __restrict__ stage, float* part, int check, CpuDev* dw) {
    if (d->want == 0 || (d->err & 1u)) return;
    const int* hd = reinterpret_cast<const int*>(stage);
    const int r = blockIdx.x, tid = threadIdx.x;
    if (r >= hd[0] || (unsigned) hd[1] != d->want) return;
    const float4* src = reinterpret_cast<const float4*>(stage + CPU_HDR) + (size_t) r * (N / 4);
    float4* dst = reinterpret_cast<float4*>(part) + (size_t) hd[8 + r] * (N / 4);
    if (!check) {
        for (int i = tid; i < N / 4; i += 256) dst[i] = src[i];
        return;
    }
    __shared__ float sd[256], sm[256];
    __shared__ unsigned sb[256];
    float md = 0.f, mm = 0.f;
    unsigned nb = 0;
    for (int i = tid; i < N / 4; i += 256) {
        const float4 a = src[i], c = dst[i];
        md = fmaxf(fmaxf(fmaxf(md, fabsf(a.x - c.x)), fabsf(a.y - c.y)), fmaxf(fabsf(a.z - c.z), fabsf(a.w - c.w)));
        mm = fmaxf(fmaxf(fmaxf(mm, fabsf(c.x)), fabsf(c.y)), fmaxf(fabsf(c.z), fabsf(c.w)));
        nb += (__float_as_uint(a.x) != __float_as_uint(c.x)) + (__float_as_uint(a.y) != __float_as_uint(c.y)) +
              (__float_as_uint(a.z) != __float_as_uint(c.z)) + (__float_as_uint(a.w) != __float_as_uint(c.w));
    }
    sd[tid] = md, sm[tid] = mm, sb[tid] = nb;
    __syncthreads();
    if (tid == 0) {
        for (int i = 1; i < 256; ++i) md = fmaxf(md, sd[i]), mm = fmaxf(mm, sm[i]), nb += sb[i];
        if (nb) atomicAdd(&dw->bitdiff, nb);
        // positive floats order as their bits do
        atomicMax(reinterpret_cast<unsigned*>(&dw->chk[0]), __float_as_uint(md));
        atomicMax(reinterpret_cast<unsigned*>(&dw->chk[1]), __float_as_uint(mm));
        atomicAdd(&dw->chk[2], 1.0f);
        if (mm > 0.f) atomicMax(reinterpret_cast<unsigned*>(&dw->chk[3]), __float_as_uint(md / mm));
    }
}

int cpu_node_of(int dev) {   // the card's NUMA node from sysfs, -1 unknown
    char id[64] = {};
    if (cudaDeviceGetPCIBusId(id, sizeof id, dev) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    for (char* c = id; *c; ++c) *c = (char) std::tolower((unsigned char) *c);
    std::ifstream f(std::string("/sys/bus/pci/devices/") + id + "/numa_node");
    int node = -1;
    if (!(f >> node)) return -1;
    return node;
}

inline void cpu_pause() {
#if defined(__x86_64__)
    __builtin_ia32_pause();
#endif
    cpux::warm256();
}
// the sum of the mailbox's 32-bit words, as box_copy adds them on the card: the list, the picks, the rows
unsigned box_sum(const CpuBox* box, int n, int T) {
    unsigned sum = 0;
    for (int i = 0; i < n; ++i) sum += (unsigned) box->list[i];
    for (int i = 0; i < T * TOPK; ++i) sum += (unsigned) box->topi[i];
    const uint32_t* x = reinterpret_cast<const uint32_t*>(box->x);
    unsigned sx = 0;
    for (size_t i = 0; i < (size_t) T * (N / 2); ++i) sx += x[i];
    return sum + sx;
}
// the card's SwiGLU rows in mapped memory against the count and the sum of 32-bit words it wrote with them
bool hb_ok(const unsigned char* hb, int nr, int ffs) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(hb);
    if (w[2] != (unsigned) nr) return false;
    const uint32_t* h = reinterpret_cast<const uint32_t*>(hb + CPU_HB_HDR);
    unsigned sum = 0;
    for (size_t i = 0; i < (size_t) nr * (ffs / 2); ++i) sum += h[i];
    return sum == w[1];
}

struct DnProbe {
    CpuStat* st;
    int T;
    std::chrono::steady_clock::time_point t0;
};
void CUDART_CB dn_probe_done(cudaStream_t, cudaError_t, void* p) {
    auto* q = static_cast<DnProbe*>(p);
    q->st->dnp_copy_ns[q->T].fetch_add((unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - q->t0).count(),
                                       std::memory_order_relaxed);
    q->st->dnp_n[q->T].fetch_add(1, std::memory_order_relaxed);
    delete q;
}
// the card's lists, one after another: its half of each listed expert for the tokens that picked it, then the rows to the card
void cpu_loop(CpuCard* c) {
    cudaSetDevice(c->dev);
    dma_pin(c->dev);
    cpux::Pool pool(c->threads - 1, c->cpus);
    std::vector<cpux::Expert> ex(DMA_MAX);
    int* hd = reinterpret_cast<int*>(c->stage_h);
    float* rows = reinterpret_cast<float*>(c->stage_h + CPU_HDR);
    unsigned last = 0;
    static const bool reuse = [] { const char* e = std::getenv("FNX_CPU_REUSE"); return e && std::atoi(e) != 0; }();
    std::vector<uint32_t> seen_at(reuse ? (size_t) NL * NEXP : 0, 0u);
    uint32_t listed = 0;
    auto idle_since = std::chrono::steady_clock::now();
    while (!c->stop.load(std::memory_order_relaxed)) {
        const unsigned s = c->box->seq;
        if (s == last) {
            if (std::chrono::steady_clock::now() - idle_since > std::chrono::milliseconds(20)) std::this_thread::sleep_for(std::chrono::microseconds(100));
            else cpux::warm256();   // FNX_CPU_WARM: this thread computes with the pool
            continue;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const auto t0 = std::chrono::steady_clock::now();
        int n = std::min(c->box->n, DMA_MAX), T = std::min(c->box->T, MAXT), layer = c->box->layer;
        // the mailbox against the card's sum of what it wrote: read again until it matches (words that land after the list's number)
        if (box_sum(c->box, n, T) != c->box->sum) {
            bool ok = false;
            const auto give_up = c->post_rows ? std::chrono::milliseconds(2) : std::chrono::milliseconds(3000);
            while (!ok && std::chrono::steady_clock::now() - t0 < give_up) {
                cpu_pause();
                std::atomic_thread_fence(std::memory_order_acquire);
                n = std::min(c->box->n, DMA_MAX), T = std::min(c->box->T, MAXT), layer = c->box->layer;
                ok = box_sum(c->box, n, T) == c->box->sum;
            }
            (ok ? c->st.box_again : c->st.box_bad).fetch_add(1, std::memory_order_relaxed);
        }
        if (layer < 0 || layer >= NL) layer = 0, n = 0;   // a mailbox that never matched: nothing to compute from it
        c->st.ns_box_t[T].fetch_add((unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(),
                                    std::memory_order_relaxed);
        cpux::Work w;
        w.gu_type = c->gu_type[layer], w.dn_type = c->dn_type[layer];
        w.ffs = c->ffs, w.N = N;
        w.out_stride = N;
        cpux::Pool::Rows xr;   // the window's rows: the pool converts them (a row a chunk) before the experts
        xr.x16 = c->box->x, xr.T = T, xr.xf = c->x.data();
        float* grows = c->exact ? reinterpret_cast<float*>(c->gstage_h + CPU_HDR) : nullptr;
        int nr = 0, m = 0;
        for (int i = 0; i < n; ++i) {
            const int g = c->box->list[i], e = g - layer * NEXP;
            cpux::Expert& x = ex[(size_t) m];
            x.gu = reinterpret_cast<const uint8_t*>(c->host_gu[g]);
            x.dn = reinterpret_cast<const uint8_t*>(c->host_dn[g]);
            x.n = 0;
            x.h = c->h.data() + (size_t) nr * c->ffs;
            x.out = rows + (size_t) nr * N;
            x.gu_out = grows ? grows + (size_t) nr * 2 * c->ffs : nullptr;
            for (int p = 0; p < T * TOPK && x.n < MAXT; ++p)
                if (c->box->topi[p] == e) {
                    x.rows[x.n++] = p / TOPK;
                    hd[8 + nr++] = p;
                }
            if (x.n && x.gu && x.dn) ++m;
            else nr -= x.n;   // not a routed expert of this window, or one without a host copy: nothing to compute (its rows stay as they are)
            if (reuse && x.n && g >= 0 && g < NL * NEXP) {
                ++listed;
                if (const uint32_t was = seen_at[(size_t) g]) {
                    const uint32_t d = listed - was;
                    for (int k = 0; k < 5; ++k)
                        if (d <= (250u << k)) c->st.reuse_hit[k].fetch_add(1, std::memory_order_relaxed);
                }
                seen_at[(size_t) g] = listed;
                c->st.reuse_n.fetch_add(1, std::memory_order_relaxed);
            }
        }
        w.ex = ex.data(), w.m = m;
        const bool probe = c->exact && c->dnprobe && m > 0;
        if (probe) {
            const size_t dnb = cpux::row_bytes(w.dn_type, c->ffs) * (size_t) N;
            for (int i = 0; i < m; ++i) cudaMemcpyAsync(c->dn_scratch, ex[(size_t) i].dn, dnb, cudaMemcpyHostToDevice, c->ds);
            cudaStreamAddCallback(c->ds, dn_probe_done, new DnProbe{&c->st, T, t0}, 0);
        }
        if (c->exact) {   // gate|up here, the SwiGLU on the card (k_cpu_swiglu: its gflag wait, then its h back), down here
            w.exact = true, w.gu_lanes = c->gu_lanes;
            pool.run_gu(w, &xr);
            int* ghd = reinterpret_cast<int*>(c->gstage_h);
            const auto th0 = std::chrono::steady_clock::now();
            c->st.ns_gu_t[T].fetch_add((unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(th0 - t0).count(), std::memory_order_relaxed);
            if (c->gcopy) {
                ghd[0] = nr, ghd[1] = (int) s;
                cudaMemcpyAsync(c->gstage_d, c->gstage_h, CPU_HDR + (size_t) nr * 2 * c->ffs * sizeof(float), cudaMemcpyHostToDevice, c->cs);
                cuStreamWriteValue32((CUstream) c->cs, (CUdeviceptr) &c->d->gflag, s, 0);
            } else {   // the card fetches them (k_cpu_gpull): the count and the rows' sum, then the list's number
                unsigned gsum = 0;
                const uint32_t* gw = reinterpret_cast<const uint32_t*>(c->gstage_h + CPU_HDR);
                for (size_t i = 0, n32 = (size_t) nr * 2 * c->ffs; i < n32; ++i) gsum += gw[i];
                ghd[0] = nr, ghd[2] = (int) gsum;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                *reinterpret_cast<volatile int*>(&ghd[1]) = (int) s;
                std::atomic_thread_fence(std::memory_order_seq_cst);
            }
            const volatile unsigned* hseq = reinterpret_cast<const volatile unsigned*>(c->hb);
            bool ok = true;
            for (unsigned spin = 0; (int) (*hseq - s) < 0; ++spin) {
                if ((spin & 1023) == 0 && std::chrono::steady_clock::now() - th0 > std::chrono::seconds(3)) {
                    ok = false;
                    break;
                }
                cpu_pause();
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            if (ok && !hb_ok(c->hb, nr, c->ffs)) {   // the rows against the card's sum of them, as the mailbox
                bool good = false;
                const auto give_up = c->hb_direct ? std::chrono::milliseconds(2) : std::chrono::milliseconds(3000);
                const auto tv = std::chrono::steady_clock::now();
                while (!good && std::chrono::steady_clock::now() - tv < give_up) {
                    cpu_pause();
                    std::atomic_thread_fence(std::memory_order_acquire);
                    good = hb_ok(c->hb, nr, c->ffs);
                }
                (good ? c->st.h_again : c->st.h_bad).fetch_add(1, std::memory_order_relaxed);
            }
            const auto th1 = std::chrono::steady_clock::now();
            const unsigned long long nh = (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(th1 - th0).count();
            c->st.ns_hand.fetch_add(nh, std::memory_order_relaxed);
            c->st.ns_hand_t[T].fetch_add(nh, std::memory_order_relaxed);
            if (ok) {
                cpux::f16_to_f32(reinterpret_cast<const uint16_t*>(c->hb + CPU_HB_HDR), c->h.data(), (size_t) nr * c->ffs);
                pool.run_down(w);
                c->st.ns_dn_t[T].fetch_add((unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - th1).count(),
                                           std::memory_order_relaxed);
            } else {   // the card did not answer: no rows (k_cpu_wait still gets its word)
                nr = 0;
                c->st.h_timeouts.fetch_add(1, std::memory_order_relaxed);
            }
        } else pool.run(w, &xr);
        hd[0] = nr, hd[1] = (int) s;
        const auto t1 = std::chrono::steady_clock::now();
        if (probe) c->st.dnp_cpu_ns[T].fetch_add((unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(), std::memory_order_relaxed);
        cudaMemcpyAsync(c->stage_d, c->stage_h, CPU_HDR + (size_t) nr * N * sizeof(float), cudaMemcpyHostToDevice, c->cs);
        cuStreamWriteValue32((CUstream) c->cs, (CUdeviceptr) &c->d->flag, s, 0);
        const auto t2 = std::chrono::steady_clock::now();
        const unsigned long long nw = (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        CpuStat& st = c->st;
        st.lists.fetch_add(1, std::memory_order_relaxed);
        st.experts.fetch_add((unsigned long long) m, std::memory_order_relaxed);
        st.rows.fetch_add((unsigned long long) nr, std::memory_order_relaxed);
        st.ns_work.fetch_add(nw, std::memory_order_relaxed);
        st.ns_send.fetch_add((unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(), std::memory_order_relaxed);
        st.by_t[T].fetch_add(1, std::memory_order_relaxed);
        st.ex_t[T].fetch_add((unsigned long long) m, std::memory_order_relaxed);
        st.ns_t[T].fetch_add(nw, std::memory_order_relaxed);
        if (nw > st.worst_ns.load(std::memory_order_relaxed)) st.worst_ns.store(nw, std::memory_order_relaxed);
        if (const unsigned long long pl = pool.late()) st.pool_late.fetch_add(pl, std::memory_order_relaxed);
        {
            const cpux::Pool::Stat ps = pool.stats();
            st.pool_busy_t[T].fetch_add(ps.busy_ns, std::memory_order_relaxed), st.pool_wall_t[T].fetch_add(ps.wall_ns, std::memory_order_relaxed);
            st.pool_first_t[T].fetch_add(ps.first_ns, std::memory_order_relaxed), st.pool_rounds_t[T].fetch_add(ps.rounds, std::memory_order_relaxed);
        }
        last = s;
        idle_since = std::chrono::steady_clock::now();
    }
}

}  // namespace

bool p100_cpu_on() {
    static const bool on = [] {
        const char* e = std::getenv("FNX_CPU_EXPERTS");
        return e && std::atoi(e) != 0;
    }();
    return on;
}
bool p100_cpu_check() {
    static const bool on = [] {
        const char* e = std::getenv("FNX_CPU_CHECK");
        return e && std::atoi(e) != 0;
    }();
    return on;
}
bool p100_cpu_exact_env() {   // FNX_CPU_EXACT=1: the CPU's rows bit for bit the card's (cpu_experts.hpp); off by default
    static const bool on = [] {
        const char* e = std::getenv("FNX_CPU_EXACT");
        return e && std::atoi(e) != 0;
    }();
    return on;
}
bool p100_cpu_type_ok(uint32_t gu_type, uint32_t dn_type, int ffs) { return cpux::type_ok(gu_type, N) && cpux::type_ok(dn_type, ffs); }

int p100_cpu_node(int dev) { return cpu_node_of(dev); }

// rank, share: this card is the rank-th of the `share` cards of the engine that hang from its socket: it takes that part of the socket's cores
void* p100_cpu_start(int dev, int rank, int share, const char* const* host_gu, const char* const* host_dn, const uint32_t* gu_type, const uint32_t* dn_type, int ffs) {
    auto* c = new CpuCard;
    c->dev = dev;
    c->node = cpu_node_of(dev);
    c->host_gu = host_gu, c->host_dn = host_dn, c->ffs = ffs;
    for (int l = 0; l < NL; ++l) c->gu_type[l] = gu_type[l], c->dn_type[l] = dn_type[l];
    const size_t stage = CPU_HDR + (size_t) DMA_MAX * N * sizeof(float);
    if (cudaHostAlloc((void**) &c->box, sizeof(CpuBox), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &c->box_d, c->box, 0) != cudaSuccess || cudaMalloc(&c->d, sizeof(CpuDev)) != cudaSuccess ||
        cudaMemset(c->d, 0, sizeof(CpuDev)) != cudaSuccess || cudaHostAlloc((void**) &c->stage_h, stage, cudaHostAllocPortable) != cudaSuccess ||
        cudaMalloc(&c->stage_d, stage) != cudaSuccess || cudaMemset(c->stage_d, 0, stage) != cudaSuccess ||
        cudaStreamCreateWithFlags(&c->cs, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error(std::string("FNX_CPU_EXPERTS: ") + cudaGetErrorString(cudaGetLastError()));
    std::memset((void*) c->box, 0, sizeof(CpuBox));
    std::memset(c->stage_h, 0, stage);
    if (p100_cpu_exact_env()) {
        bool ok = true;
        for (int l = 0; l < NL; ++l) ok = ok && cpux::exact_ok(gu_type[l], dn_type[l]);
        c->gu_lanes = FNX_ROW_G;   // the card's lanes a gate|up row in phases 0 and 1 (p100_moe_rows.cuh)
        if (!ok || ffs % 8 || (c->gu_lanes != 1 && c->gu_lanes != 2 && c->gu_lanes != 4 && c->gu_lanes != 8))
            std::fprintf(stderr, "fnx: FNX_CPU_EXACT: %s, so the CPU's rows stay as without it (fp32 sums, not bit-exact)\n",
                         ok ? "FNX_ROW_G is not 1, 2, 4 or 8 (or an expert's width is not a multiple of 8)"
                            : "a layer's expert types have no exact steps on the CPU (gate|up Q4_K or Q5_K, down IQ4_NL, Q5_1 or Q8_0)");
        else {
            const size_t gst = CPU_HDR + (size_t) DMA_MAX * 2 * ffs * sizeof(float), hbs = CPU_HB_HDR + (size_t) DMA_MAX * ffs * sizeof(uint16_t);
            if (cudaHostAlloc((void**) &c->gstage_h, gst, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &c->gstage_m, c->gstage_h, 0) != cudaSuccess || cudaMalloc(&c->gstage_d, gst) != cudaSuccess ||
                cudaMemset(c->gstage_d, 0, gst) != cudaSuccess || cudaHostAlloc((void**) &c->hb, hbs, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &c->hb_d, c->hb, 0) != cudaSuccess || cudaMalloc(&c->hdev, hbs) != cudaSuccess ||
                cudaMemset(c->hdev, 0, hbs) != cudaSuccess)
                throw std::runtime_error(std::string("FNX_CPU_EXACT: ") + cudaGetErrorString(cudaGetLastError()));
            std::memset(c->gstage_h, 0, gst);
            std::memset(c->hb, 0, hbs);
            c->exact = true;
            const char* hd = std::getenv("FNX_CPU_HB_DIRECT");
            c->hb_direct = hd && std::atoi(hd) != 0;
            if (const char* wm = std::getenv("FNX_CPU_WARM"); wm && std::atoi(wm) != 0) {
                cpux::keep_warm(true);
                if (rank == 0) std::fprintf(stderr, "fnx: FNX_CPU_WARM: the pools' waiting threads run 256-bit FMAs, so a list after a pause starts at full speed\n");
            }
            const char* gc = std::getenv("FNX_CPU_GCOPY");
            c->gcopy = gc && std::atoi(gc) != 0;
            if (const char* dp = std::getenv("FNX_CPU_DNPROBE"); dp && std::atoi(dp) != 0) {
                if (cudaStreamCreateWithFlags(&c->ds, cudaStreamNonBlocking) != cudaSuccess || cudaMalloc(&c->dn_scratch, (size_t) N * (size_t) ffs * 2) != cudaSuccess)
                    throw std::runtime_error(std::string("FNX_CPU_DNPROBE: ") + cudaGetErrorString(cudaGetLastError()));
                c->dnprobe = true;
                std::fprintf(stderr, "fnx: FNX_CPU_EXPERTS: card %d: FNX_CPU_DNPROBE: every list's down weights also copied to a scratch buffer and timed "
                             "(measuring only; the rows are the CPU's as ever)\n", dev);
            }
        }
    }
    {
        const char* pr = std::getenv("FNX_CPU_POST_ROWS");
        c->post_rows = pr && std::atoi(pr) != 0;
        if (c->gcopy)
            std::fprintf(stderr, "fnx: FNX_CPU_EXPERTS: card %d: FNX_CPU_GCOPY: the CPU's gate|up rows to the card by a copy on its copy queue (measuring only)\n", dev);
        if (c->post_rows || c->hb_direct)
            std::fprintf(stderr, "fnx: FNX_CPU_EXPERTS: card %d: %s%s (measuring only: the host thread's sums count what does not arrive whole)\n", dev,
                         c->post_rows ? "FNX_CPU_POST_ROWS: the mailbox from a block a row; " : "", c->hb_direct ? "FNX_CPU_HB_DIRECT: the SwiGLU rows from every block" : "");
    }
    for (int t = 0; t <= MAXT; ++t) c->st.by_t[t].store(0), c->st.ex_t[t].store(0), c->st.ns_t[t].store(0);
    c->x.assign((size_t) MAXT * N, 0.f);
    c->h.assign((size_t) DMA_MAX * ffs, 0.f);
    // the socket's cores (a machine without NUMA nodes in sysfs: node 0's), less the first two (the engine's card threads, the copier and
    // this card's host thread run there), in `share` equal parts
    std::vector<int> cores = cpux::node_cores(c->node >= 0 ? c->node : 0);
    if (cores.size() > 4) cores.erase(cores.begin(), cores.begin() + 2);
    share = std::max(1, share), rank = std::max(0, std::min(rank, share - 1));
    const size_t part = cores.size() / (size_t) share;
    if (part >= 1) c->cpus.assign(cores.begin() + (long) (part * (size_t) rank), cores.begin() + (long) (part * (size_t) (rank + 1)));
    const char* e = std::getenv("FNX_CPU_THREADS");
    const int dflt = c->cpus.empty() ? 4 : std::min<int>(16, (int) c->cpus.size());
    c->threads = std::max(1, std::min(64, e && std::atoi(e) > 0 ? std::atoi(e) : dflt));
    if (const size_t n0 = c->cpus.size(); n0 && (size_t) c->threads > n0)   // more threads than cores (FNX_CPU_THREADS): the cores' other hyperthreads next
        for (size_t i = 0; i < n0; ++i)
            for (int sib : cpux::core_siblings(c->cpus[i])) c->cpus.push_back(sib);
    std::fprintf(stderr, "fnx: FNX_CPU_EXPERTS: card %d (NUMA node %d): the window's experts that are not in VRAM computed on %d CPU threads%s, %s%s\n", dev, c->node,
                 c->threads, c->cpus.empty() ? " (not pinned)" : " of its socket",
                 cpux::have_simd() ? "AVX2" : "no AVX2: the scalar steps (slow)",
                 c->exact ? "; FNX_CPU_EXACT: the card's own arithmetic, its SwiGLU on the card (rows bit for bit the card's)" : "");
    c->th = std::thread(cpu_loop, c);
    return c;
}

void p100_cpu_stop(void* h) {
    auto* c = (CpuCard*) h;
    if (!c) return;
    c->stop = true;
    if (c->th.joinable()) c->th.join();
    cudaSetDevice(c->dev);
    cudaStreamSynchronize(c->cs);
    cudaStreamDestroy(c->cs);
    cudaFree(c->stage_d);
    cudaFreeHost(c->stage_h);
    cudaFree(c->d);
    cudaFreeHost((void*) c->box);
    if (c->gstage_h) cudaFreeHost(c->gstage_h);
    if (c->gstage_d) cudaFree(c->gstage_d);
    if (c->hb) cudaFreeHost(c->hb);
    if (c->hdev) cudaFree(c->hdev);
    delete c;
}

bool p100_cpu_exact_on(void* h) { return h && ((CpuCard*) h)->exact; }
// the test's hook (tools/cpu_exact_gpu_test.cu): the hand-off's SwiGLU on nr rows of gate|up sums in device memory ([nr][2 * ffs]), its
// h (f16, [nr][ffs]) into device memory. Both forms run: the one-block kernel the engine uses (device buffers stand in for the two
// mapped ones) and k_cpu_swiglu of the measuring switches; it throws if their h differ, and returns the one-block kernel's
void p100_cpu_swiglu_probe(const float* gu, int nr, int ffs, uint16_t* h, void* st) {
    cudaStream_t s = (cudaStream_t) st;
    CpuDev* d = nullptr;
    char* stage = nullptr;
    char* gm = nullptr;
    char* gd = nullptr;
    __half* hb = nullptr;
    __half* h1 = nullptr;
    unsigned char* post = nullptr;
    const size_t gb = (size_t) nr * 2 * ffs * sizeof(float), hbytes = (size_t) nr * ffs * sizeof(uint16_t);
    const unsigned one = 1;
    std::vector<uint32_t> host(gb / 4);
    std::vector<uint16_t> a(hbytes / 2), b2(hbytes / 2);
    if (cudaMemcpy(host.data(), gu, gb, cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("cpu swiglu probe: the rows");
    unsigned sum = 0;
    for (uint32_t v : host) sum += v;
    const int hd[3] = {nr, 1, (int) sum};
    if (cudaMalloc(&d, sizeof(CpuDev)) != cudaSuccess || cudaMemset(d, 0, sizeof(CpuDev)) != cudaSuccess ||
        cudaMemcpy(&d->want, &one, sizeof one, cudaMemcpyHostToDevice) != cudaSuccess || cudaMalloc(&stage, CPU_HDR + gb) != cudaSuccess ||
        cudaMemcpy(stage, hd, sizeof hd, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(stage + CPU_HDR, gu, gb, cudaMemcpyDeviceToDevice) != cudaSuccess || cudaMalloc(&hb, hbytes) != cudaSuccess ||
        cudaMalloc(&gm, CPU_HDR + gb) != cudaSuccess || cudaMemcpy(gm, stage, CPU_HDR + gb, cudaMemcpyDeviceToDevice) != cudaSuccess ||
        cudaMalloc(&gd, CPU_HDR + gb) != cudaSuccess || cudaMemset(gd, 0, CPU_HDR + gb) != cudaSuccess || cudaMalloc(&h1, hbytes) != cudaSuccess ||
        cudaMalloc(&post, CPU_HB_HDR + hbytes) != cudaSuccess || cudaMemset(post, 0, CPU_HB_HDR + hbytes) != cudaSuccess)
        throw std::runtime_error(std::string("cpu swiglu probe: ") + cudaGetErrorString(cudaGetLastError()));
    k_cpu_swiglu<<<40, 256, 0, s>>>(d, stage, ffs, hb);
    k_cpu_swiglu1<<<1, 256, 0, s>>>(d, gm, gd, ffs, h1, post);
    cudaMemcpyAsync(h, h1, hbytes, cudaMemcpyDeviceToDevice, s);
    cudaError_t e = cudaStreamSynchronize(s);
    unsigned head[3] = {0, 0, 0};
    if (e == cudaSuccess) e = cudaMemcpy(a.data(), hb, hbytes, cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(b2.data(), h1, hbytes, cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(head, post, sizeof head, cudaMemcpyDeviceToHost);
    std::vector<uint16_t> posted(hbytes / 2);
    if (e == cudaSuccess) e = cudaMemcpy(posted.data(), post + CPU_HB_HDR, hbytes, cudaMemcpyDeviceToHost);
    cudaFree(post);
    cudaFree(h1);
    cudaFree(gd);
    cudaFree(gm);
    cudaFree(hb);
    cudaFree(stage);
    cudaFree(d);
    if (e != cudaSuccess) throw std::runtime_error(std::string("cpu swiglu probe: ") + cudaGetErrorString(e));
    if (a != b2) throw std::runtime_error("cpu swiglu probe: the one-block kernel's h is not k_cpu_swiglu's");
    if (head[0] != 1u || head[2] != (unsigned) nr || posted != b2) throw std::runtime_error("cpu swiglu probe: the one-block kernel posted other rows than it computed");
}
// FNX_CPU_EXACT: on `st` (the stream the list was posted on, or a fork of the one it was published on): wait for the CPU's gate|up
// rows, then the card's SwiGLU of them back to the host thread. Nothing without FNX_CPU_EXACT
void p100_cpu_swiglu(void* h, void* st) {
    auto* c = (CpuCard*) h;
    if (!c || !c->exact) return;
    if (!c->gcopy && !c->hb_direct)   // the whole hand-off in one block
        k_cpu_swiglu1<<<1, 256, 0, (cudaStream_t) st>>>(c->d, c->gstage_m, c->gstage_d, c->ffs, reinterpret_cast<__half*>(c->hdev), c->hb_d);
    else {   // the measuring switches: the rows by the copy queue (FNX_CPU_GCOPY), h from every block (FNX_CPU_HB_DIRECT); three launches
        if (c->gcopy) k_cpu_wait_g<<<1, 32, 0, (cudaStream_t) st>>>(c->d);
        else k_cpu_gpull<<<1, 256, 0, (cudaStream_t) st>>>(c->d, c->gstage_m, c->gstage_d, c->ffs);
        if (c->hb_direct) k_cpu_swiglu_direct<<<40, 256, 0, (cudaStream_t) st>>>(c->d, c->gstage_d, c->ffs, c->hb_d);
        else {
            k_cpu_swiglu<<<40, 256, 0, (cudaStream_t) st>>>(c->d, c->gstage_d, c->ffs, reinterpret_cast<__half*>(c->hdev));
            k_cpu_hpost<<<1, 256, 0, (cudaStream_t) st>>>(c->d, c->gstage_d, c->ffs, reinterpret_cast<const __half*>(c->hdev), c->hb_d);
        }
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("cpu swiglu: ") + cudaGetErrorString(e));
}
// FNX_CPU_CHECK with FNX_CPU_EXACT, on `st` after the take: the CPU's gate|up sums and the card's SwiGLU of them against the card's
// own of the same pairs (gu [P][2 * ffs], h16 [P][ffs]): the stage a row that is not the card's went wrong at. Not with
// FNX_CPU_HB_DIRECT (h is then only in mapped memory)
void p100_cpu_check_stages(void* h, const float* gu, const uint16_t* h16, void* st) {
    auto* c = (CpuCard*) h;
    if (!c || !c->exact || c->hb_direct) return;
    k_cpu_check_gh<<<(unsigned) DMA_MAX, 256, 0, (cudaStream_t) st>>>(c->d, c->stage_d, c->gstage_d, reinterpret_cast<const __half*>(c->hdev), c->ffs, gu,
                                                                     reinterpret_cast<const __half*>(h16));
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("cpu check stages: ") + cudaGetErrorString(e));
}

void p100_cpu_publish(void* h, Job* gu, Job* dn, const int* njobs, int layer, const int* list, const int* topi, int T, const uint16_t* xh, bool mark, void* st) {
    auto* c = (CpuCard*) h;
    k_cpu_publish<<<1, 256, 0, (cudaStream_t) st>>>(gu, dn, njobs, layer, list, topi, T, xh, c->box_d, c->d, mark ? 1 : 0);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("cpu publish: ") + cudaGetErrorString(e));
}

void* p100_cpu_dev(void* h) { return ((CpuCard*) h)->d; }
// the window's rows (f16), the list and the picks to the mailbox, on `st`, a stream beside the card's own
void p100_cpu_post(void* h, int layer, const int* list, const int* topi, int T, const uint16_t* xh, void* st) {
    auto* c = (CpuCard*) h;
    if (c->post_rows) k_cpu_post_rows<<<(unsigned) T, 256, 0, (cudaStream_t) st>>>(layer, list, topi, T, xh, c->box_d, c->d);
    else k_cpu_post<<<1, 256, 0, (cudaStream_t) st>>>(layer, list, topi, T, xh, c->box_d, c->d);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("cpu post: ") + cudaGetErrorString(e));
}

void p100_cpu_take(void* h, float* part, int T, bool check, void* st) {
    auto* c = (CpuCard*) h;
    k_cpu_wait<<<1, 32, 0, (cudaStream_t) st>>>(c->d);
    k_cpu_scatter<<<(unsigned) (T * TOPK), 256, 0, (cudaStream_t) st>>>(c->d, c->stage_d, part, check ? 1 : 0, c->d);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("cpu take: ") + cudaGetErrorString(e));
}

// the counts since the last report, a line a card (the engine's device must be idle: the check's words are read from VRAM)
void p100_cpu_report(void* h, bool check) {
    auto* c = (CpuCard*) h;
    if (!c) return;
    CpuStat& s = c->st;
    const unsigned long long lists = s.lists.exchange(0), ex = s.experts.exchange(0), rows = s.rows.exchange(0), nw = s.ns_work.exchange(0),
                             ns = s.ns_send.exchange(0), worst = s.worst_ns.exchange(0);
    if (!lists) return;
    std::string by;
    for (int t = 1; t <= MAXT; ++t) {
        const unsigned long long n = s.by_t[t].exchange(0), e = s.ex_t[t].exchange(0), w = s.ns_t[t].exchange(0);
        const unsigned long long g = s.ns_gu_t[t].exchange(0), h = s.ns_hand_t[t].exchange(0), dn = s.ns_dn_t[t].exchange(0);
        const unsigned long long pb = s.pool_busy_t[t].exchange(0), pw = s.pool_wall_t[t].exchange(0), pf = s.pool_first_t[t].exchange(0),
                                 pr = s.pool_rounds_t[t].exchange(0), bx = s.ns_box_t[t].exchange(0);
        if (!n) continue;
        char buf[320];
        if (c->exact) std::snprintf(buf, sizeof buf, " T%d: %llu x %.1f, %.0f us (gate|up %.0f, the card's SwiGLU %.0f, down %.0f; mailbox %.0f; pool: %.1f rounds, "
                                    "%.0f us, threads' work %.0f us, first worker after %.1f us);", t, n, (double) e / n, w / 1e3 / n,
                                    g / 1e3 / n, h / 1e3 / n, dn / 1e3 / n, bx / 1e3 / n, (double) pr / n, pw / 1e3 / n, pb / 1e3 / n, pr ? pf / 1e3 / pr : 0.0);
        else std::snprintf(buf, sizeof buf, " T%d: %llu x %.1f, %.0f us;", t, n, (double) e / n, w / 1e3 / n);
        by += buf;
    }
    std::fprintf(stderr, "fnx: cpu experts, card %d: %llu lists, %.2f experts and %.2f rows a list, %.0f us a list on the CPU (worst %.0f), %.0f us to send;%s\n",
                 c->dev, lists, (double) ex / lists, (double) rows / lists, nw / 1e3 / lists, worst / 1e3, ns / 1e3 / lists, by.c_str());
    const unsigned long long ba = s.box_again.exchange(0), bb = s.box_bad.exchange(0), ha = s.h_again.exchange(0), hbd = s.h_bad.exchange(0);
    if (c->dnprobe) {
        std::string dp;
        for (int t = 1; t <= MAXT; ++t) {
            const unsigned long long n = s.dnp_n[t].exchange(0), cp = s.dnp_copy_ns[t].exchange(0), cu = s.dnp_cpu_ns[t].exchange(0);
            if (!n) continue;
            char buf[96];
            std::snprintf(buf, sizeof buf, " T%d: %llu lists, %.0f us / %.0f us;", t, n, cp / 1e3 / n, cu / 1e3 / n);
            dp += buf;
        }
        if (!dp.empty())
            std::fprintf(stderr, "fnx: FNX_CPU_DNPROBE, card %d: from a list's arrival to its experts' down weights landed on the card / to the CPU's rows done:%s\n",
                         c->dev, dp.c_str());
    }
    if (const unsigned long long rn = s.reuse_n.exchange(0)) {
        unsigned long long hit[5];
        for (int k = 0; k < 5; ++k) hit[k] = s.reuse_hit[k].exchange(0);
        std::fprintf(stderr, "fnx: FNX_CPU_REUSE, card %d: of %llu listed experts %.1f %% had been listed within the last 250 of this card's listed experts, %.1f %% within 500, "
                     "%.1f %% within 1000, %.1f %% within 2000, %.1f %% within 4000\n", c->dev, rn, 100.0 * hit[0] / rn, 100.0 * hit[1] / rn, 100.0 * hit[2] / rn,
                     100.0 * hit[3] / rn, 100.0 * hit[4] / rn);
    }
    if (const unsigned long long pl = s.pool_late.exchange(0); pl || check)
        std::fprintf(stderr, "fnx: cpu experts, card %d: %llu workers came to a finished round of the pool as the next was set up (turned away)\n", c->dev, pl);
    if (c->exact) {
        const unsigned long long hand = s.ns_hand.exchange(0), hto = s.h_timeouts.exchange(0);
        std::fprintf(stderr, "fnx: FNX_CPU_EXACT, card %d: %.0f us a list from the gate|up rows sent to the card's SwiGLU back (in the CPU's time above)%s\n", c->dev,
                     hand / 1e3 / lists, hto ? ", some lists timed out waiting for it" : "");
        static std::atomic<uint64_t> seen{0};   // every card's pool counts into one figure
        const uint64_t now = cpux::exact_ref_runs(), was = seen.exchange(now);
        if (now != was || check)
            std::fprintf(stderr, "fnx: FNX_CPU_EXACT: %llu runs of the CPU's chains left fp16's range and were done again by the oracle (both cards)\n",
                         (unsigned long long) (now - was));
    }
    if (ba || bb || ha || hbd || c->exact || c->post_rows || check)
        std::fprintf(stderr, "fnx: cpu experts, card %d: the card's writes against their sums: mailbox %llu lists read again, %llu never whole; SwiGLU rows %llu read again, "
                     "%llu never whole%s\n", c->dev, ba, bb, ha, hbd, bb || hbd ? " (ROWS COMPUTED FROM WHAT WAS NOT THE CARD'S)" : "");
    if (check) {
        CpuDev d;
        cudaSetDevice(c->dev);
        if (cudaMemcpy(&d, c->d, sizeof d, cudaMemcpyDeviceToHost) == cudaSuccess) {
            std::fprintf(stderr, "fnx: FNX_CPU_CHECK, card %d: %.0f rows compared with the card's own: largest difference %.3g, largest value %.3g, worst row %.3g of its largest value; "
                         "%u values not bit for bit the card's%s\n",
                         c->dev, d.chk[2], d.chk[0], d.chk[1], d.chk[3], d.bitdiff, d.err ? " (a wait timed out)" : "");
            if (c->exact && !c->hb_direct)
                std::fprintf(stderr, "fnx: FNX_CPU_CHECK, card %d, by stage: %u gate|up sums of the CPU's not the card's bits, %u values of the SwiGLU made of them\n", c->dev,
                             d.gdiff, d.hdiff);
            cudaMemset(reinterpret_cast<char*>(c->d) + offsetof(CpuDev, chk), 0, sizeof d.chk);
            cudaMemset(reinterpret_cast<char*>(c->d) + offsetof(CpuDev, bitdiff), 0, 3 * sizeof(unsigned));
        }
    } else {
        unsigned err = 0;
        cudaSetDevice(c->dev);
        if (cudaMemcpy(&err, &c->d->err, sizeof err, cudaMemcpyDeviceToHost) == cudaSuccess && err)
            std::fprintf(stderr, "fnx: FNX_CPU_EXPERTS, card %d: a window waited 3 s for the CPU's rows and went on without them\n", c->dev);
    }
}

}  // namespace fnx
