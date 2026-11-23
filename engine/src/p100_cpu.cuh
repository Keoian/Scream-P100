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
};
// FNX_CPU_EXACT: the card's SwiGLU of the CPU's gate|up rows goes back through mapped host memory: the list's number, then the rows' h (f16)
constexpr size_t CPU_HB_HDR = 64;
static_assert(offsetof(CpuDev, seq) == 0 && offsetof(CpuDev, want) == sizeof(unsigned), "the router's last block writes CpuDev's seq and want as words 0 and 1 (moe.cuh)");
struct CpuStat {
    std::atomic<unsigned long long> lists{0}, experts{0}, rows{0}, ns_work{0}, ns_send{0}, by_t[MAXT + 1], ex_t[MAXT + 1], ns_t[MAXT + 1], worst_ns{0};
    std::atomic<unsigned long long> pool_late{0};   // cpux::Pool::late
    std::atomic<unsigned long long> ns_hand{0}, h_timeouts{0};   // FNX_CPU_EXACT: gate|up sent to the card's h back; waits given up
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
    char* gstage_h = nullptr;   // page-locked: the head (n_rows, seq), then each row's gate sums and up sums
    char* gstage_d = nullptr;
    unsigned char* hb = nullptr;     // mapped: the list's number, then the rows' h (f16)
    unsigned char* hb_d = nullptr;
};

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
    const uint4* src = reinterpret_cast<const uint4*>(xh);
    uint4* dst = reinterpret_cast<uint4*>(box->x);
    for (int i = tid; i < T * (N / 8); i += 256) dst[i] = src[i];
    for (int i = tid; i < k; i += 256) box->list[i] = list[1 + i];
    for (int i = tid; i < T * TOPK; i += 256) box->topi[i] = topi[i];
    __threadfence_system();
    __syncthreads();
    if (tid == 0) {
        box->n = k;
        box->T = T;
        box->layer = layer;
        const unsigned s = ++d->seq;
        d->want = s;
        __threadfence_system();
        box->seq = s;
        __threadfence_system();
    }
}

// the window's rows (a block a row), the plan's list and the picks to the mailbox, then the list's number (the router's last block set it in
// d->want) by the last block to finish: on the third stream, beside the card's own experts
__global__ void __launch_bounds__(256) k_cpu_post(int layer, const int* __restrict__ list, const int* __restrict__ topi, int T, const uint16_t* __restrict__ xh, CpuBox* box,
                                                  CpuDev* d) {
    const int k = min(list[0], DMA_MAX), tid = threadIdx.x, r = blockIdx.x;
    if (k == 0) return;
    const uint4* src = reinterpret_cast<const uint4*>(xh) + (size_t) r * (N / 8);
    uint4* dst = reinterpret_cast<uint4*>(box->x) + (size_t) r * (N / 8);
    for (int i = tid; i < N / 8; i += 256) dst[i] = src[i];
    if (r == 0) {
        for (int i = tid; i < k; i += 256) box->list[i] = list[1 + i];
        for (int i = tid; i < T * TOPK; i += 256) box->topi[i] = topi[i];
    }
    __threadfence_system();
    __syncthreads();
    if (tid == 0 && atomicAdd(&d->cnt, 1u) == (unsigned) (T - 1)) {
        d->cnt = 0u;
        box->n = k;
        box->T = T;
        box->layer = layer;
        __threadfence_system();
        box->seq = d->want;
        __threadfence_system();
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
// FNX_CPU_EXACT: the card's SwiGLU of the CPU's gate|up rows, the very expression of k_swiglu_after (p100_moe_rows.cuh; its __expf and
// division under --use_fast_math are the card's own and cannot be done on a CPU), into mapped host memory as f16; the last block then
// writes the list's number in front of the rows
__global__ void __launch_bounds__(256) k_cpu_swiglu(CpuDev* d, const char* __restrict__ stage, int ffs, unsigned char* hb) {
    const unsigned want = *(volatile unsigned*) &d->want;
    if (want == 0) return;
    const int* hd = reinterpret_cast<const int*>(stage);
    const int nr = (unsigned) hd[1] == want ? hd[0] : 0;
    const float* gu = reinterpret_cast<const float*>(stage + CPU_HDR);
    __half* h = reinterpret_cast<__half*>(hb + CPU_HB_HDR);
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) nr * ffs; e += (size_t) gridDim.x * blockDim.x) {
        const size_t r = e / ffs, i = e % ffs;
        const float g = gu[r * 2 * ffs + i];
        h[e] = __float2half(silu_over(g) * gu[r * 2 * ffs + ffs + i]);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(&d->hcnt, 1u) == gridDim.x - 1) {
        d->hcnt = 0u;
        __threadfence_system();
        *reinterpret_cast<volatile unsigned*>(hb) = want;
        __threadfence_system();
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

// the card's lists, one after another: its half of each listed expert for the tokens that picked it, then the rows to the card
void cpu_loop(CpuCard* c) {
    cudaSetDevice(c->dev);
    dma_pin(c->dev);
    cpux::Pool pool(c->threads - 1, c->cpus);
    std::vector<cpux::Expert> ex(DMA_MAX);
    int* hd = reinterpret_cast<int*>(c->stage_h);
    float* rows = reinterpret_cast<float*>(c->stage_h + CPU_HDR);
    unsigned last = 0;
    auto idle_since = std::chrono::steady_clock::now();
    while (!c->stop.load(std::memory_order_relaxed)) {
        const unsigned s = c->box->seq;
        if (s == last) {
            if (std::chrono::steady_clock::now() - idle_since > std::chrono::milliseconds(20)) std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const auto t0 = std::chrono::steady_clock::now();
        const int n = std::min(c->box->n, DMA_MAX), T = std::min(c->box->T, MAXT), layer = c->box->layer;
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
        }
        w.ex = ex.data(), w.m = m;
        if (c->exact) {   // gate|up here, the SwiGLU on the card (k_cpu_swiglu: its gflag wait, then its h back), down here
            w.exact = true, w.gu_lanes = c->gu_lanes;
            pool.run_gu(w, &xr);
            int* ghd = reinterpret_cast<int*>(c->gstage_h);
            ghd[0] = nr, ghd[1] = (int) s;
            const auto th0 = std::chrono::steady_clock::now();
            cudaMemcpyAsync(c->gstage_d, c->gstage_h, CPU_HDR + (size_t) nr * 2 * c->ffs * sizeof(float), cudaMemcpyHostToDevice, c->cs);
            cuStreamWriteValue32((CUstream) c->cs, (CUdeviceptr) &c->d->gflag, s, 0);
            const volatile unsigned* hseq = reinterpret_cast<const volatile unsigned*>(c->hb);
            bool ok = true;
            for (unsigned spin = 0; (int) (*hseq - s) < 0; ++spin) {
                if ((spin & 1023) == 0 && std::chrono::steady_clock::now() - th0 > std::chrono::seconds(3)) {
                    ok = false;
                    break;
                }
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            c->st.ns_hand.fetch_add((unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - th0).count(),
                                    std::memory_order_relaxed);
            if (ok) {
                cpux::f16_to_f32(reinterpret_cast<const uint16_t*>(c->hb + CPU_HB_HDR), c->h.data(), (size_t) nr * c->ffs);
                pool.run_down(w);
            } else {   // the card did not answer: no rows (k_cpu_wait still gets its word)
                nr = 0;
                c->st.h_timeouts.fetch_add(1, std::memory_order_relaxed);
            }
        } else pool.run(w, &xr);
        hd[0] = nr, hd[1] = (int) s;
        const auto t1 = std::chrono::steady_clock::now();
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
        if (!ok || (c->gu_lanes != 1 && c->gu_lanes != 2 && c->gu_lanes != 4 && c->gu_lanes != 8))
            std::fprintf(stderr, "fnx: FNX_CPU_EXACT: %s, so the CPU's rows stay as without it (fp32 sums, not bit-exact)\n",
                         ok ? "FNX_ROW_G is not 1, 2, 4 or 8" : "a layer's expert types have no exact steps on the CPU (gate|up Q4_K or Q5_K, down IQ4_NL, Q5_1 or Q8_0)");
        else {
            const size_t gst = CPU_HDR + (size_t) DMA_MAX * 2 * ffs * sizeof(float), hbs = CPU_HB_HDR + (size_t) DMA_MAX * ffs * sizeof(uint16_t);
            if (cudaHostAlloc((void**) &c->gstage_h, gst, cudaHostAllocPortable) != cudaSuccess || cudaMalloc(&c->gstage_d, gst) != cudaSuccess ||
                cudaMemset(c->gstage_d, 0, gst) != cudaSuccess || cudaHostAlloc((void**) &c->hb, hbs, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                cudaHostGetDevicePointer((void**) &c->hb_d, c->hb, 0) != cudaSuccess)
                throw std::runtime_error(std::string("FNX_CPU_EXACT: ") + cudaGetErrorString(cudaGetLastError()));
            std::memset(c->gstage_h, 0, gst);
            std::memset(c->hb, 0, hbs);
            c->exact = true;
        }
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
    delete c;
}

bool p100_cpu_exact_on(void* h) { return h && ((CpuCard*) h)->exact; }
// the test's hook (tools/cpu_exact_gpu_test.cu): k_cpu_swiglu itself on nr rows of gate|up sums in device memory ([nr][2 * ffs]), its h
// (f16, [nr][ffs]) into device memory
void p100_cpu_swiglu_probe(const float* gu, int nr, int ffs, uint16_t* h, void* st) {
    cudaStream_t s = (cudaStream_t) st;
    CpuDev* d = nullptr;
    char* stage = nullptr;
    unsigned char* hb = nullptr;
    const size_t gb = (size_t) nr * 2 * ffs * sizeof(float), hbytes = (size_t) nr * ffs * sizeof(uint16_t);
    const unsigned one = 1;
    const int hd[2] = {nr, 1};
    if (cudaMalloc(&d, sizeof(CpuDev)) != cudaSuccess || cudaMemset(d, 0, sizeof(CpuDev)) != cudaSuccess ||
        cudaMemcpy(&d->want, &one, sizeof one, cudaMemcpyHostToDevice) != cudaSuccess || cudaMalloc(&stage, CPU_HDR + gb) != cudaSuccess ||
        cudaMemcpy(stage, hd, sizeof hd, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(stage + CPU_HDR, gu, gb, cudaMemcpyDeviceToDevice) != cudaSuccess || cudaMalloc(&hb, CPU_HB_HDR + hbytes) != cudaSuccess)
        throw std::runtime_error(std::string("cpu swiglu probe: ") + cudaGetErrorString(cudaGetLastError()));
    k_cpu_swiglu<<<40, 256, 0, s>>>(d, stage, ffs, hb);
    cudaMemcpyAsync(h, hb + CPU_HB_HDR, hbytes, cudaMemcpyDeviceToDevice, s);
    const cudaError_t e = cudaStreamSynchronize(s);
    cudaFree(hb);
    cudaFree(stage);
    cudaFree(d);
    if (e != cudaSuccess) throw std::runtime_error(std::string("cpu swiglu probe: ") + cudaGetErrorString(e));
}
// FNX_CPU_EXACT: on `st` (the stream the list was posted on, or a fork of the one it was published on): wait for the CPU's gate|up
// rows, then the card's SwiGLU of them back to the host thread. Nothing without FNX_CPU_EXACT
void p100_cpu_swiglu(void* h, void* st) {
    auto* c = (CpuCard*) h;
    if (!c || !c->exact) return;
    k_cpu_wait_g<<<1, 32, 0, (cudaStream_t) st>>>(c->d);
    k_cpu_swiglu<<<40, 256, 0, (cudaStream_t) st>>>(c->d, c->gstage_d, c->ffs, c->hb_d);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("cpu swiglu: ") + cudaGetErrorString(e));
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
    k_cpu_post<<<(unsigned) T, 256, 0, (cudaStream_t) st>>>(layer, list, topi, T, xh, c->box_d, c->d);
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
        if (!n) continue;
        char buf[96];
        std::snprintf(buf, sizeof buf, " T%d: %llu x %.1f, %.0f us;", t, n, (double) e / n, w / 1e3 / n);
        by += buf;
    }
    std::fprintf(stderr, "fnx: cpu experts, card %d: %llu lists, %.2f experts and %.2f rows a list, %.0f us a list on the CPU (worst %.0f), %.0f us to send;%s\n",
                 c->dev, lists, (double) ex / lists, (double) rows / lists, nw / 1e3 / lists, worst / 1e3, ns / 1e3 / lists, by.c_str());
    if (const unsigned long long pl = s.pool_late.exchange(0); pl || check)
        std::fprintf(stderr, "fnx: cpu experts, card %d: %llu workers came to a finished round of the pool as the next was set up (turned away)\n", c->dev, pl);
    if (c->exact) {
        const unsigned long long hand = s.ns_hand.exchange(0), hto = s.h_timeouts.exchange(0);
        std::fprintf(stderr, "fnx: FNX_CPU_EXACT, card %d: %.0f us a list from the gate|up rows sent to the card's SwiGLU back (in the CPU's time above)%s\n", c->dev,
                     hand / 1e3 / lists, hto ? ", some lists timed out waiting for it" : "");
    }
    if (check) {
        CpuDev d;
        cudaSetDevice(c->dev);
        if (cudaMemcpy(&d, c->d, sizeof d, cudaMemcpyDeviceToHost) == cudaSuccess) {
            std::fprintf(stderr, "fnx: FNX_CPU_CHECK, card %d: %.0f rows compared with the card's own: largest difference %.3g, largest value %.3g, worst row %.3g of its largest value; "
                         "%u values not bit for bit the card's%s\n",
                         c->dev, d.chk[2], d.chk[0], d.chk[1], d.chk[3], d.bitdiff, d.err ? " (a wait timed out)" : "");
            cudaMemset(reinterpret_cast<char*>(c->d) + offsetof(CpuDev, chk), 0, sizeof d.chk);
            cudaMemset(reinterpret_cast<char*>(c->d) + offsetof(CpuDev, bitdiff), 0, sizeof d.bitdiff);
        }
    } else {
        unsigned err = 0;
        cudaSetDevice(c->dev);
        if (cudaMemcpy(&err, &c->d->err, sizeof err, cudaMemcpyDeviceToHost) == cudaSuccess && err)
            std::fprintf(stderr, "fnx: FNX_CPU_EXPERTS, card %d: a window waited 3 s for the CPU's rows and went on without them\n", c->dev);
    }
}

}  // namespace fnx
