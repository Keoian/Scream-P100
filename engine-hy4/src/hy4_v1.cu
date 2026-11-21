// hy4_v1.cu (2026-11-20): Hy4 flash split across both P100s, after v0 (hy4_v0.cu, exact against the numpy reference). Every
// layer runs on both cards: the attention's heads are divided (sparse layers 16 / 16, window layers 32 / 32; each card keeps the
// whole latent cache, the indexer and Engram run on both), every routed and shared expert's FF width is divided (768 / 768:
// gate and up rows, down columns), layer 0's dense FFN too (6912 / 6912), and the output vocabulary (60416 / 60416). Two sums
// across the cards a layer (the attention's and the FFN's output) through page-locked mapped RAM: each card writes its part
// and adds the other's, so both hold the same bits (a + b == b + a). Each card's expert halves live in page-locked RAM and in
// its own VRAM slots, under one host LRU (the same slot on both cards), and copy over its own PCIe link.
//   hy4_v1 --model TRUNK.gguf --tokens tokens.bin [--gen N] [--dump DIR] [--slots S] [--ctx C]
//   hy4_v1 --model TRUNK.gguf --serve SOCKET [--slots S] [--ctx C]
#include "hy4_kern.cuh"
#include "hy4_pf.cuh"
#include "hy4_dec.cuh"
#include "hy4_pf2.cuh"

#include <cuda_profiler_api.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cctype>
#include <csignal>
#include <functional>
#include <future>
#include <thread>

namespace hy4 {

// v1i: our server has two sockets, card 0 on node 0 and card 1 on node 1 (nvidia-smi topo -m: SYS). The calling thread moves to
// card c's node (the card's /sys/bus/pci/devices/<bus id>/numa_node), to that node's CPUs (/sys/devices/system/node/nodeN/cpulist,
// e.g. "0-3,8-11"); page-locked memory it allocates afterwards lands on that node. No node known (-1, one socket): nothing moves.
// HY4_NUMA_OFF=1: nothing moves
static bool numa_on() { return getenv("HY4_NUMA_OFF") == nullptr; }
static int card_node(int c) {
    static int node[8] = {-2, -2, -2, -2, -2, -2, -2, -2};
    if (node[c] != -2) return node[c];
    node[c] = -1;
    char id[32] = {};
    if (cudaDeviceGetPCIBusId(id, sizeof id, c) != cudaSuccess) return -1;
    for (char* p = id; *p; ++p) *p = (char) std::tolower((unsigned char) *p);
    if (FILE* f = std::fopen((std::string("/sys/bus/pci/devices/") + id + "/numa_node").c_str(), "r")) {
        if (std::fscanf(f, "%d", &node[c]) != 1) node[c] = -1;
        std::fclose(f);
    }
    return node[c];
}
static void bind_node(int c) {   // to card c's node
    if (!numa_on()) return;
    const int n = card_node(c);
    if (n < 0) return;
    FILE* f = std::fopen(("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist").c_str(), "r");
    if (!f) return;
    char buf[256] = {};
    const bool ok = std::fgets(buf, sizeof buf, f) != nullptr;
    std::fclose(f);
    if (!ok) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (char* p = buf;;) {
        char* e;
        const long a = std::strtol(p, &e, 10);
        if (e == p) break;
        long b = a;
        if (*e == '-') b = std::strtol(e + 1, &e, 10);
        for (long i = a; i <= b && i < CPU_SETSIZE; ++i) CPU_SET((int) i, &set);
        if (*e != ',') break;
        p = e + 1;
    }
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}
template <class F> static void on_node(int c, F fn) {   // fn on a thread of card c's node, then back
    std::thread t([&]() {
        bind_node(c);
        fn();
    });
    t.join();
}

constexpr int NC = 2;                                  // cards
constexpr int FFE_C = FFE / NC, FFD_C = FFD / NC, NV = 120832, NV_C = NV / NC;
constexpr size_t HALF_GU = (size_t) FFE_C * (LAT / 256) * 98;   // gate or up half: 768 rows of 6 blocks
constexpr size_t HALF_DN = (size_t) LAT * (FFE_C / 256) * 98;   // down half: 1536 rows of 3 blocks
constexpr size_t HALF_EXP = 2 * HALF_GU + HALF_DN;             // one card's half of an expert

struct CLayer {   // one card's part of a layer
    bool swa = false, sparse_idx = false, engram = false, dense = false;
    int H = 0, h0 = 0;                                 // the layer's heads and the card's first
    float rope_base = 0, clamp = 10.0f;
    const float *attn_norm, *q_a_norm, *kv_a_norm, *sinks, *post_attn_norm, *ffn_norm, *post_ffw_norm;
    const float *hc_attn_fn, *hc_attn_scale, *hc_attn_base, *hc_ffn_fn, *hc_ffn_scale, *hc_ffn_base;
    Mat q_a, q_b, kv_a, k_b, v_b, gate, wo;
    Mat idx_k, idx_qb, idx_proj, idx_cv, idx_cg;
    const float *idx_k_norm = nullptr, *idx_ape = nullptr, *idx_cnorm = nullptr;
    Mat e_kv;
    const float *e_kv_b = nullptr, *e_gate = nullptr, *e_lnorm = nullptr, *e_gamma = nullptr, *e_hcs = nullptr;
    int e_index = -1;
    Mat ffn_gate, ffn_up, ffn_down;
    Mat gate_inp, r_down, r_up, sh_gate, sh_up, sh_down;
    const float* probs_b = nullptr;
    __half* kv = nullptr;   // the latent cache, f16
    float* ivg = nullptr;
    float* pooled = nullptr;
};

struct Card {
    int dev = 0;
    cudaStream_t st, cst;
    cudaEvent_t cev;
    std::vector<CLayer> L;
    Mat out;   // vocab rows [c * NV_C, (c + 1) * NV_C)
    const float *out_norm, *head_fn, *head_scale, *head_base;
    size_t vram = 0;
    std::vector<void*> allocs;
    uint8_t* pool = nullptr;   // slots of expert halves
    // scratch
    float *R, *x, *xn, *post, *qr, *q, *qa, *qabs, *kvr, *o, *av, *gt, *ao, *lat, *gu_g, *gu_u, *hh, *ey, *moe, *sh_g, *sh_u, *fo,
        *fo2, *logits, *iq, *ik, *iv, *ig, *iw, *erows, *ekv, *dbig_g, *dbig_u, *head_x, *bestv;
    int *pos, *row, *topi, *sel, *nsel, *groups, *valid, *best;
    float* topw;
    EJob* jobs;
    // the window path's host side: page-locked router picks ([0]: this layer's, [MAXT * TOPK]: the next layer's guess) and job table
    int* h_ids = nullptr;
    EJob* h_jobs = nullptr;
    cudaEvent_t ev_route, ev_guess, ev_pref;
    int pref_layer = -1;               // the layer whose guessed experts the copy stream is bringing in (ev_pref)
    float *logits2 = nullptr, *topw2 = nullptr;
    int* topi2 = nullptr;
    float *opart = nullptr, *mlpart = nullptr;   // k_mla_part2's ranges (hy4_dec.cuh)
    float* iacc = nullptr;                       // v1v: the indexer's scores of a window [MAXT][ctx / KPOOL + 1]
    PJob *pjobs = nullptr, *h_pjobs = nullptr;   // v1x: a window's distinct experts (k_iq3_rows_mt), on the card and page-locked
    int2 *plists = nullptr, *h_plists = nullptr;
};

// the sum across the cards: page-locked mapped buffers, two of each card's (exchange parity), and a flag each
struct Xch {
    float* buf[NC][2];
    volatile unsigned* flag[NC];
    unsigned seq = 1;   // the flags start at 0: the first exchange waits for 1
};
__global__ void k_xput(const float* y, int n, float* out, volatile unsigned* flag, unsigned seq) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = y[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *flag = seq;
}
// v1s: k_add (the FFN's routed + shared output) and k_xput in one: y += y2, then y to the other card
__global__ void k_xput_add(float* y, const float* y2, int n, float* out, volatile unsigned* flag, unsigned seq) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = y[i] + y2[i];
        y[i] = v;
        out[i] = v;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *flag = seq;
}
__global__ void k_xget(float* y, int n, const volatile float* in, const volatile unsigned* flag, unsigned seq) {
    if (threadIdx.x == 0)
        while (*flag < seq) {}
    __syncthreads();
    __threadfence_system();
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = y[i] + in[i];
}
// v1r: k_xget, the rms that follows it (y = sum / rms(sum) * w) and the iHC write R_c += post_c * y in one block: two launches less
// an exchange. Both cards hold the same sum (a + b == b + a), so both get the same bits
__global__ void __launch_bounds__(1024) k_xget_rpost(float* y, int T, const volatile float* in, const volatile unsigned* flag, unsigned seq,
                                                     const float* w, float eps, float* R, const float* post) {
    __shared__ float red[32];
    if (threadIdx.x == 0)
        while (*flag < seq) {}
    __syncthreads();
    __threadfence_system();
    for (int t = 0; t < T; ++t) {
        float v[N / 1024], ss = 0;
#pragma unroll
        for (int k = 0; k < N / 1024; ++k) {
            const int i = threadIdx.x + 1024 * k;
            const float a = y[(size_t) t * N + i] + in[(size_t) t * N + i];
            v[k] = a;
            ss += a * a;
        }
        ss = block_sum(ss, red);
        const float kk = rsqrtf(ss / N + eps);
#pragma unroll
        for (int k = 0; k < N / 1024; ++k) {
            const int i = threadIdx.x + 1024 * k;
            const float o = v[k] * kk * w[i];
            y[(size_t) t * N + i] = o;
#pragma unroll
            for (int c = 0; c < HC; ++c) R[(size_t) t * D + c * N + i] += post[t * HC + c] * o;
        }
    }
}

struct Model {
    GgufMin g;
    Card cards[NC];
    const GTensor* tok = nullptr;
    const GTensor* etab = nullptr;
    float eps, hc_eps, hc_mag, wscale;
    std::vector<int64_t> tmap;
    int64_t pad = 0;
    std::vector<uint64_t> mult, offs, voc;
    std::vector<std::string> vocab;
    std::vector<const uint8_t*> exp_g, exp_u, exp_d;   // host mapping, per layer
    int ctx;

    void* alloc(Card& c, size_t bytes) {
        void* p = nullptr;
        CK(cudaSetDevice(c.dev));
        CK(cudaMalloc(&p, bytes));
        c.allocs.push_back(p);
        c.vram += bytes;
        return p;
    }
    const float* f32(Card& c, const std::string& name, size_t off = 0, size_t n = 0) {   // F32, or a slice of its values
        const GTensor& t = g.t(name);
        if (t.type != 0) throw std::runtime_error(name + " is not F32");
        const size_t bytes = n ? n * 4 : t.nbytes;
        void* p = alloc(c, bytes);
        CK(cudaMemcpy(p, t.data + off * 4, bytes, cudaMemcpyHostToDevice));
        return (const float*) p;
    }
    float* dev(Card& c, size_t floats) {
        float* p = (float*) alloc(c, floats * 4);
        CK(cudaMemset(p, 0, floats * 4));
        return p;
    }
    // rows [r0, r1) and columns [c0, c1) of a 2D view (rows = ne1 * ne2 * ne3, cols = ne0): Q8_0 into the split layout, F32/F16 as they are
    Mat mat(Card& c, const std::string& name, int r0 = 0, int r1 = -1, int c0 = 0, int c1 = -1) {
        const GTensor& t = g.t(name);
        const int rows = (int) (t.ne[1] * t.ne[2] * t.ne[3]), cols = (int) t.ne[0];
        if (r1 < 0) r1 = rows;
        if (c1 < 0) c1 = cols;
        Mat m;
        m.type = t.type;
        m.rows = r1 - r0;
        m.cols = c1 - c0;
        const size_t rb = t.row_bytes();
        if (t.type == 8) {
            if (c0 % 32 || c1 % 32) throw std::runtime_error(name + ": column slice off the 32-value blocks");
            const int nb = m.cols / 32;
            std::vector<int8_t> qv((size_t) m.rows * m.cols);
            std::vector<uint16_t> dv((size_t) m.rows * nb);
            for (int r = 0; r < m.rows; ++r) {
                const uint8_t* src = t.data + (size_t) (r0 + r) * rb + (size_t) (c0 / 32) * 34;
                for (int b = 0; b < nb; ++b) {
                    std::memcpy(&dv[(size_t) r * nb + b], src + b * 34, 2);
                    std::memcpy(&qv[(size_t) r * m.cols + b * 32], src + b * 34 + 2, 32);
                }
            }
            void* pq = alloc(c, qv.size());
            void* pd = alloc(c, dv.size() * 2);
            CK(cudaMemcpy(pq, qv.data(), qv.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(pd, dv.data(), dv.size() * 2, cudaMemcpyHostToDevice));
            m.q = (const int8_t*) pq;
            m.d = pd;
            return m;
        }
        if (t.type != 0 && t.type != 1) throw std::runtime_error(name + ": type " + std::to_string(t.type));
        const size_t es = t.type == 0 ? 4 : 2;
        std::vector<uint8_t> h((size_t) m.rows * m.cols * es);
        for (int r = 0; r < m.rows; ++r) std::memcpy(&h[(size_t) r * m.cols * es], t.data + (size_t) (r0 + r) * rb + c0 * es, (size_t) m.cols * es);
        void* p = alloc(c, h.size());
        CK(cudaMemcpy(p, h.data(), h.size(), cudaMemcpyHostToDevice));
        m.d = p;
        return m;
    }

    Model(const std::string& path, int ctx_) : g(path), ctx(ctx_) {
        const std::string a = "hy_v4_flash.";
        if (g.s("general.architecture") != "hy_v4_flash" || g.i(a + "block_count") != NL || g.i(a + "embedding_length") != N ||
            g.i(a + "expert_count") != NEXP || g.i(a + "expert_used_count") != TOPK || g.i(a + "expert_latent_length") != LAT)
            throw std::runtime_error("geometry differs from v1's constants");
        eps = (float) g.f(a + "attention.layer_norm_rms_epsilon");
        hc_eps = (float) g.f(a + "hyper_connection.epsilon");
        hc_mag = (float) g.f(a + "hyper_connection.magnitude");
        wscale = (float) g.f(a + "expert_weights_scale");
        const auto heads = g.ia(a + "attention.head_count");
        const auto swa = g.ia(a + "attention.sliding_window_pattern");
        const auto idx = g.ia(a + "attention.indexer.types");
        const auto clamp = g.fa(a + "swiglu_clamp_exp");
        const auto ple = g.ia(a + "ple.layers");
        const double base = g.f(a + "rope.freq_base"), base_swa = g.f(a + "rope.freq_base_swa");
        mult = g.ua(a + "ple.layer_multipliers");
        offs = g.ua(a + "ple.head_offsets");
        voc = g.ua(a + "ple.head_vocab_sizes");
        tmap = g.ia(a + "engram.token_map");
        pad = tmap.at((size_t) g.i(a + "engram.pad_token_id"));
        tok = &g.t("token_embd.weight");
        etab = &g.t("per_layer_token_embd.weight");
        vocab = g.sa("tokenizer.ggml.tokens");
        exp_g.assign(NL, nullptr);
        exp_u.assign(NL, nullptr);
        exp_d.assign(NL, nullptr);
        for (int c = 0; c < NC; ++c) {
            Card& K = cards[c];
            K.dev = c;
            K.L.resize(NL);
            for (int il = 0; il < NL; ++il) {
                CLayer& l = K.L[il];
                auto p = [&](const std::string& s) { return "blk." + std::to_string(il) + "." + s; };
                l.H = (int) heads[il] / NC;
                l.h0 = c * l.H;
                l.swa = swa[il] != 0;
                l.sparse_idx = idx[il] != 0;
                l.dense = il == 0;
                l.rope_base = (float) (l.swa ? base_swa : base);
                const int qh0 = l.h0 * HDQ, qh1 = (l.h0 + l.H) * HDQ;
                l.attn_norm = f32(K, p("attn_norm.weight"));
                l.q_a = mat(K, p("attn_q_a.weight"));
                l.q_a_norm = f32(K, p("attn_q_a_norm.weight"));
                l.q_b = mat(K, p("attn_q_b.weight"), qh0, qh1);
                l.kv_a = mat(K, p("attn_kv_a_mqa.weight"));
                l.kv_a_norm = f32(K, p("attn_kv_a_norm.weight"));
                l.k_b = mat(K, p("attn_k_b.weight"), l.h0 * KVL, (l.h0 + l.H) * KVL);
                l.v_b = mat(K, p("attn_v_b.weight"), l.h0 * HDV, (l.h0 + l.H) * HDV);
                l.gate = mat(K, p("attn_gate.weight"), l.h0 * HDV, (l.h0 + l.H) * HDV);
                l.wo = mat(K, p("attn_output.weight"), 0, -1, l.h0 * HDV, (l.h0 + l.H) * HDV);
                l.sinks = f32(K, p("attn_sinks.weight"), l.h0, l.H);
                l.post_attn_norm = f32(K, p("post_attention_norm.weight"));
                l.ffn_norm = f32(K, p("ffn_norm.weight"));
                l.post_ffw_norm = f32(K, p("post_ffw_norm.weight"));
                l.hc_attn_fn = f32(K, p("hc_attn_fn.weight"));
                l.hc_attn_scale = f32(K, p("hc_attn_scale.weight"));
                l.hc_attn_base = f32(K, p("hc_attn_base.weight"));
                l.hc_ffn_fn = f32(K, p("hc_ffn_fn.weight"));
                l.hc_ffn_scale = f32(K, p("hc_ffn_scale.weight"));
                l.hc_ffn_base = f32(K, p("hc_ffn_base.weight"));
                if (l.sparse_idx) {
                    l.idx_k = mat(K, p("indexer.attn_k.weight"));
                    l.idx_k_norm = f32(K, p("indexer.k_norm.weight"));
                    l.idx_qb = mat(K, p("indexer.attn_q_b.weight"));
                    l.idx_proj = mat(K, p("indexer.proj.weight"));
                    l.idx_cv = mat(K, p("indexer_compressor_value.weight"));
                    l.idx_cg = mat(K, p("indexer_compressor_gate.weight"));
                    l.idx_ape = f32(K, p("indexer_compressor_ape.weight"));
                    l.idx_cnorm = f32(K, p("indexer_compressor_norm.weight"));
                    l.ivg = dev(K, (size_t) IVR * 2 * IDXD);   // v1v: a ring (it was ctx rows: 384 MB a card at 32k)
                    l.pooled = dev(K, (size_t) (ctx / KPOOL + 1) * IDXD);
                }
                for (size_t e = 0; e < ple.size(); ++e)
                    if (ple[e] == il) l.e_index = (int) e;
                l.engram = l.e_index >= 0;
                if (l.engram) {
                    l.e_kv = mat(K, p("engram_kv.weight"));
                    l.e_kv_b = f32(K, p("engram_kv_b.weight"));
                    l.e_gate = f32(K, p("engram_gate.weight"));
                    l.e_lnorm = f32(K, p("engram_lookup_norm.weight"));
                    l.e_gamma = f32(K, p("engram_pre_gamma.weight"));
                    l.e_hcs = f32(K, p("engram_hc_scale.weight"));
                }
                if (l.dense) {
                    l.ffn_gate = mat(K, p("ffn_gate.weight"), c * FFD_C, (c + 1) * FFD_C);
                    l.ffn_up = mat(K, p("ffn_up.weight"), c * FFD_C, (c + 1) * FFD_C);
                    l.ffn_down = mat(K, p("ffn_down.weight"), 0, -1, c * FFD_C, (c + 1) * FFD_C);
                } else {
                    l.gate_inp = mat(K, p("ffn_gate_inp.weight"));
                    l.probs_b = f32(K, p("exp_probs_b.bias"));
                    l.r_down = mat(K, p("ffn_routed_down.weight"));
                    l.r_up = mat(K, p("ffn_routed_up.weight"));
                    l.sh_gate = mat(K, p("ffn_gate_shexp.weight"), c * FFE_C, (c + 1) * FFE_C);
                    l.sh_up = mat(K, p("ffn_up_shexp.weight"), c * FFE_C, (c + 1) * FFE_C);
                    l.sh_down = mat(K, p("ffn_down_shexp.weight"), 0, -1, c * FFE_C, (c + 1) * FFE_C);
                    l.clamp = (float) clamp[il];
                    if (c == 0) {
                        const GTensor &tg = g.t(p("ffn_gate_exps.weight")), &tu = g.t(p("ffn_up_exps.weight")), &td = g.t(p("ffn_down_exps.weight"));
                        if (tg.type != G_IQ3_XXS || tu.type != G_IQ3_XXS || td.type != G_IQ3_XXS || tg.nbytes != NEXP * EXP_BYTES)
                            throw std::runtime_error("routed experts: v1 reads IQ3_XXS 1536 x 1536 only");
                        exp_g[il] = tg.data;
                        exp_u[il] = tu.data;
                        exp_d[il] = td.data;
                    }
                }
                l.kv = (__half*) dev(K, ((size_t) (l.swa ? RING : ctx) * KROW + 1) / 2);
            }
            K.out = mat(K, "output.weight", c * NV_C, (c + 1) * NV_C);
            K.out_norm = f32(K, "output_norm.weight");
            K.head_fn = f32(K, "output_hc_fn.weight");
            K.head_scale = f32(K, "output_hc_scale.weight");
            K.head_base = f32(K, "output_hc_base.weight");
            std::fprintf(stderr, "hy4_v1: card %d loaded, %.2f GiB\n", c, K.vram / 1073741824.0);
        }
    }
};

struct Prof {   // HY4_PROF=1: card 0's thread, waits for card 0 at each mark
    bool on = getenv("HY4_PROF") != nullptr;
    std::map<std::string, double> ms;
    std::chrono::steady_clock::time_point t;
    void start() { if (on) t = std::chrono::steady_clock::now(); }
    void stop(const std::string& k, cudaStream_t st) {
        if (!on) return;
        CK(cudaStreamSynchronize(st));
        const auto n = std::chrono::steady_clock::now();
        ms[k] += std::chrono::duration<double, std::milli>(n - t).count();
        t = n;
    }
    void print(int tokens) {
        if (!on) return;
        double sum = 0;
        for (auto& kv : ms) sum += kv.second;
        std::printf("profile (%d tokens, ms a token):", tokens);
        for (auto& kv : ms) std::printf(" %s %.1f", kv.first.c_str(), kv.second / tokens);
        std::printf(" | total %.1f\n", sum / tokens);
        ms.clear();
    }
};

// One card's copy of the host LRU over the slot numbers. Both cards see the same router picks in the same order, so both
// copies take the same decisions without talking to each other (each card's host thread runs its own)
struct Lru {
    std::unordered_map<int, int> slot_of;
    std::vector<int> owner;
    std::list<int> lru;
    std::vector<std::list<int>::iterator> lru_it;
    size_t hits = 0, misses = 0;
    void init(int slots) {
        owner.assign(slots, -1);
        lru_it.resize(slots);
        for (int s = 0; s < slots; ++s) lru_it[s] = lru.insert(lru.end(), s);
    }
};

struct Engine {
    Model& M;
    Prof prof;
    Xch X;
    unsigned seq[NC];   // each card's next exchange number (both run the same sequence)
    int slots;
    Lru lru[NC];
    std::vector<int> hist;
    int top = -1;   // the highest position written since the cache was last read from 0 (window rows live in a ring of RING)
    uint8_t* hexp[NC] = {};   // page-locked: [card][MoE layer][expert] halves
    std::string dump;
    // the window being run, set by window() before the card threads start
    const int* w_tok = nullptr;
    std::vector<int> w_pos;
    std::vector<float> w_emb;
    int w_T = 0;
    int out_i[NC][MAXT];
    float out_v[NC][MAXT];
    std::vector<float> w_logits;   // the last row's, both halves (dumps)
    size_t drafted = 0, accepted = 0;   // lookup drafts (generate)

    const uint8_t* host_half(int c, int il, int e) const { return hexp[c] + ((size_t) (il - 1) * NEXP + e) * HALF_EXP; }
    size_t hits() const { return lru[0].hits; }
    size_t misses() const { return lru[0].misses; }

    Engine(Model& m, int slots_) : M(m), slots(slots_) {
        for (int c = 0; c < NC; ++c) {
            lru[c].init(slots);
            seq[c] = X.seq;
        }
        const int HM = 32;   // the most heads a card holds
        bind_node(0);
        for (int c = 0; c < NC; ++c) on_node(c, [&]() {
            Card& K = M.cards[c];
            CK(cudaSetDevice(c));
            CK(cudaStreamCreateWithFlags(&K.st, cudaStreamNonBlocking));
            CK(cudaStreamCreateWithFlags(&K.cst, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&K.cev, cudaEventDisableTiming));
            K.pool = (uint8_t*) M.alloc(K, (size_t) slots * HALF_EXP);
            auto f = [&](size_t n) { return M.dev(K, n); };
            auto i32 = [&](size_t n) { return (int*) M.dev(K, n); };
            K.R = f(MAXT * D); K.x = f(MAXT * N); K.xn = f(MAXT * N); K.post = f(MAXT * HC); K.qr = f(MAXT * QLR);
            K.q = f((size_t) MAXT * HM * HDQ); K.qa = f((size_t) MAXT * HM * KROW); K.qabs = f((size_t) MAXT * HM * KVL);
            K.kvr = f(MAXT * KROW); K.o = f((size_t) MAXT * HM * KVL); K.av = f((size_t) MAXT * HM * HDV); K.gt = f((size_t) MAXT * HM * HDV);
            K.ao = f(MAXT * N); K.lat = f(MAXT * LAT); K.gu_g = f((size_t) MAXT * TOPK * FFE_C); K.gu_u = f((size_t) MAXT * TOPK * FFE_C);
            K.hh = f((size_t) MAXT * TOPK * FFE_C); K.ey = f((size_t) MAXT * TOPK * LAT); K.moe = f(MAXT * LAT);
            K.sh_g = f(MAXT * FFE_C); K.sh_u = f(MAXT * FFE_C); K.fo = f(MAXT * N); K.fo2 = f(MAXT * N); K.logits = f(MAXT * NEXP);
            K.iq = f(MAXT * IDXH * IDXD); K.ik = f(MAXT * IDXD); K.iv = f(MAXT * IDXD); K.ig = f(MAXT * IDXD); K.iw = f(MAXT * IDXH);
            K.erows = f(MAXT * E_W); K.ekv = f((size_t) MAXT * (HC + 1) * N); K.dbig_g = f((size_t) MAXT * FFD_C);
            K.dbig_u = f((size_t) MAXT * FFD_C); K.head_x = f((size_t) MAXT * NV_C); K.bestv = f(MAXT);
            K.pos = i32(MAXT); K.row = i32(MAXT); K.topi = i32(MAXT * TOPK); K.sel = i32((size_t) MAXT * MAXSEL); K.nsel = i32(MAXT);
            K.groups = i32(MAXT); K.valid = i32(MAXT); K.best = i32(MAXT); K.topw = f(MAXT * TOPK);
            K.jobs = (EJob*) M.alloc(K, sizeof(EJob) * MAXT * TOPK * 3);
            K.pjobs = (PJob*) M.alloc(K, sizeof(PJob) * MAXT * TOPK * 3);
            K.plists = (int2*) M.alloc(K, sizeof(int2) * MAXT * TOPK);
            CK(cudaHostAlloc((void**) &K.h_pjobs, sizeof(PJob) * MAXT * TOPK * 3, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &K.h_plists, sizeof(int2) * MAXT * TOPK, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &K.h_ids, sizeof(int) * 2 * MAXT * TOPK, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &K.h_jobs, sizeof(EJob) * MAXT * TOPK * 3, cudaHostAllocPortable));
            CK(cudaEventCreateWithFlags(&K.ev_route, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&K.ev_guess, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&K.ev_pref, cudaEventDisableTiming));
            K.logits2 = f(MAXT * NEXP); K.topw2 = f(MAXT * TOPK); K.topi2 = i32(MAXT * TOPK);
            K.opart = f((size_t) MLA_PART * KVL); K.mlpart = f((size_t) MLA_PART * 2);
            K.iacc = f((size_t) MAXT * (M.ctx / KPOOL + 1));
            std::fprintf(stderr, "hy4_v1: card %d, %d expert-half slots (%.2f GiB), VRAM in all %.2f GiB\n", c, slots,
                         slots * (double) HALF_EXP / 1073741824.0, K.vram / 1073741824.0);
        });
        for (int c = 0; c < NC; ++c) on_node(1 - c, [&]() {   // card c writes its part, the other card waits on it and reads it
            for (int p = 0; p < 2; ++p) CK(cudaHostAlloc((void**) &X.buf[c][p], MAXT * N * 4, cudaHostAllocMapped | cudaHostAllocPortable));
            unsigned* f = nullptr;
            CK(cudaHostAlloc((void**) &f, 64, cudaHostAllocMapped | cudaHostAllocPortable));
            std::memset(f, 0, 64);
            X.flag[c] = f;
        });
        pin_experts();
        std::thread([this]() { w1_loop(); }).detach();
    }

    // card 1's host thread for good, on node 1 (v1 started a thread a window: ~50 us on this 1.7 GHz CPU); card 0's work runs on
    // the calling thread. w1_gen counts the jobs given, w1_fin the jobs done; kind 1: run_card(1), 2: run_card_pf(1)
    std::atomic<unsigned> w1_gen{0}, w1_fin{0};
    int w1_kind = 0;
    void w1_loop() {
        bind_node(1);
        unsigned seen = 0;
        for (;;) {
            unsigned g;
            for (long spin = 0; (g = w1_gen.load(std::memory_order_acquire)) == seen; ++spin)
                if (spin > 200000) std::this_thread::sleep_for(std::chrono::microseconds(50));
                else __builtin_ia32_pause();
            seen = g;
            if (w1_kind == 1) run_card(1);
            else run_card_pf(1);
            w1_fin.store(g, std::memory_order_release);
        }
    }
    void card1(int kind) {
        w1_kind = kind;
        w1_gen.fetch_add(1, std::memory_order_acq_rel);
    }
    void card1_wait() {
        const unsigned g = w1_gen.load(std::memory_order_acquire);
        while (w1_fin.load(std::memory_order_acquire) != g) __builtin_ia32_pause();
    }

    void pin_experts() {
        const size_t per = (size_t) (NL - 1) * NEXP * HALF_EXP;
        auto t0 = std::chrono::steady_clock::now();
        for (int c = 0; c < NC; ++c) on_node(c, [&]() { CK(cudaHostAlloc((void**) &hexp[c], per, cudaHostAllocPortable)); });   // card c's node
        std::vector<std::thread> th;
        for (int w = 0; w < 8; ++w)
            th.emplace_back([&, w]() {
                const int c = w & 1;   // 4 writers a card, on its node
                bind_node(c);
                for (int il = 1 + (w >> 1); il < NL; il += 4)
                    for (int e = 0; e < NEXP; ++e) {
                            uint8_t* d = hexp[c] + ((size_t) (il - 1) * NEXP + e) * HALF_EXP;
                            const uint8_t* g = M.exp_g[il] + (size_t) e * EXP_BYTES;
                            const uint8_t* u = M.exp_u[il] + (size_t) e * EXP_BYTES;
                            const uint8_t* dn = M.exp_d[il] + (size_t) e * EXP_BYTES;
                            std::memcpy(d, g + (size_t) c * HALF_GU, HALF_GU);                 // gate rows [768c, 768c + 768)
                            std::memcpy(d + HALF_GU, u + (size_t) c * HALF_GU, HALF_GU);
                            for (int r = 0; r < LAT; ++r)                                       // down: blocks [3c, 3c + 3) of each row
                                std::memcpy(d + 2 * HALF_GU + (size_t) r * 3 * 98, dn + (size_t) r * 6 * 98 + (size_t) c * 3 * 98, 3 * 98);
                        }
            });
        for (auto& t : th) t.join();
        std::fprintf(stderr, "hy4_v1: expert halves in page-locked RAM, %.2f GiB in %.1f s\n", 2 * per / 1073741824.0,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    // card c's slots for a layer's picks, copying its halves of the missing experts on its copy stream (then its event)
    void experts(int c, int il, const std::vector<int>& ids, std::vector<int>& slot, std::vector<char>& miss) {
        Lru& U = lru[c];
        Card& K = M.cards[c];
        std::map<int, int> got;
        std::map<int, char> copied;
        for (int e : ids) {
            const int key = il * NEXP + e;
            if (got.count(key)) continue;
            auto it = U.slot_of.find(key);
            int s;
            if (it != U.slot_of.end()) {
                s = it->second;
                ++U.hits;
                copied[key] = 0;
            } else {
                s = -1;
                for (int cnd : U.lru)
                    if (U.owner[cnd] < 0 || !got.count(U.owner[cnd])) { s = cnd; break; }
                if (U.owner[s] >= 0) U.slot_of.erase(U.owner[s]);
                U.owner[s] = key;
                U.slot_of[key] = s;
                CK(cudaMemcpyAsync(K.pool + (size_t) s * HALF_EXP, host_half(c, il, e), HALF_EXP, cudaMemcpyHostToDevice, K.cst));
                ++U.misses;
                copied[key] = 1;
            }
            got[key] = s;
            U.lru.erase(U.lru_it[s]);
            U.lru_it[s] = U.lru.insert(U.lru.end(), s);
        }
        CK(cudaEventRecord(K.cev, K.cst));
        slot.clear();
        miss.clear();
        for (int e : ids) {
            slot.push_back(got[il * NEXP + e]);
            miss.push_back(copied[il * NEXP + e]);
        }
    }

    FILE* trace = getenv("HY4_TRACE") ? std::fopen(getenv("HY4_TRACE"), "ab") : nullptr;   // v1j: routing picks of every window
    bool dec_on = getenv("HY4_DEC_OFF") == nullptr;
    const bool guess_stat = getenv("HY4_GUESS_STAT") != nullptr;   // v1g's writing kernels (hy4_dec.cuh); HY4_DEC_OFF=1: v1's
    void iq3(const EJob* jobs, const float* x, int ldx, float* y, int nj, int ldy, int rows, int cols, cudaStream_t st) {
        if (dec_on) k_iq3_rows5<4><<<dim3(rows / 32, nj), 256, 0, st>>>(jobs, x, ldx, y, ldy, rows, cols);
        else k_iq3_rows<<<dim3(rows / IQ_BR, nj), 256, 0, st>>>(jobs, x, ldx, y, ldy, rows, cols);
    }

    int prefetch_mode = -1;   // a request's "prefetch" (serve); -1: HY4_PREFETCH (unset: off)
    int prefetch_kind() const {   // 0 off, 1 v1e's (layer il + 1 from layer il's FFN input), 2 v1q's (layer il from its attention input)
        if (prefetch_mode >= 0) return prefetch_mode;
        const char* v = getenv("HY4_PREFETCH");
        return v ? atoi(v) : 0;
    }
    bool prefetch_on() const { return prefetch_kind() == 1; }
    size_t pref_copies = 0;
    // the next layer's guessed experts into the cache (the LRU's coldest slots; the current layer's are at its other end)
    void prefetch(int c, int il, const std::vector<int>& ids, int most = NEXP) {   // at most `most` copies, in the ids' order
        Lru& U = lru[c];
        Card& K = M.cards[c];
        bool any = false;
        int n = 0;
        for (int e : ids) {
            if (n >= most) break;
            const int key = il * NEXP + e;
            if (U.slot_of.count(key)) continue;
            const int s = U.lru.front();
            if (U.owner[s] >= 0) U.slot_of.erase(U.owner[s]);
            U.owner[s] = key;
            U.slot_of[key] = s;
            U.lru.erase(U.lru_it[s]);
            U.lru_it[s] = U.lru.insert(U.lru.end(), s);
            CK(cudaMemcpyAsync(K.pool + (size_t) s * HALF_EXP, host_half(c, il, e), HALF_EXP, cudaMemcpyHostToDevice, K.cst));
            ++pref_copies;
            ++n;
            any = true;
        }
        if (any) {
            CK(cudaEventRecord(K.ev_pref, K.cst));
            K.pref_layer = il;
        }
    }

    void engram_rows(int e, const int* tok, const int* p, int T, std::vector<float>& out, std::vector<int>& ok) {
        out.assign((size_t) T * E_W, 0.0f);
        ok.assign(T, 1);
        engram_rows(e, tok, p, T, out.data(), ok.data());
    }
    void engram_rows(int e, const int* tok, const int* p, int T, float* out, int* ok) {   // writes every value of out and ok
        const uint64_t* mu = M.mult.data() + (size_t) e * E_NGRAM;
        const int cols = (E_NGRAM - 1) * E_HPN;
        const uint64_t* pr = M.voc.data() + (size_t) e * cols;
        const uint64_t* of = M.offs.data() + (size_t) e * cols;
        auto cid = [&](int t) -> int64_t { return t < 0 || t >= (int) M.tmap.size() ? -1 : M.tmap[(size_t) t]; };
        const size_t rb = M.etab->row_bytes();
        for (int i = 0; i < T; ++i) {
            uint64_t toks[E_NGRAM];
            const int64_t c0 = cid(tok[i]);
            ok[i] = c0 >= 0;
            toks[0] = (uint64_t) (c0 < 0 ? M.pad : c0);
            for (int s = 1; s < E_NGRAM; ++s) {
                const int at = p[i] - s;
                const int64_t c = at < 0 ? -1 : cid(hist[(size_t) at]);
                toks[s] = (uint64_t) (c < 0 ? M.pad : c);
            }
            uint64_t rolling = toks[0] * mu[0];
            for (int o = 1; o < E_NGRAM; ++o) {
                rolling ^= toks[o] * mu[o];
                for (int h = 0; h < E_HPN; ++h) {
                    const int col = (o - 1) * E_HPN + h;
                    const int64_t rid = (int64_t) (rolling % pr[col] + of[col]);
                    const uint8_t* blk = M.etab->data + (size_t) rid * rb;
                    float* y = out + ((size_t) i * E_COLS + col) * E_DIM;
                    for (int b = 0; b < E_DIM / 32; ++b) {
                        __half hh;
                        std::memcpy(&hh, blk + b * 18, 2);
                        const float d = __half2float(hh);
                        const uint8_t* qs = blk + b * 18 + 2;
                        for (int j = 0; j < 16; ++j) {
                            y[b * 32 + j] = d * h_kvalues_iq4nl[qs[j] & 15];
                            y[b * 32 + j + 16] = d * h_kvalues_iq4nl[qs[j] >> 4];
                        }
                    }
                }
            }
        }
    }

    void save(const std::string& name, const float* d, size_t n) {   // card 0's thread
        if (dump.empty() || !n) return;
        std::vector<float> h(n);
        CK(cudaStreamSynchronize(M.cards[0].st));
        CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost));
        FILE* f = std::fopen((dump + "/" + name + ".v0").c_str(), "ab");
        std::fwrite(h.data(), 4, n, f);
        std::fclose(f);
    }

    // the sum of y [T][N] across the cards, in place (card c's part of exchange number seq[c])
    void xsum(int c, float* y, int T) {
        Card& K = M.cards[c];
        const unsigned s = seq[c]++;
        const int par = s & 1;
        k_xput<<<1, 1024, 0, K.st>>>(y, T * N, X.buf[c][par], X.flag[c], s);
        k_xget<<<1, 1024, 0, K.st>>>(y, T * N, X.buf[1 - c][par], X.flag[1 - c], s);
    }
    // the sum, its rms with weight w and R += post * it (k_xget_rpost)
    void xsum_rpost(int c, float* y, int T, const float* w, float* R, const float* post, const float* add = nullptr) {   // add: y += add first
        Card& K = M.cards[c];
        const unsigned s = seq[c]++;
        const int par = s & 1;
        if (add) k_xput_add<<<1, 1024, 0, K.st>>>(y, add, T * N, X.buf[c][par], X.flag[c], s);
        else k_xput<<<1, 1024, 0, K.st>>>(y, T * N, X.buf[c][par], X.flag[c], s);
        k_xget_rpost<<<1, 1024, 0, K.st>>>(y, T, X.buf[1 - c][par], X.flag[1 - c], s, w, M.eps, R, post);
    }

    std::vector<int> window(const int* tok, int p0, int T) {
        w_tok = tok;
        w_T = T;
        w_pos.resize(T);
        for (int t = 0; t < T; ++t) {
            w_pos[t] = p0 + t;
            if ((int) hist.size() <= p0 + t) hist.resize((size_t) p0 + t + 1);
            hist[(size_t) p0 + t] = tok[t];
        }
        w_emb.assign((size_t) T * D, 0.0f);
        const size_t rb = M.tok->row_bytes();
        for (int t = 0; t < T; ++t) {
            const uint8_t* r = M.tok->data + (size_t) tok[t] * rb;
            for (int b = 0; b < N / 32; ++b) {
                __half hh;
                std::memcpy(&hh, r + b * 34, 2);
                const float d = __half2float(hh);
                for (int j = 0; j < 32; ++j) {
                    const float v = d * (float) (int8_t) r[b * 34 + 2 + j];
                    for (int c = 0; c < HC; ++c) w_emb[(size_t) t * D + c * N + b * 32 + j] = v;
                }
            }
        }
        if (!dump.empty()) w_logits.assign(NV, 0.0f);
        card1(1);
        run_card(0);
        card1_wait();
        if (!dump.empty()) {
            FILE* f = std::fopen((dump + "/logiti.v0").c_str(), "ab");
            std::fwrite(w_logits.data(), 4, NV, f);
            std::fclose(f);
        }
        std::vector<int> out(T);
        for (int t = 0; t < T; ++t) out[t] = out_v[1][t] > out_v[0][t] ? out_i[1][t] + NV_C : out_i[0][t];
        return out;
    }

    // one card's whole window, on its own host thread
    void run_card(int c) {
        Card& K = M.cards[c];
        const int T = w_T;
        CK(cudaSetDevice(c));
        CK(cudaMemcpyAsync(K.pos, w_pos.data(), T * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(K.R, w_emb.data(), w_emb.size() * 4, cudaMemcpyHostToDevice, K.st));
        for (int il = 0; il < NL; ++il) layer(c, il);
        if (c == 0) prof.start();
        if (dec_on) hc_pre_rms(K.R, K.head_fn, K.head_scale, K.head_base, HC, M.eps, M.hc_eps, 0.0f, K.x, nullptr, K.out_norm, K.xn, M.eps, T, K.st);
        else {
            hc_pre(K.R, K.head_fn, K.head_scale, K.head_base, HC, M.eps, M.hc_eps, 0.0f, K.x, nullptr, T, K.st);
            rms(K.x, N, K.out_norm, K.xn, N, T, N, M.eps, K.st);
        }
        for (int t = 0; t < T; ++t) {
            gemv(K.out, K.xn + (size_t) t * N, N, K.head_x, NV_C, 1, K.st);
            k_argmax<<<1, 1024, 0, K.st>>>(K.head_x, NV_C, K.best + t, K.bestv + t);
        }
        if (!dump.empty()) {
            CK(cudaStreamSynchronize(K.st));
            CK(cudaMemcpy(w_logits.data() + (size_t) c * NV_C, K.head_x, NV_C * 4, cudaMemcpyDeviceToHost));
        }
        CK(cudaMemcpyAsync(out_i[c], K.best, T * 4, cudaMemcpyDeviceToHost, K.st));
        CK(cudaMemcpyAsync(out_v[c], K.bestv, T * 4, cudaMemcpyDeviceToHost, K.st));
        CK(cudaStreamSynchronize(K.st));
        if (c == 0) prof.stop("glava", K.st);
    }

    void layer(int c, int il) {
        Card& K = M.cards[c];
        const CLayer& l = K.L[il];
        cudaStream_t st = K.st;
        const int T = w_T;
        const int* hp = w_pos.data();
        const bool p0 = c == 0;
        std::vector<int> hrow(T);
        for (int t = 0; t < T; ++t) hrow[t] = l.swa ? hp[t] % RING : hp[t];
        std::vector<int> done;
        for (int t = 0; t < T; ++t)
            if (hp[t] % KPOOL == KPOOL - 1) done.push_back(hp[t] / KPOOL);
        const bool use_sel = l.sparse_idx && hp[T - 1] / KPOOL > TOPPOOL;
        const int maxng = hp[T - 1] / KPOOL;
        if (p0) prof.start();
        if (l.engram) {
            std::vector<float> rows;
            std::vector<int> ok;
            engram_rows(l.e_index, w_tok, hp, T, rows, ok);
            CK(cudaMemcpyAsync(K.erows, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(K.valid, ok.data(), T * 4, cudaMemcpyHostToDevice, st));
            k_engram_rows<<<dim3(T, E_COLS), 128, 0, st>>>(K.erows, l.e_lnorm);
            gemv(l.e_kv, K.erows, E_W, K.ekv, (HC + 1) * N, T, st);
            k_engram_apply<<<T, 256, 0, st>>>(K.R, K.ekv, l.e_kv_b, l.e_gate, l.e_gamma, l.e_hcs, K.valid);
        }
        if (dec_on)
            hc_pre_rms(K.R, l.hc_attn_fn, l.hc_attn_scale, l.hc_attn_base, 2 * HC, M.eps, M.hc_eps, M.hc_mag, K.x, K.post, l.attn_norm, K.xn, M.eps, T, st);
        const bool guess2 = prefetch_kind() == 2 && T == 1 && !l.dense && dec_on;
        if ((guess_stat && c == 0 || guess2) && T == 1 && !l.dense && dec_on) {   // the layer's router on its attention input
            gemv32_t1(l.gate_inp, K.xn, K.logits2, st);
            k_route4<<<1, 256, 0, st>>>(K.logits2, l.probs_b, M.wscale, K.topi2, K.topw2);
            if (guess2) {
                CK(cudaMemcpyAsync(K.h_ids + MAXT * TOPK, K.topi2, TOPK * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaEventRecord(K.ev_guess, st));
            }
        }
        else {
            hc_pre(K.R, l.hc_attn_fn, l.hc_attn_scale, l.hc_attn_base, 2 * HC, M.eps, M.hc_eps, M.hc_mag, K.x, K.post, T, st);
            rms(K.x, N, l.attn_norm, K.xn, N, T, N, M.eps, st);
        }
        const int H = l.H;
        const bool grp = dec_on && T == 1;   // v1r: the Q8S matrices on the attention input in one launch (gemv_group)
        if (grp) {
            if (l.sparse_idx) gemv_group({{&l.q_a, K.qr}, {&l.kv_a, K.kvr}, {&l.gate, K.gt}, {&l.idx_k, K.ik}}, K.xn, st);
            else gemv_group({{&l.q_a, K.qr}, {&l.kv_a, K.kvr}, {&l.gate, K.gt}}, K.xn, st);
        } else
            gemv(l.q_a, K.xn, N, K.qr, QLR, T, st);
        rms(K.qr, QLR, l.q_a_norm, K.qr, QLR, T, QLR, M.eps, st);
        gemv(l.q_b, K.qr, QLR, K.q, H * HDQ, T, st);
        k_rope<<<dim3(T, H), ROT / 2, 0, st>>>(K.q, H * HDQ, H, HDQ, NOPE, K.pos, l.rope_base);
        if (!grp) gemv(l.kv_a, K.xn, N, K.kvr, KROW, T, st);
        k_rope<<<dim3(T, 1), ROT / 2, 0, st>>>(K.kvr, KROW, 1, KROW, KVL, K.pos, l.rope_base);
        CK(cudaMemcpyAsync(K.row, hrow.data(), T * 4, cudaMemcpyHostToDevice, st));
        k_kv_write<<<T, 256, 0, st>>>(K.kvr, l.kv_a_norm, M.eps, l.kv, K.row);
        for (int t = 0; t < T; ++t) gemv(l.k_b, K.q + (size_t) t * H * HDQ, 0, K.qabs + (size_t) t * H * KVL, 0, 1, st, KVL, HDQ);
        k_q_assemble<<<dim3(T, H), 256, 0, st>>>(K.q, K.qabs, K.qa, H);
        if (l.sparse_idx) {
            if (!grp) gemv(l.idx_k, K.xn, N, K.ik, IDXD, T, st);
            rms(K.ik, IDXD, l.idx_k_norm, K.ik, IDXD, T, IDXD, M.eps, st);
            gemv(l.idx_cv, K.ik, IDXD, K.iv, IDXD, T, st);
            gemv(l.idx_cg, K.ik, IDXD, K.ig, IDXD, T, st);
            k_idx_store<<<T, IDXD, 0, st>>>(K.iv, K.ig, l.ivg, K.pos);
            if (!done.empty()) {
                CK(cudaMemcpyAsync(K.groups, done.data(), done.size() * 4, cudaMemcpyHostToDevice, st));
                k_idx_pool<<<(int) done.size(), IDXD, 0, st>>>(l.ivg, l.idx_ape, l.idx_cnorm, M.eps, l.rope_base, K.groups, l.pooled);
            }
            if (use_sel) {
                gemv(l.idx_qb, K.qr, QLR, K.iq, IDXH * IDXD, T, st);
                k_rope<<<dim3(T, IDXH), ROT / 2, 0, st>>>(K.iq, IDXH * IDXD, IDXH, IDXD, IDXD - ROT, K.pos, l.rope_base);
                gemv(l.idx_proj, K.xn, N, K.iw, IDXH, T, st);
                k_scale<<<1, 256, 0, st>>>(K.iw, (size_t) T * IDXH, 1.0f / sqrtf((float) (IDXD * IDXH)));
                if (dec_on) {   // v1v: any context, many blocks
                    k_idx_scores<<<dim3((maxng + ISG - 1) / ISG, T), 256, 0, st>>>(K.iq, K.iw, l.pooled, K.pos, maxng, K.iacc);
                    k_idx_topk2<<<T, 1024, 0, st>>>(K.iacc, maxng, K.pos, K.sel, K.nsel);
                } else
                    k_idx_select<<<T, 256, (size_t) maxng * 8, st>>>(K.iq, K.iw, l.pooled, K.pos, K.sel, K.nsel);
            }
        }
        if (dec_on) {
            const int pm = hp[T - 1], nmax = l.swa ? std::min(SWA, pm + 1) : use_sel ? MAXSEL : pm + 1;
            mla_dec2_t<16>(K.qa, l.kv, K.pos, l.swa ? 1 : 0, use_sel ? K.sel : nullptr, use_sel ? K.nsel : nullptr, l.sinks,
                           1.0f / sqrtf((float) HDQ), K.o, H, T, nmax, K.opart, K.mlpart, st);
        } else
            k_mla_attn<<<dim3(H, T), 256, (size_t) (KROW + MAXSEL) * 4, st>>>(K.qa, l.kv, K.pos, l.swa ? 1 : 0, use_sel ? K.sel : nullptr,
                                                                              use_sel ? K.nsel : nullptr, l.sinks, 1.0f / sqrtf((float) HDQ), K.o, H);
        for (int t = 0; t < T; ++t) gemv(l.v_b, K.o + (size_t) t * H * KVL, 0, K.av + (size_t) t * H * HDV, 0, 1, st, HDV, KVL);
        if (!grp) gemv(l.gate, K.xn, N, K.gt, H * HDV, T, st);
        k_sigmul<<<dim3(64, T), 256, 0, st>>>(K.av, K.gt, H * HDV);
        gemv(l.wo, K.av, H * HDV, K.ao, N, T, st);
        if (dec_on) xsum_rpost(c, K.ao, T, l.post_attn_norm, K.R, K.post);
        else {
            xsum(c, K.ao, T);
            rms(K.ao, N, l.post_attn_norm, K.ao, N, T, N, M.eps, st);
            k_hc_post<<<dim3(T, HCB), 256, 0, st>>>(K.R, K.ao, K.post);
        }
        if (dec_on)
            hc_pre_rms(K.R, l.hc_ffn_fn, l.hc_ffn_scale, l.hc_ffn_base, 2 * HC, M.eps, M.hc_eps, M.hc_mag, K.x, K.post, l.ffn_norm, K.xn, M.eps, T, st);
        else {
            hc_pre(K.R, l.hc_ffn_fn, l.hc_ffn_scale, l.hc_ffn_base, 2 * HC, M.eps, M.hc_eps, M.hc_mag, K.x, K.post, T, st);
            rms(K.x, N, l.ffn_norm, K.xn, N, T, N, M.eps, st);
        }
        if (p0) prof.stop(l.swa ? "pozornost_okno" : "pozornost_redka", st);
        if (l.dense) {
            gemv(l.ffn_gate, K.xn, N, K.dbig_g, FFD_C, T, st);
            gemv(l.ffn_up, K.xn, N, K.dbig_u, FFD_C, T, st);
            k_swiglu<<<dim3(64, T), 256, 0, st>>>(K.dbig_g, K.dbig_u, K.dbig_g, FFD_C, 0.0f);
            gemv(l.ffn_down, K.dbig_g, FFD_C, K.fo, N, T, st);
        } else {
            if (guess2) {   // HY4_PREFETCH=2: the guessed picks not cached yet (HY4_PREFETCH_N, default 6, best first) copy during the attention
                static const int pn = getenv("HY4_PREFETCH_N") ? atoi(getenv("HY4_PREFETCH_N")) : 6;
                CK(cudaEventSynchronize(K.ev_guess));
                prefetch(c, il, std::vector<int>(K.h_ids + MAXT * TOPK, K.h_ids + MAXT * TOPK + TOPK), pn);
            }
            if (dec_on && T == 1 && l.gate_inp.type == 0 && l.gate_inp.cols % 128 == 0) gemv32_t1(l.gate_inp, K.xn, K.logits, st);
            else gemv(l.gate_inp, K.xn, N, K.logits, NEXP, T, st);
            if (dec_on) k_route4<<<T, 256, 0, st>>>(K.logits, l.probs_b, M.wscale, K.topi, K.topw);
            else k_route<<<T, 32, 0, st>>>(K.logits, l.probs_b, M.wscale, K.topi, K.topw);
            if (guess_stat && c == 0 && T == 1 && dec_on) k_guess_overlap<<<1, 32, 0, st>>>(K.topi, K.topi2);
            CK(cudaMemcpyAsync(K.h_ids, K.topi, (size_t) T * TOPK * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaEventRecord(K.ev_route, st));
            // what needs no routing runs while the host decides: the latent input, the shared expert, the next layer's guess
            if (grp) gemv_group({{&l.r_down, K.lat}, {&l.sh_gate, K.sh_g}, {&l.sh_up, K.sh_u}}, K.xn, st);
            else {
                gemv(l.r_down, K.xn, N, K.lat, LAT, T, st);
                gemv(l.sh_gate, K.xn, N, K.sh_g, FFE_C, T, st);
                gemv(l.sh_up, K.xn, N, K.sh_u, FFE_C, T, st);
            }
            k_swiglu<<<dim3(4, T), 256, 0, st>>>(K.sh_g, K.sh_u, K.sh_g, FFE_C, 0.0f);
            gemv(l.sh_down, K.sh_g, FFE_C, K.fo2, N, T, st);
            const bool guess = prefetch_on() && il + 1 < NL;
            if (guess) {   // HY4_PREFETCH: layer il + 1's router on this layer's input; its picks are copied while layer il + 1's attention runs
                const CLayer& ln = K.L[il + 1];
                gemv(ln.gate_inp, K.xn, N, K.logits2, NEXP, T, st);
                k_route<<<T, 32, 0, st>>>(K.logits2, ln.probs_b, M.wscale, K.topi2, K.topw2);
                CK(cudaMemcpyAsync(K.h_ids + MAXT * TOPK, K.topi2, (size_t) T * TOPK * 4, cudaMemcpyDeviceToHost, st));
                CK(cudaEventRecord(K.ev_guess, st));
            }
            CK(cudaEventSynchronize(K.ev_route));
            if (p0) prof.stop("usmerjevalnik", st);
            const std::vector<int> ids(K.h_ids, K.h_ids + (size_t) T * TOPK);
            if (c == 0 && trace) {   // HY4_TRACE: (layer, T, picks) for the cache-policy simulator
                const int hd[2] = {il, T};
                std::fwrite(hd, 4, 2, trace);
                std::fwrite(ids.data(), 4, ids.size(), trace);
            }
            std::vector<int> slot;
            std::vector<char> miss;
            experts(c, il, ids, slot, miss);
            const int nj = (int) slot.size();
            static const bool mt_env = getenv("HY4_MT") && std::string(getenv("HY4_MT")) != "0";   // v1x: slower than per pick (lookup 30.7 vs 35.2), off
            if (T > 1 && dec_on && mt_env) {   // v1x: each distinct expert once for its picks (k_iq3_rows_mt), hits before misses
                std::vector<int> dslot, dmiss, dfirst;   // distinct experts in pick order, and where their picks start
                std::vector<int> dpick;                  // the picks, grouped by expert
                {
                    std::unordered_map<int, int> at;
                    std::vector<std::vector<int>> by;
                    for (int j = 0; j < nj; ++j) {
                        auto it = at.find(slot[j]);
                        if (it == at.end()) {
                            it = at.emplace(slot[j], (int) by.size()).first;
                            by.emplace_back();
                            dslot.push_back(slot[j]);
                            dmiss.push_back(miss[j]);
                        }
                        by[(size_t) it->second].push_back(j);
                    }
                    std::vector<int> ord;
                    for (int ph = 0; ph < 2; ++ph)
                        for (int d = 0; d < (int) by.size(); ++d)
                            if (dmiss[(size_t) d] == ph) ord.push_back(d);
                    std::vector<int> s2, m2;
                    for (int d : ord) {
                        s2.push_back(dslot[(size_t) d]);
                        m2.push_back(dmiss[(size_t) d]);
                        dfirst.push_back((int) dpick.size());
                        dpick.insert(dpick.end(), by[(size_t) d].begin(), by[(size_t) d].end());
                    }
                    dslot.swap(s2);
                    dmiss.swap(m2);
                }
                const int nd = (int) dslot.size();
                int ndh = 0;
                for (int d = 0; d < nd; ++d) ndh += !dmiss[(size_t) d];
                for (int j = 0; j < nj; ++j) K.h_plists[j] = make_int2(dpick[(size_t) j] / TOPK, dpick[(size_t) j]);
                int nt = 0;
                for (int ph = 0; ph < 2; ++ph)   // gate and up of the hits, then of the misses
                    for (int m = 0; m < 2; ++m)
                        for (int d = 0; d < nd; ++d)
                            if (dmiss[(size_t) d] == ph) {
                                const int n = (d + 1 < nd ? dfirst[(size_t) d + 1] : nj) - dfirst[(size_t) d];
                                K.h_pjobs[nt++] = {K.pool + (size_t) dslot[(size_t) d] * HALF_EXP + m * HALF_GU, dfirst[(size_t) d], n};
                            }
                for (int d = 0; d < nd; ++d) {
                    const int n = (d + 1 < nd ? dfirst[(size_t) d + 1] : nj) - dfirst[(size_t) d];
                    K.h_pjobs[nt++] = {K.pool + (size_t) dslot[(size_t) d] * HALF_EXP + 2 * HALF_GU, dfirst[(size_t) d], n};
                }
                CK(cudaMemcpyAsync(K.plists, K.h_plists, (size_t) nj * sizeof(int2), cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(K.pjobs, K.h_pjobs, (size_t) nt * sizeof(PJob), cudaMemcpyHostToDevice, st));
                if (K.pref_layer == il) {
                    CK(cudaStreamWaitEvent(st, K.ev_pref, 0));
                    K.pref_layer = -1;
                }
                if (ndh) {
                    k_iq3_rows_mt<4><<<dim3(FFE_C / 32, ndh), 256, 0, st>>>(K.pjobs, K.plists, K.lat, LAT, K.gu_g, FFE_C, FFE_C, LAT, 0);
                    k_iq3_rows_mt<4><<<dim3(FFE_C / 32, ndh), 256, 0, st>>>(K.pjobs + ndh, K.plists, K.lat, LAT, K.gu_u, FFE_C, FFE_C, LAT, 0);
                }
                CK(cudaStreamWaitEvent(st, K.cev, 0));
                if (nd - ndh) {
                    k_iq3_rows_mt<4><<<dim3(FFE_C / 32, nd - ndh), 256, 0, st>>>(K.pjobs + 2 * ndh, K.plists, K.lat, LAT, K.gu_g, FFE_C, FFE_C, LAT, 0);
                    k_iq3_rows_mt<4><<<dim3(FFE_C / 32, nd - ndh), 256, 0, st>>>(K.pjobs + 2 * ndh + (nd - ndh), K.plists, K.lat, LAT, K.gu_u, FFE_C,
                                                                                FFE_C, LAT, 0);
                }
                k_swiglu<<<dim3(4, nj), 256, 0, st>>>(K.gu_g, K.gu_u, K.hh, FFE_C, l.clamp);
                k_iq3_rows_mt<4><<<dim3(LAT / 32, nd), 256, 0, st>>>(K.pjobs + 2 * nd, K.plists, K.hh, FFE_C, K.ey, LAT, LAT, FFE_C, 1);
            } else {
            int nt = 0, nh = 0;
            for (int ph = 0; ph < 2; ++ph)
                for (int m = 0; m < 2; ++m)
                    for (int j = 0; j < nj; ++j)
                        if (miss[j] == ph) K.h_jobs[nt++] = {K.pool + (size_t) slot[j] * HALF_EXP + m * HALF_GU, j / TOPK, j};
            for (int j = 0; j < nj; ++j) nh += !miss[j];
            for (int ph = 0; ph < 2; ++ph)   // the downs: hits first (v1i runs them before the misses' copies are in)
                for (int j = 0; j < nj; ++j)
                    if (miss[j] == ph) K.h_jobs[nt++] = {K.pool + (size_t) slot[j] * HALF_EXP + 2 * HALF_GU, j, j};
            CK(cudaMemcpyAsync(K.jobs, K.h_jobs, (size_t) nt * sizeof(EJob), cudaMemcpyHostToDevice, st));
            if (K.pref_layer == il) {   // a guessed expert counts as a hit: its copy must be done
                CK(cudaStreamWaitEvent(st, K.ev_pref, 0));
                K.pref_layer = -1;
            }
            const EJob* dn = K.jobs + 2 * nj;
            if (nh) {
                iq3(K.jobs, K.lat, LAT, K.gu_g, nh, FFE_C, FFE_C, LAT, st);
                iq3(K.jobs + nh, K.lat, LAT, K.gu_u, nh, FFE_C, FFE_C, LAT, st);
                if (dec_on) {
                    k_swiglu_jobs<<<dim3(4, nh), 256, 0, st>>>(K.gu_g, K.gu_u, K.hh, FFE_C, l.clamp, dn);
                    iq3(dn, K.hh, FFE_C, K.ey, nh, LAT, LAT, FFE_C, st);
                }
            }
            CK(cudaStreamWaitEvent(st, K.cev, 0));
            if (nj - nh) {
                iq3(K.jobs + 2 * nh, K.lat, LAT, K.gu_g, nj - nh, FFE_C, FFE_C, LAT, st);
                iq3(K.jobs + 2 * nh + (nj - nh), K.lat, LAT, K.gu_u, nj - nh, FFE_C, FFE_C, LAT, st);
            }
            if (dec_on) {
                if (nj - nh) {
                    k_swiglu_jobs<<<dim3(4, nj - nh), 256, 0, st>>>(K.gu_g, K.gu_u, K.hh, FFE_C, l.clamp, dn + nh);
                    iq3(dn + nh, K.hh, FFE_C, K.ey, nj - nh, LAT, LAT, FFE_C, st);
                }
            } else {
                k_swiglu<<<dim3(4, nj), 256, 0, st>>>(K.gu_g, K.gu_u, K.hh, FFE_C, l.clamp);
                iq3(dn, K.hh, FFE_C, K.ey, nj, LAT, LAT, FFE_C, st);
            }
            }
            k_moe_sum<<<T, 256, 0, st>>>(K.ey, K.topw, K.moe);
            gemv(l.r_up, K.moe, LAT, K.fo, N, T, st);
            if (!dec_on) k_add<<<64, 256, 0, st>>>(K.fo, K.fo2, (size_t) T * N);   // v1s: else in k_xput_add
            if (guess) {   // the guessed experts of the next layer: copies now (they queue behind this layer's on the copy stream)
                CK(cudaEventSynchronize(K.ev_guess));
                prefetch(c, il + 1, std::vector<int>(K.h_ids + MAXT * TOPK, K.h_ids + MAXT * TOPK + (size_t) T * TOPK));
            }
        }
        if (dec_on) xsum_rpost(c, K.fo, T, l.post_ffw_norm, K.R, K.post, l.dense ? nullptr : K.fo2);
        else {
            xsum(c, K.fo, T);
            rms(K.fo, N, l.post_ffw_norm, K.fo, N, T, N, M.eps, st);
            k_hc_post<<<dim3(T, HCB), 256, 0, st>>>(K.R, K.fo, K.post);
        }
        if (p0) {
            prof.stop(l.dense ? "gosta_plast" : "strokovnjaki_in_ffn", st);
            save("l_out-" + std::to_string(il), K.R, (size_t) T * D);
        }
    }
#include "v1_pf_part.inc"
};

}  // namespace hy4

using namespace hy4;

static std::vector<int> read_tokens(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot read " + path);
    int32_t n = 0;
    if (std::fread(&n, 4, 1, f) != 1) throw std::runtime_error("bad tokens file");
    std::vector<int> v(n);
    if (std::fread(v.data(), 4, n, f) != (size_t) n) throw std::runtime_error("bad tokens file");
    std::fclose(f);
    return v;
}

static std::string detok(const std::string& s) {
    static std::map<int, int> u2b;
    if (u2b.empty()) {
        std::vector<int> bs;
        for (int b = 33; b <= 126; ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b)
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n++); }
        for (size_t i = 0; i < bs.size(); ++i) u2b[cs[i]] = bs[i];
    }
    std::string out;
    for (size_t i = 0; i < s.size();) {
        unsigned c = (unsigned char) s[i];
        int cp, len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 6) { cp = ((c & 31) << 6) | (s[i + 1] & 63); len = 2; }
        else if ((c >> 4) == 14) { cp = ((c & 15) << 12) | ((s[i + 1] & 63) << 6) | (s[i + 2] & 63); len = 3; }
        else { cp = '?'; len = 1; }
        auto it = u2b.find(cp);
        if (it != u2b.end()) out += (char) it->second;
        else out += s.substr(i, len);
        i += len;
    }
    return out;
}

// --serve SOCKET: one JSON line a connection on a Unix socket, {"tokens": [...], "max": N, "stop": [...]} -> {"tokens": [...],
// "prompt_n", "cached", "read_s", "write_s", "hits", "misses"}. A prompt that starts with the tokens already read reads only
// the rest (the caches of those positions hold); the end of turn (eos) and "stop" end a reply
static std::vector<long long> json_ints(const std::string& j, const std::string& key) {
    std::vector<long long> v;
    const size_t k = j.find("\"" + key + "\"");
    if (k == std::string::npos) return v;
    size_t p = j.find('[', k), e = j.find(']', p);
    if (p == std::string::npos || e == std::string::npos) return v;
    const char* c = j.c_str() + p + 1;
    const char* end = j.c_str() + e;
    while (c < end) {
        char* nx;
        const long long x = std::strtoll(c, &nx, 10);
        if (nx == c) { ++c; continue; }
        v.push_back(x);
        c = nx;
    }
    return v;
}
static long long json_int(const std::string& j, const std::string& key, long long def) {
    const size_t k = j.find("\"" + key + "\"");
    if (k == std::string::npos) return def;
    const size_t p = j.find(':', k);
    return std::strtoll(j.c_str() + p + 1, nullptr, 10);
}

static size_t pf_chunk() {   // HY4_PF_CHUNK (default and most HY4_PFC): tokens a prompt-path chunk
    static const size_t n = (size_t) std::max(MAXT + 1, std::min(HY4_PFC, getenv("HY4_PF_CHUNK") ? atoi(getenv("HY4_PF_CHUNK")) : HY4_PFC));
    return n;
}

struct Gen {
    std::vector<int> out;
    int cached = 0;
    double read_s = 0, write_s = 0;
};
static Gen generate(Engine& e, const std::vector<int>& tok, int max, const std::vector<int>& stop, int eos, int lookup = -1, bool reuse = true,
                    bool pf = true, const std::function<void(int)>& emit = {}) {   // emit: each written token as soon as it is known
    Gen g;
    // the prefix already read (at least one token to read again: the last row's logits are needed)
    int L = 0;
    while (reuse && L < (int) e.hist.size() && L < (int) tok.size() - 1 && e.hist[(size_t) L] == tok[(size_t) L]) ++L;
    // the window layers read the SWA - 1 rows before each position from their ring: a prefix is only reusable while no later position
    // has written over those rows of it (a long earlier request wraps the ring; a smaller prefix needs older rows still, so then 0)
    if (L > 0 && std::max(0, L - (SWA - 1)) <= e.top - RING) L = 0;
    if (!L) e.top = -1;
    g.cached = L;
    e.hist.resize((size_t) L);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<int> last;
    // HY4_PF_MIN (default 80): fewer tokens left go through windows. The prompt path stages every picked expert of a layer, nearly
    // all 512 from ~60 tokens on (~1.8 s whatever the count), while windows use the cache (~30 ms a token)
    const size_t pf_min = (size_t) std::max(MAXT + 1, getenv("HY4_PF_MIN") ? atoi(getenv("HY4_PF_MIN")) : 80);
    // HY4_NVPROF=branje: nvprof --profile-from-start off sees the reading only (any other value: the writing only)
    const char* nvs = getenv("HY4_NVPROF");
    const bool nvr = nvs && std::string(nvs) == "branje";
    if (nvr)
        for (int c = 0; c < NC; ++c) {
            CK(cudaSetDevice(c));
            CK(cudaProfilerStart());
        }
    e.prep_p0[0] = e.prep_p0[1] = -1;   // v2f: nothing prepared for this request yet
    for (size_t p = (size_t) L; p < tok.size();) {
        const size_t left = tok.size() - p;
        if (pf && left >= pf_min && !getenv("HY4_PF_OFF")) {
            const int C = (int) std::min<size_t>(pf_chunk(), left);
            const size_t left2 = left - (size_t) C;   // v2f: the next chunk, prepared on the host while this one runs
            const int C2 = left2 >= pf_min ? (int) std::min<size_t>(pf_chunk(), left2) : 0;
            last.assign(1, e.prefill(tok.data() + p, (int) p, C, C2 ? tok.data() + p + C : nullptr, C2));
            p += C;
        } else {
            const int T = (int) std::min<size_t>(MAXT, left);
            last = e.window(tok.data() + p, (int) p, T);
            p += T;
        }
        e.top = std::max(e.top, (int) p - 1);
    }
    if (nvr)
        for (int c = 0; c < NC; ++c) {
            CK(cudaSetDevice(c));
            CK(cudaDeviceSynchronize());
            CK(cudaProfilerStop());
        }
    auto t1 = std::chrono::steady_clock::now();
    g.read_s = std::chrono::duration<double>(t1 - t0).count();
    int next = last.back();
    // HY4_LOOKUP=K (default 8, 0 off): prompt lookup drafts. The last 4 (then 3, 2) tokens of the history are looked up at their latest
    // earlier place; up to K of the tokens that followed are drafted and checked in one window with the pending token. The accepted
    // prefix stays; positions past it are simply read again later (their cache rows, indexer rows and pooled groups are written anew
    // before anything reads them), so a rejection costs nothing but the window
    const int K = std::max(0, std::min(MAXT - 1, lookup >= 0 ? lookup : getenv("HY4_LOOKUP") ? atoi(getenv("HY4_LOOKUP")) : 8));
    std::vector<int> seq(tok);   // the history as the drafts see it: prompt, then every accepted token
    // v1j: a draft needs a match of HY4_LOOKUP_MIN (default 3) tokens, and its length follows the last result: all taken doubles
    // it (up to K), otherwise it becomes the taken count plus one (code written anew rejects most drafts, and a window of 9 costs
    // far more experts than one of 1)
    const int mmin = std::max(2, std::min(4, getenv("HY4_LOOKUP_MIN") ? atoi(getenv("HY4_LOOKUP_MIN")) : 3));
    int kcur = K;
    auto is_end = [&](int t) { return t == eos || std::find(stop.begin(), stop.end(), t) != stop.end(); };
    auto draft = [&]() {
        std::vector<int> d;
        if (!K) return d;
        const int n = (int) seq.size();
        for (int m = 4; m >= mmin && d.empty(); --m) {
            if (n < m + 1) continue;
            for (int i = n - m - 1; i >= 0; --i) {
                bool same = true;
                for (int j = 0; j < m && same; ++j) same = seq[(size_t) i + j] == seq[(size_t) n - m + j];
                if (!same) continue;
                for (int j = i + m; j < n && (int) d.size() < kcur; ++j) d.push_back(seq[(size_t) j]);
                break;
            }
        }
        return d;
    };
    const bool nvp = nvs && !nvr;
    if (nvp)
        for (int c = 0; c < NC; ++c) {
            CK(cudaSetDevice(c));
            CK(cudaProfilerStart());
        }
    while ((int) g.out.size() < max) {
        g.out.push_back(next);
        seq.push_back(next);
        if (emit) emit(next);
        if (is_end(next) || (int) g.out.size() >= max) break;
        const int pos = (int) seq.size() - 1;   // where `next` goes
        std::vector<int> d = draft();
        if ((int) d.size() > max - (int) g.out.size()) d.resize((size_t) (max - (int) g.out.size()));
        if (d.empty()) {
            next = e.window(&next, pos, 1)[0];
            e.top = std::max(e.top, pos);
            continue;
        }
        std::vector<int> w(1, next);
        w.insert(w.end(), d.begin(), d.end());
        const std::vector<int> am = e.window(w.data(), pos, (int) w.size());
        e.top = std::max(e.top, pos + (int) w.size() - 1);
        size_t a = 0;   // drafts accepted: row i predicted d[i]
        while (a < d.size() && am[a] == d[a] && !is_end(d[a])) ++a;
        for (size_t i = 0; i < a && (int) g.out.size() < max; ++i) {
            g.out.push_back(d[i]);
            seq.push_back(d[i]);
            if (emit) emit(d[i]);
        }
        e.drafted += d.size();
        e.accepted += a;
        kcur = a == d.size() ? std::min(K, 2 * kcur) : std::max(1, (int) a + 1);
        e.hist.resize((size_t) pos + a + 1);   // the rows past the accepted ones are stale: the history ends at the last kept token
        next = am[a];
        if (a < d.size() && is_end(d[a]) && am[a] == d[a]) next = d[a];
    }
    g.write_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    if (nvp)
        for (int c = 0; c < NC; ++c) {
            CK(cudaSetDevice(c));
            CK(cudaProfilerStop());
        }
    return g;
}

static std::string b64(const std::string& in) {
    static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const unsigned v = (unsigned char) in[i] << 16 | (unsigned char) in[i + 1] << 8 | (unsigned char) in[i + 2];
        o += t[v >> 18]; o += t[(v >> 12) & 63]; o += t[(v >> 6) & 63]; o += t[v & 63];
    }
    if (i + 1 == in.size()) {
        const unsigned v = (unsigned char) in[i] << 16;
        o += t[v >> 18]; o += t[(v >> 12) & 63]; o += "==";
    } else if (i + 2 == in.size()) {
        const unsigned v = (unsigned char) in[i] << 16 | (unsigned char) in[i + 1] << 8;
        o += t[v >> 18]; o += t[(v >> 12) & 63]; o += t[(v >> 6) & 63]; o += '=';
    }
    return o;
}

static void serve(Engine& e, const std::string& path, int eos) {
    std::signal(SIGPIPE, SIG_IGN);   // a client gone mid-stream must not end the server
    unlink(path.c_str());
    const int s = socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    std::snprintf(a.sun_path, sizeof a.sun_path, "%s", path.c_str());
    if (bind(s, (sockaddr*) &a, sizeof a) || listen(s, 4)) throw std::runtime_error("serve: cannot listen on " + path);
    std::fprintf(stderr, "hy4_v1: serving on %s\n", path.c_str());
    const char* lastlog = getenv("HY4_LASTLOG");   // the last request's speeds, in the format of our own monitor app (off by default)
    if (lastlog)
        if (FILE* zl = std::fopen(lastlog, "w")) std::fclose(zl);   // one file a server start
    for (;;) {
        const int c = accept(s, nullptr, nullptr);
        if (c < 0) continue;
        std::string req;
        char buf[65536];
        for (;;) {
            const ssize_t n = read(c, buf, sizeof buf);
            if (n <= 0) break;
            req.append(buf, (size_t) n);
            if (req.find('\n') != std::string::npos) break;
        }
        std::string resp;
        if (json_int(req, "exit", 0)) {   // {"exit": 1}: a clean end (nvprof writes its summary at exit)
            (void) !write(c, "{\"exit\": 1}\n", 12);
            close(c);
            close(s);
            return;
        }
        try {
            std::vector<int> tok, stop;
            for (long long x : json_ints(req, "tokens")) tok.push_back((int) x);
            for (long long x : json_ints(req, "stop")) stop.push_back((int) x);
            const int ctxn = e.M.ctx;   // v1v: a prompt past the context ran off the latent cache (illegal access); the answer stops at it
            const int max = (int) std::min<long long>(json_int(req, "max", 256), std::max<long long>(1, (long long) ctxn - (long long) tok.size()));
            if (tok.empty()) throw std::runtime_error("no tokens");
            if ((int) tok.size() + 1 > ctxn)
                throw std::runtime_error("the prompt (" + std::to_string(tok.size()) + " tokens) is longer than the context (" + std::to_string(ctxn) + ")");
            const size_t h0 = e.hits(), m0 = e.misses(), d0 = e.drafted, a0 = e.accepted;
            e.prefetch_mode = (int) json_int(req, "prefetch", -1);
            const long long dec = json_int(req, "dec", -1);   // v1g's writing kernels: 1 on, 0 v1's, -1 HY4_DEC_OFF decides
            e.dec_on = dec < 0 ? getenv("HY4_DEC_OFF") == nullptr : dec != 0;
            g_g1_fused = e.dec_on;
            e.pf_gemm_on = json_int(req, "pfg", 1) != 0;
            e.pf_selh_on = json_int(req, "selh", 1) != 0;
            auto env_int = [](const char* k, int d) { return getenv(k) ? atoi(getenv(k)) : d; };
            e.pf_host = json_int(req, "host", env_int("HY4_PF_HOST", 1)) != 0;   // v2: see v1_pf_part.inc
            e.pf_selk = (int) json_int(req, "selk", env_int("HY4_PF_SELK", 1));
            e.pf_skip = json_int(req, "skip", env_int("HY4_PF_SKIP", 1)) != 0;
            e.pf_xs = json_int(req, "xs", env_int("HY4_PF_XS", 0)) != 0;
            e.pf_f16 = (int) json_int(req, "f16", env_int("HY4_PF_F16", 0));
            const size_t p0c = e.pref_copies;
            // "stream": 1 - a line {"tok": id, "b": base64 of its bytes} per written token before the summary line (tokens can end
            // inside a UTF-8 character: the client joins the bytes)
            std::function<void(int)> emit;
            if (json_int(req, "stream", 0))
                emit = [&](int t) {
                    const std::string line = "{\"tok\": " + std::to_string(t) + ", \"b\": \"" + (t == eos ? std::string() : b64(detok(e.M.vocab[(size_t) t]))) + "\"}\n";
                    (void) !send(c, line.data(), line.size(), MSG_NOSIGNAL);
                };
            Gen g = generate(e, tok, max, stop, eos, (int) json_int(req, "lookup", -1), json_int(req, "reuse", 1) != 0,
                             json_int(req, "pf", 1) != 0, emit);
            e.prof.print(1);   // HY4_PROF: ms by part for the whole request
            if (e.trace) std::fflush(e.trace);
            if (e.guess_stat) {
                unsigned long long ov[2] = {};
                CK(cudaSetDevice(0));
                CK(cudaMemcpyFromSymbol(ov, g_guess_ov, sizeof ov));
                std::fprintf(stderr, "guess from the attention input: %llu / %llu picks (%.1f %%)\n", ov[0], ov[1], ov[1] ? 100.0 * ov[0] / ov[1] : 0.0);
            }
            // HY4_LASTLOG: the lines our Windows monitor app reads (this exact format)
            if (FILE* zl = lastlog ? std::fopen(lastlog, "a") : nullptr) {
                const int nr = (int) tok.size() - g.cached, nw = (int) g.out.size();
                std::fprintf(zl, "branje: %d zetonov v %.2f s (%.1f tok/s)\n", nr, g.read_s, nr / std::max(g.read_s, 1e-9));
                std::fprintf(zl, "pisanje: %d zetonov v %.2f s (%.2f tok/s)\n", nw, g.write_s, nw / std::max(g.write_s, 1e-9));
                std::fclose(zl);
            }
            char hd[384];
            std::snprintf(hd, sizeof hd,
                          "{\"prompt_n\": %zu, \"cached\": %d, \"read_s\": %.3f, \"write_s\": %.3f, \"hits\": %zu, \"misses\": %zu, \"drafted\": %zu, "
                          "\"accepted\": %zu, \"prefetched\": %zu, \"tokens\": [",
                          tok.size(), g.cached, g.read_s, g.write_s, e.hits() - h0, e.misses() - m0, e.drafted - d0, e.accepted - a0, e.pref_copies - p0c);
            resp = hd;
            for (size_t i = 0; i < g.out.size(); ++i) resp += (i ? ", " : "") + std::to_string(g.out[i]);
            std::string text;
            for (int t : g.out)
                if (t != eos) text += detok(e.M.vocab[(size_t) t]);
            std::string esc;
            for (unsigned char ch : text) {
                if (ch == '"' || ch == '\\') { esc += '\\'; esc += (char) ch; }
                else if (ch == '\n') esc += "\\n";
                else if (ch == '\t') esc += "\\t";
                else if (ch < 0x20) { char u[8]; std::snprintf(u, sizeof u, "\\u%04x", ch); esc += u; }
                else esc += (char) ch;
            }
            resp += "], \"text\": \"" + esc + "\"}\n";
        } catch (const std::exception& ex) {
            resp = std::string("{\"error\": \"") + ex.what() + "\"}\n";
        }
        (void) !write(c, resp.data(), resp.size());
        close(c);
    }
}

int main(int argc, char** argv) {
    std::string model, tokens, dump, sock;
    int gen = 8, slots = 7000, ctx = 8192;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--model") model = nx();
        else if (a == "--tokens") tokens = nx();
        else if (a == "--dump") dump = nx();
        else if (a == "--gen") gen = std::stoi(nx());
        else if (a == "--slots") slots = std::stoi(nx());
        else if (a == "--ctx") ctx = std::stoi(nx());
        else if (a == "--serve") sock = nx();
    }
    try {
        // never destroyed: card 1's host thread (detached) keeps reading the engine until the process ends
        Model& m = *new Model(model, ctx);
        Engine& e = *new Engine(m, slots);
        e.dump = dump;
        e.pf_host = !getenv("HY4_PF_HOST") || atoi(getenv("HY4_PF_HOST")) != 0;   // serve: also a request's "host", "selk", "f16"
        e.pf_selk = getenv("HY4_PF_SELK") ? atoi(getenv("HY4_PF_SELK")) : 1;
        e.pf_skip = !getenv("HY4_PF_SKIP") || atoi(getenv("HY4_PF_SKIP")) != 0;
        e.pf_xs = getenv("HY4_PF_XS") && atoi(getenv("HY4_PF_XS")) != 0;
        e.pf_f16 = getenv("HY4_PF_F16") ? atoi(getenv("HY4_PF_F16")) : 0;
        const int eos = (int) m.g.i("tokenizer.ggml.eos_token_id");
        if (!sock.empty()) {
            serve(e, sock, eos);
            return 0;
        }
        std::vector<int> tok = read_tokens(tokens);
        std::fprintf(stderr, "hy4_v1: %zu prompt tokens\n", tok.size());
        auto t0 = std::chrono::steady_clock::now();
        std::vector<int> last;
        for (size_t p = 0; p < tok.size();) {
            const size_t left = tok.size() - p;
            if (left > (size_t) MAXT && !getenv("HY4_PF_OFF")) {
                const int C = (int) std::min<size_t>(pf_chunk(), left);
                last.assign(1, e.prefill(tok.data() + p, (int) p, C));
                p += C;
            } else {
                const int T = (int) std::min<size_t>(MAXT, left);
                last = e.window(tok.data() + p, (int) p, T);
                p += T;
            }
        }
        const double rs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("read: %zu tokens in %.2f s (%.1f tok/s), hits %zu / misses %zu\n", tok.size(), rs, tok.size() / rs, e.hits(), e.misses());
        e.prof.ms.clear();
        e.dump.clear();
        int next = last.back();
        std::string text;
        std::vector<int> outt;
        auto t2 = std::chrono::steady_clock::now();
        for (int g = 0; g < gen; ++g) {
            outt.push_back(next);
            text += detok(m.vocab[(size_t) next]);
            next = e.window(&next, (int) (tok.size() + g), 1)[0];
        }
        const double ws = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
        e.prof.print(gen);
        std::printf("write: %d tokens in %.2f s (%.2f tok/s), hits %zu / misses %zu\ntokens:", gen, ws, gen / ws, e.hits(), e.misses());
        for (int t : outt) std::printf(" %d", t);
        std::printf("\ntext: %s\n", text.c_str());
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
    return 0;
}
