// ink2_v1.cu (2026-11-21): Inkling 2 Mini on both P100s at once, after Hy4's hy4_v1. Every layer pass runs on both cards: the
// attention's heads are divided (12 / 12 query heads, 4 / 4 KV heads; each card keeps only its heads' KV), every expert's FF width
// (576 / 576: gate and up rows, down columns), the output vocabulary (100512 / 100512). The depth mixing, the loop's messages and
// controllers, the norms, the router and the post-convolutions run on both cards on the same input (the same bits). Two sums
// across the cards a layer pass (the attention's wo and the MoE output) through page-locked mapped RAM (k_xput / k_xget).
// The whole model stays in VRAM (~8.6 GB a card). Prompts run in windows of MAXT tokens (the writing kernels); see VODNIK-inkling2.md.
//   ink2_v1 --model M.gguf --tokens zetoni.bin [--gen N] [--dump DIR] [--ctx C]
//   ink2_v1 --model M.gguf --serve SOCKET [--ctx C]
#include "ink2_kern.cuh"
#include "ink2_pf.cuh"
#include "ink2_dspark.cuh"
#include "gguf_min.hpp"

#include <cublas_v2.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace ink2 {

static std::string g_draft_vocab;   // p13: --draft-vocab FILE (int32 count + token ids), "" = the whole vocabulary
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct LW {   // one card's part of a physical layer
    bool full = false;
    int nrq = 0;
    const float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr, *attn_rq = nullptr, *ffn_rq = nullptr;
    Mat wq, wk, wv, wo, wgate, wrel, wsink, wkd, wvd;
    Mat cat;   // p4a: [wq | wgate | wrel | wsink | wk | (wkd) | wv | (wvd)] in one Q8S matrix; the Mats above are views into it
    const float* rel_bias_pad = nullptr;   // p3: [EXT + 1][NREL], the last row 0 (the reading path's R.B table)
    const float *q_clip, *rel_clip, *rel_bias, *k_conv, *v_conv, *a_pconv, *f_pconv, *ffn_norm, *gate_inp, *probs_b;
    bool fp4 = false;
    const uint8_t *gu = nullptr, *dn = nullptr;   // NVFP4 rows, or Q8S values
    const __half *gu_d = nullptr, *dn_d = nullptr;
    const float *gu_s = nullptr, *dn_s = nullptr;
    const float* msg_q = nullptr;   // loop layers 6-17
    const float *c_down[4] = {}, *c_gain[4] = {}, *c_forget[4] = {};
};

// p6: the DSpark drafter on one card (whole, on both cards)
struct DSW {
    Mat wq[DSL], wk[DSL], wv[DSL], wo[DSL], kctx[DSL], vctx[DSL], gate[DSL], up[DSL], down[DSL];
    Mat dvout;   // p13: the output rows of this card's half of the draft vocabulary
    const float *attn_norm[DSL], *ffn_norm[DSL], *dflare[DSL], *qn[DSL], *kn[DSL], *kcn[DSL], *fuse, *mask, *dsnorm, *outnorm, *jb, *hb;
    const uint16_t *w1, *jw, *hw;
    __half *kc[DSL], *vc[DSL];
    float *feat, *frnn, *ctxv, *cbuf, *kb, *vb, *x, *xn, *q, *k, *v, *ao, *o, *g, *u, *h, *part, *hid, *state, *jin, *jout, *lat, *y, *amv, *ams;
    float *pfeat = nullptr, *pfrnn = nullptr, *pcbuf = nullptr, *pkb = nullptr, *pvb = nullptr;   // the reading path's rows
    int *ami, *prev, *bpos;
    int nparts = 0;
};

// p3: one card's buffers of the reading path (a chunk of up to C tokens)
struct PFB {
    int C = 0;
    size_t S_n = 0;   // p14: floats in S
    float* PV = nullptr;   // p14b: P.V parts by key pieces [3][nsbmax][QSF][HD]
    int pv_nsb = 0;
    cublasHandle_t cb;
    float *x0, *cur, *B[MAXB], *prel, *bankS, *bankO, *X3, *MS, *XC[4], *xm, *g64, *gp, *fpp, *NQ, *NK, *NV, *q, *k, *v, *kcv, *vcv, *r,
        *sink, *gate, *dk, *dv, *ao, *ab, *A, *F, *F2, *FN, *lg, *topw, *MO, *tau, *Xg, *GU, *HH, *EY, *Wf, *Wo, *Egu, *Edn, *Kf, *Vf, *S,
        *RB, *tmp, *hx1;
    float **pk = nullptr, **pq, **ps, **pv, **pao;   // p3f: pointer arrays of the window layers' batched attention
    int *pos, *topi, *cnt, *offs, *cursor, *slotrow, *rowtok, *rowexp;
    float* hbuf[2] = {};   // page-locked: this card's part of an exchange, by parity
    cudaEvent_t ev[2];
    std::vector<int> h_cnt, h_offs;
    unsigned seq = 1;
};

struct Card {
    PFB P;
    DSW D;   // p6
    int dev = 0, c = 0;
    cudaStream_t st;
    LW L[NPHYS];
    Mat out;
    const float *tok_norm, *out_norm, *out_rq;
    size_t vram = 0;
    unsigned seq = 1;
    // per slot: KV caches (f16) and the convolutions' last 3 inputs
    __half *kc[NSLOT], *vc[NSLOT];
    float *sk[NSLOT], *sv[NSLOT], *sa[NSLOT], *sf[NSLOT];
    // scratch
    float *x0, *cur, *B[MAXB], *prel, *bankS, *bankO, *X3, *MS, *XC[4], *xm, *g64, *gp, *fpp, *NQ, *NK, *NV, *q, *k, *v, *kcv, *vcv, *r,
        *sink, *gate, *dk, *dv, *ao, *part, *ab, *A, *F, *F2, *FN, *lg, *topw, *GU, *HH, *EY, *MO, *hx, *logits, *tau;
    int *pos, *topi;
    EJob* jobs;
    int* perm = nullptr;   // p11
    MJob* mjobs;   // p5b
    int* nmjobs;
    int nb = 0, nparts_max = 0;
    unsigned* dtok = nullptr;                   // p4c: forwards so far (device), the exchanges' sequence numbers
    float *stbuf = nullptr, *stbak = nullptr, *hist = nullptr;   // p5a: conv states, their snapshot, a window's inputs
    float* stsnap = nullptr;   // p8: the conv states before the last prompt token (the next request of a conversation goes back there)
    bool keep_hist = false;
    unsigned xidx = 0;                          // p4c: the next exchange of this forward
    std::map<int, cudaGraphExec_t> graphs;      // p4c: one T = 1 forward a parts bucket
};

struct Xch {
    float* buf[NC][2];
    volatile unsigned* flag[NC];
};

struct Engine {
    GgufMin g;
    Card cards[NC];
    Xch X;
    const GTensor* tok = nullptr;
    int ctx;
    bool stare = getenv("INK2_STARE") != nullptr;   // p4a: v1's writing kernels
    bool gemm2 = getenv("INK2_GEMM2") != nullptr;   // p3e
    bool win_seg = getenv("INK2_WIN_SEG") != nullptr;   // p6c off: windows' dense rows by k_gemv8_seg (other f32 order than T = 1)
    bool mt_off = getenv("INK2_MT") == nullptr;   // p5b: windows read each picked expert once (INK2_MT=1; measured 21. 11.: slower, off)
    bool attn_old = getenv("INK2_ATTN_OLD") != nullptr;   // p4f off: the attention by query head (k_attn_part)
    bool fuse = getenv("INK2_FUSE_OFF") == nullptr;   // p4e: fused writing kernels at T = 1
    bool ilv = getenv("INK2_FP4_ILV") != nullptr;   // p4d off: NVFP4 experts interleaved as in the file (the older kernels need it)
    bool rows2 = getenv("INK2_ROWS2") != nullptr;   // p4b off: p4a's NVFP4 writing kernel
    bool pf_deq = getenv("INK2_PF_DEQ") != nullptr;  // p3c: NVFP4 experts of the reading path dequantized to f32 + cuBLAS (p3b)
    float eps, qk_clip, rel_clip_thr, log_alpha, log_floor, situ_b, situ_lb, wscale, softcap;
    // host side of a step
    float* h_x0 = nullptr;   // p4c: page-locked (a graph's memcpy nodes read them at each replay)
    int* h_pos = nullptr;
    float* h_tau = nullptr;
    bool graf = getenv("INK2_GRAF") != nullptr;   // p4c: the writing path's forward as a CUDA graph (measured 21. 11.: no gain, off)
    int pfc = 2048;                           // p3: INK2_PF_CHUNK, tokens a reading chunk
    int pf_min = MAXT + 1;                    // p3: INK2_PF_MIN, shortest piece that goes the reading path (INK2_PF_OFF: never)
    std::atomic<unsigned> pf_rec[NC];         // p3: the exchange each card has recorded
    static constexpr int EB = 16, QS = 256;   // p3: experts dequantized at once; queries per attention block
    bool perm_on = getenv("INK2_PERM") ? atoi(getenv("INK2_PERM")) != 0 : false;   // p11: a window's expert jobs by expert (21. 11.: same speed, off)
    int feed_t = getenv("INK2_FEED_T") ? std::max(1, std::min(MAXT, atoi(getenv("INK2_FEED_T")))) : MAXT;   // windows of the writing path when reading (test: 1 = token by token)
    bool t1tt_on = getenv("INK2_T1TT") ? atoi(getenv("INK2_T1TT")) != 0 : true;   // p9: window GEMV with T at compile time
    bool ds_t1 = getenv("INK2_DS_T1") ? atoi(getenv("INK2_DS_T1")) != 0 : true;      // p9: the drafter's GEMV by gemv8_t1T
    bool pf_attb = getenv("INK2_PF_ATTB") ? atoi(getenv("INK2_PF_ATTB")) != 0 : true;   // p3f: window layers' attention in one batched call
    bool pf_rb = getenv("INK2_PF_RB") ? atoi(getenv("INK2_PF_RB")) != 0 : false;        // p3f: R.B by k_rb instead of cuBLAS (no gain, off)
    int pf_qs_full = getenv("INK2_PF_QSF") ? atoi(getenv("INK2_PF_QSF")) : 1024;
    static constexpr int KFB = 32768;   // p14: full layers' keys converted to f32 at most this many at a time (longer: in blocks)
    static constexpr int PVK = 1024;    // p14b: the full layers' P.V in pieces of this many keys (one batched call, then k_pv_sum)
    bool pv_split = getenv("INK2_PV_SPLIT") ? atoi(getenv("INK2_PV_SPLIT")) != 0 : true;
    static constexpr int DS_RING = 4096, DS_RING1 = 16384, DS_WIN1 = 12288;   // p14: the drafter's caches above 64k (layer 1 sees 12k)
    bool ds_ring = false;   // 21. 11.: 14k 641 -> 695 (2048: 700, +0.7 GB)   // p3d: queries per block in the full layers
    float* h_logits = nullptr;   // page-locked [MAXT][NV]
    int n_past = 0;
    int snap_n = -1;   // p8: tokens before the snapshot of the conv states (stsnap), -1 = none
    std::vector<int> processed;   // the tokens whose state the caches hold
    // dumps (card 0): per slot l_out, attn_out, ffn_out rows of every window
    bool dumping = false;
    std::vector<std::vector<float>> d_lout, d_aout, d_fout;
    // card 1's worker
    std::thread w1;
    std::atomic<int> go{0}, done{0};
    std::function<void(int)> job;
    bool quit = false;

    void* alloc(Card& K, size_t bytes) {
        void* p = nullptr;
        CK(cudaSetDevice(K.dev));
        CK(cudaMalloc(&p, bytes));
        K.vram += bytes;
        return p;
    }
    float* dev(Card& K, size_t floats) {
        float* p = (float*) alloc(K, floats * 4);
        CK(cudaMemset(p, 0, floats * 4));
        return p;
    }
    GgufMin* dsg = nullptr;   // p6: the drafter's file (--mtp)
    bool ds_on = false;
    const float* f32(Card& K, const std::string& name, size_t off = 0, size_t n = 0, GgufMin* gg = nullptr) {
        const GTensor& t = (gg ? *gg : g).t(name);
        if (t.type != G_F32) throw std::runtime_error(name + " is not F32");
        const size_t bytes = n ? n * 4 : t.nbytes;
        void* p = alloc(K, bytes);
        CK(cudaMemcpy(p, t.data + off * 4, bytes, cudaMemcpyHostToDevice));
        return (const float*) p;
    }
    // rows [r0, r1) and columns [c0, c1) of a Q8_0 matrix (rows = ne1 * ne2, cols = ne0) into Q8S
    Mat q8(Card& K, const std::string& name, int r0 = 0, int r1 = -1, int c0 = 0, int c1 = -1, GgufMin* gg = nullptr) {
        const GTensor& t = (gg ? *gg : g).t(name);
        if (t.type != G_Q8_0) throw std::runtime_error(name + " is not Q8_0");
        const int rows = (int) (t.ne[1] * t.ne[2] * t.ne[3]), cols = (int) t.ne[0];
        if (r1 < 0) r1 = rows;
        if (c1 < 0) c1 = cols;
        if (c0 % 32 || c1 % 32) throw std::runtime_error(name + ": column slice off the blocks");
        Mat m;
        m.type = G_Q8_0;
        m.rows = r1 - r0;
        m.cols = c1 - c0;
        const int nb = m.cols / 32;
        const size_t rb = t.row_bytes();
        std::vector<int8_t> qv((size_t) m.rows * m.cols);
        std::vector<uint16_t> dv((size_t) m.rows * nb);
        for (int r = 0; r < m.rows; ++r) {
            const uint8_t* src = t.data + (size_t) (r0 + r) * rb + (size_t) (c0 / 32) * 34;
            for (int b = 0; b < nb; ++b) {
                std::memcpy(&dv[(size_t) r * nb + b], src + b * 34, 2);
                std::memcpy(&qv[(size_t) r * m.cols + b * 32], src + b * 34 + 2, 32);
            }
        }
        void* pq = alloc(K, qv.size());
        void* pd = alloc(K, dv.size() * 2);
        CK(cudaMemcpy(pq, qv.data(), qv.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(pd, dv.data(), dv.size() * 2, cudaMemcpyHostToDevice));
        m.q = (const int8_t*) pq;
        m.d = pd;
        return m;
    }
    // p13: the given rows of a Q8_0 matrix as one Q8S matrix
    Mat q8_pick(Card& K, const std::string& name, const std::vector<int>& pick) {
        const GTensor& t = g.t(name);
        if (t.type != G_Q8_0) throw std::runtime_error(name + " is not Q8_0");
        Mat m;
        m.type = G_Q8_0;
        m.rows = (int) pick.size();
        m.cols = (int) t.ne[0];
        const int nb = m.cols / 32;
        const size_t rb = t.row_bytes();
        std::vector<int8_t> qv((size_t) m.rows * m.cols);
        std::vector<uint16_t> dv((size_t) m.rows * nb);
        for (int r = 0; r < m.rows; ++r) {
            const uint8_t* src = t.data + (size_t) pick[r] * rb;
            for (int b = 0; b < nb; ++b) {
                std::memcpy(&dv[(size_t) r * nb + b], src + b * 34, 2);
                std::memcpy(&qv[(size_t) r * m.cols + b * 32], src + b * 34 + 2, 32);
            }
        }
        void* pq = alloc(K, qv.size());
        void* pd = alloc(K, dv.size() * 2);
        CK(cudaMemcpy(pq, qv.data(), qv.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(pd, dv.data(), dv.size() * 2, cudaMemcpyHostToDevice));
        m.q = (const int8_t*) pq;
        m.d = pd;
        return m;
    }
    // p4a: several row ranges of Q8_0 matrices with the same columns as one Q8S matrix (rows one after another)
    Mat q8cat(Card& K, const std::vector<std::tuple<std::string, int, int>>& parts, std::vector<Mat>& views) {
        int rows = 0, cols = -1;
        for (auto& [name, r0, r1] : parts) {
            const GTensor& t = g.t(name);
            if (t.type != G_Q8_0) throw std::runtime_error(name + " is not Q8_0");
            if (cols >= 0 && cols != (int) t.ne[0]) throw std::runtime_error(name + ": columns differ");
            cols = (int) t.ne[0];
            rows += r1 - r0;
        }
        const int nb = cols / 32;
        std::vector<int8_t> qv((size_t) rows * cols);
        std::vector<uint16_t> dv((size_t) rows * nb);
        int o = 0;
        std::vector<int> starts;
        for (auto& [name, r0, r1] : parts) {
            const GTensor& t = g.t(name);
            const size_t rb = t.row_bytes();
            starts.push_back(o);
            for (int r = r0; r < r1; ++r, ++o) {
                const uint8_t* src = t.data + (size_t) r * rb;
                for (int b = 0; b < nb; ++b) {
                    std::memcpy(&dv[(size_t) o * nb + b], src + b * 34, 2);
                    std::memcpy(&qv[(size_t) o * cols + b * 32], src + b * 34 + 2, 32);
                }
            }
        }
        Mat m;
        m.type = G_Q8_0;
        m.rows = rows;
        m.cols = cols;
        void* pq = alloc(K, qv.size());
        void* pd = alloc(K, dv.size() * 2);
        CK(cudaMemcpy(pq, qv.data(), qv.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(pd, dv.data(), dv.size() * 2, cudaMemcpyHostToDevice));
        m.q = (const int8_t*) pq;
        m.d = pd;
        views.clear();
        for (size_t i = 0; i < parts.size(); ++i) {
            Mat v = m;
            v.rows = std::get<2>(parts[i]) - std::get<1>(parts[i]);
            v.q = m.q + (size_t) starts[i] * cols;
            v.d = (const __half*) m.d + (size_t) starts[i] * nb;
            views.push_back(v);
        }
        return m;
    }
    // this card's part of a layer's experts: gate_up rows [c*FF_C, +FF_C) and [FF + c*FF_C, +FF_C), down columns [c*FF_C, +FF_C)
    void experts(Card& K, LW& L, int p) {
        const std::string b = "blk." + std::to_string(p) + ".";
        const GTensor& gu = g.t(b + "ffn_gate_up_exps.weight");
        const GTensor& dn = g.t(b + "ffn_down_exps.weight");
        const int c = K.c;
        if (gu.type == G_NVFP4) {
            L.fp4 = true;
            const size_t rbg = gu.row_bytes(), rbd = dn.row_bytes(), cb = (size_t) (FF_C / 64) * 36;   // 1296, 648, 324
            std::vector<uint8_t> hg((size_t) NE * 2 * FF_C * rbg), hd((size_t) NE * N * cb);
            for (int e = 0; e < NE; ++e) {
                for (int h = 0; h < 2; ++h)
                    std::memcpy(&hg[((size_t) e * 2 * FF_C + h * FF_C) * rbg], gu.data + ((size_t) e * 2 * FF + h * FF + c * FF_C) * rbg,
                                (size_t) FF_C * rbg);
                for (int r = 0; r < N; ++r)
                    std::memcpy(&hd[((size_t) e * N + r) * cb], dn.data + ((size_t) e * N + r) * rbd + (size_t) c * cb, cb);
            }
            if (!ilv) {   // p4d: planar experts (codes, then scales)
                std::vector<uint8_t> t(hg.size());
                fp4_to_planar(hg.data(), t.data(), (size_t) 2 * FF_C, N / 64, NE);
                hg.swap(t);
                t.resize(hd.size());
                fp4_to_planar(hd.data(), t.data(), (size_t) N, FF_C / 64, NE);
                hd.swap(t);
            }
            void* pg = alloc(K, hg.size());
            void* pd = alloc(K, hd.size());
            CK(cudaMemcpy(pg, hg.data(), hg.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(pd, hd.data(), hd.size(), cudaMemcpyHostToDevice));
            L.gu = (const uint8_t*) pg;
            L.dn = (const uint8_t*) pd;
            L.gu_s = f32(K, b + "ffn_gate_up_exps.scale");
            L.dn_s = f32(K, b + "ffn_down_exps.scale");
        } else if (gu.type == G_Q8_0) {
            L.fp4 = false;
            const size_t rbg = gu.row_bytes(), rbd = dn.row_bytes();
            std::vector<int8_t> qg((size_t) NE * 2 * FF_C * N), qd((size_t) NE * N * FF_C);
            std::vector<uint16_t> dg((size_t) NE * 2 * FF_C * (N / 32)), dd((size_t) NE * N * (FF_C / 32));
            for (int e = 0; e < NE; ++e) {
                for (int h = 0; h < 2; ++h)
                    for (int r = 0; r < FF_C; ++r) {
                        const uint8_t* src = gu.data + ((size_t) e * 2 * FF + h * FF + c * FF_C + r) * rbg;
                        const size_t row = (size_t) e * 2 * FF_C + h * FF_C + r;
                        for (int bb = 0; bb < N / 32; ++bb) {
                            std::memcpy(&dg[row * (N / 32) + bb], src + bb * 34, 2);
                            std::memcpy(&qg[row * N + bb * 32], src + bb * 34 + 2, 32);
                        }
                    }
                for (int r = 0; r < N; ++r) {
                    const uint8_t* src = dn.data + ((size_t) e * N + r) * rbd + (size_t) (c * FF_C / 32) * 34;
                    const size_t row = (size_t) e * N + r;
                    for (int bb = 0; bb < FF_C / 32; ++bb) {
                        std::memcpy(&dd[row * (FF_C / 32) + bb], src + bb * 34, 2);
                        std::memcpy(&qd[row * FF_C + bb * 32], src + bb * 34 + 2, 32);
                    }
                }
            }
            void* a = alloc(K, qg.size());
            void* bq = alloc(K, qd.size());
            void* cdg = alloc(K, dg.size() * 2);
            void* cdd = alloc(K, dd.size() * 2);
            CK(cudaMemcpy(a, qg.data(), qg.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(bq, qd.data(), qd.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(cdg, dg.data(), dg.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(cdd, dd.data(), dd.size() * 2, cudaMemcpyHostToDevice));
            L.gu = (const uint8_t*) a;
            L.dn = (const uint8_t*) bq;
            L.gu_d = (const __half*) cdg;
            L.dn_d = (const __half*) cdd;
        } else {
            throw std::runtime_error(b + "ffn_gate_up_exps: type " + std::to_string(gu.type));
        }
    }

    void load_card(Card& K) {
        CK(cudaSetDevice(K.dev));
        CK(cudaStreamCreateWithFlags(&K.st, cudaStreamNonBlocking));
        const int c = K.c, h0 = c * NH_C, kv0 = c * NKV_C;
        K.tok_norm = f32(K, "token_embd_norm.weight");
        K.out_norm = f32(K, "output_norm.weight");
        K.out_rq = f32(K, "output_res_query.weight");
        K.out = q8(K, "output.weight", c * NV_C, (c + 1) * NV_C);
        const std::vector<int64_t> swa = g.ia("inkling2_loop.attention.sliding_window_pattern");
        for (int p = 0; p < NPHYS; ++p) {
            LW& L = K.L[p];
            const std::string b = "blk." + std::to_string(p) + ".";
            L.full = !swa[p];
            if (L.full) {
                L.q_norm = f32(K, b + "attn_q_inp_norm.weight");
                L.k_norm = f32(K, b + "attn_k_inp_norm.weight");
                L.v_norm = f32(K, b + "attn_v_inp_norm.weight");
            } else {
                L.attn_norm = f32(K, b + "attn_norm.weight");
            }
            if (p > 0) {
                L.attn_rq = f32(K, b + "attn_res_query.weight");
                L.nrq = (int) g.t(b + "attn_res_query.weight").ne[1];
            }
            L.ffn_rq = f32(K, b + "ffn_res_query.weight");
            {
                std::vector<std::tuple<std::string, int, int>> parts = {
                    {b + "attn_q.weight", h0 * HD, (h0 + NH_C) * HD}, {b + "attn_gate.weight", h0 * HD, (h0 + NH_C) * HD},
                    {b + "attn_rel.weight", h0 * NREL, (h0 + NH_C) * NREL}, {b + "attn_sink_proj.weight", h0, h0 + NH_C},
                    {b + "attn_k.weight", kv0 * HD, (kv0 + NKV_C) * HD}};
                if (L.full) parts.push_back({b + "attn_k_conv_proj.weight", kv0 * NCONV, (kv0 + NKV_C) * NCONV});
                parts.push_back({b + "attn_v.weight", kv0 * HD, (kv0 + NKV_C) * HD});
                if (L.full) parts.push_back({b + "attn_v_conv_proj.weight", kv0 * NCONV, (kv0 + NKV_C) * NCONV});
                std::vector<Mat> v;
                L.cat = q8cat(K, parts, v);
                L.wq = v[0];
                L.wgate = v[1];
                L.wrel = v[2];
                L.wsink = v[3];
                L.wk = v[4];
                if (L.full) {
                    L.wkd = v[5];
                    L.wv = v[6];
                    L.wvd = v[7];
                } else {
                    L.wv = v[5];
                }
            }
            L.wo = q8(K, b + "attn_output.weight", 0, N, h0 * HD, (h0 + NH_C) * HD);
            L.q_clip = f32(K, b + "attn_q_clip.weight");
            L.rel_clip = f32(K, b + "attn_rel_clip.weight");
            L.rel_bias = f32(K, b + "attn_rel_bias.weight");
            {
                float* bp = dev(K, (size_t) (EXT + 1) * NREL);
                CK(cudaMemcpy(bp, L.rel_bias, (size_t) EXT * NREL * 4, cudaMemcpyDeviceToDevice));
                L.rel_bias_pad = bp;
            }
            L.k_conv = f32(K, b + "attn_k_conv.weight", (size_t) kv0 * HD * NCONV, (size_t) KVC * NCONV);
            L.v_conv = f32(K, b + "attn_v_conv.weight", (size_t) kv0 * HD * NCONV, (size_t) KVC * NCONV);
            L.a_pconv = f32(K, b + "attn_post_conv.weight");
            L.f_pconv = f32(K, b + "ffn_post_conv.weight");
            L.ffn_norm = f32(K, b + "ffn_norm.weight");
            L.gate_inp = f32(K, b + "ffn_gate_inp.weight");
            L.probs_b = f32(K, b + "exp_probs_b.bias");
            experts(K, L, p);
            if (p >= LS && p < LE) {
                L.msg_q = f32(K, b + "loop_msg_query.weight");
                const char* nm[4] = {"q", "k", "v", "ffn"};
                for (int j = 0; j < 4; ++j) {
                    const std::string lb = b + "loop_" + nm[j] + "_ctrl_";
                    L.c_down[j] = f32(K, lb + "down.weight");
                    L.c_gain[j] = f32(K, lb + "gain.weight");
                    L.c_forget[j] = f32(K, lb + "forget.weight");
                }
            }
        }
        // caches and states
        for (int s = 0; s < NSLOT; ++s) {
            const int p = s < NPHYS ? s : LS + (s - NPHYS) % NCORE;
            const size_t rows = K.L[p].full ? (size_t) ctx : (size_t) RING;
            K.kc[s] = (__half*) alloc(K, rows * KVC * 2);
            K.vc[s] = (__half*) alloc(K, rows * KVC * 2);
            if (s == 0) {   // p5a: all slots' convolution states in one buffer (a draft window's snapshot is one copy)
                K.stbuf = dev(K, (size_t) NSLOT * SLOTST);
                K.stbak = dev(K, (size_t) NSLOT * SLOTST);
                K.stsnap = dev(K, (size_t) NSLOT * SLOTST);
                K.hist = dev(K, (size_t) NSLOT * HISTS);
            }
            K.sk[s] = K.stbuf + (size_t) s * SLOTST;
            K.sv[s] = K.sk[s] + 3 * KVC;
            K.sa[s] = K.sk[s] + 6 * KVC;
            K.sf[s] = K.sa[s] + 3 * N;
        }
        const size_t TN = (size_t) MAXT * N;
        K.x0 = dev(K, TN);
        K.cur = dev(K, TN);
        for (int i = 0; i < MAXB; ++i) K.B[i] = dev(K, TN);
        K.prel = dev(K, TN);
        K.bankS = dev(K, TN);
        K.bankO = dev(K, TN);
        K.X3 = dev(K, 4 * TN);
        K.MS = dev(K, 4 * TN);
        for (int i = 0; i < 4; ++i) K.XC[i] = dev(K, TN);
        K.xm = dev(K, 2 * TN);
        K.g64 = dev(K, (size_t) MAXT * RANK);
        K.gp = dev(K, TN);
        K.fpp = dev(K, TN);
        K.NQ = dev(K, TN);
        K.NK = dev(K, TN);
        K.NV = dev(K, TN);
        K.q = dev(K, (size_t) MAXT * QC);
        K.k = dev(K, (size_t) MAXT * KVC);
        K.v = dev(K, (size_t) MAXT * KVC);
        K.kcv = dev(K, (size_t) MAXT * KVC);
        K.vcv = dev(K, (size_t) MAXT * KVC);
        K.r = dev(K, (size_t) MAXT * NH_C * NREL);
        K.sink = dev(K, (size_t) MAXT * NH_C);
        K.gate = dev(K, (size_t) MAXT * QC);
        K.dk = dev(K, (size_t) MAXT * NKV_C * NCONV);
        K.dv = dev(K, (size_t) MAXT * NKV_C * NCONV);
        K.ao = dev(K, (size_t) MAXT * QC);
        K.nparts_max = (ctx + KC - 1) / KC + 1;
        K.part = dev(K, (size_t) MAXT * NH_C * 2 * K.nparts_max * (HD + 2));   // p4f: parts of 64 keys at T = 1
        K.ab = dev(K, TN);
        K.A = dev(K, TN);
        K.F = dev(K, TN);
        K.F2 = dev(K, TN);
        K.FN = dev(K, TN);
        K.lg = dev(K, (size_t) MAXT * NE);
        K.topw = dev(K, (size_t) MAXT * TOPK);
        K.topi = (int*) alloc(K, (size_t) MAXT * TOPK * 4);
        K.jobs = (EJob*) alloc(K, sizeof(EJob) * MAXT * TOPK);
        K.perm = (int*) alloc(K, sizeof(int) * MAXT * TOPK);
        K.mjobs = (MJob*) alloc(K, sizeof(MJob) * MAXT * TOPK);
        K.nmjobs = (int*) alloc(K, 4);
        K.GU = dev(K, (size_t) MAXT * TOPK * 2 * FF_C);
        K.HH = dev(K, (size_t) MAXT * TOPK * FF_C);
        K.EY = dev(K, (size_t) MAXT * TOPK * N);
        K.MO = dev(K, TN);
        K.hx = dev(K, TN);
        K.logits = dev(K, (size_t) MAXT * NV_C);
        K.tau = dev(K, MAXT);
        K.pos = (int*) alloc(K, MAXT * 4);
        K.dtok = (unsigned*) alloc(K, 4);
        CK(cudaMemset(K.dtok, 0, 4));
    }

    Engine(const std::string& path, int ctx_, const std::string& mtp = "") : g(path), ctx(ctx_) {
        ds_ring = getenv("INK2_DS_RING") ? atoi(getenv("INK2_DS_RING")) != 0 : ctx > 65536;   // p14
        if (!getenv("INK2_PF_QSF")) {   // p14: the full layers' score matrix 3 x QSF x (ctx + QSF) within the budget
            const double mb = getenv("INK2_PF_SMB") ? atof(getenv("INK2_PF_SMB")) : (ctx <= 98304 ? 1100.0 : ctx <= 196608 ? 450.0 : 220.0);
            while (pf_qs_full > 64 && 3.0 * pf_qs_full * (ctx + pf_qs_full) * 4 > mb * 1048576.0) pf_qs_full /= 2;
        }
        const std::string a = "inkling2_loop.";
        if (g.s("general.architecture") != "inkling2_loop" || g.i(a + "block_count") != NPHYS || g.i(a + "embedding_length") != N ||
            g.i(a + "expert_count") != NE || g.i(a + "expert_used_count") != TOPK || g.i(a + "expert_feed_forward_length") != FF ||
            g.i(a + "loop.start") != LS || g.i(a + "loop.end") != LE || g.i(a + "loop.count") != NPASS || g.i(a + "vocab_size") != NV ||
            g.i(a + "vocab_size_unpadded") != NVU || g.i(a + "attention.sliding_window") != SWA ||
            g.i(a + "attention.relative_extent") != EXT || g.i(a + "attention.relative_length") != NREL ||
            g.i(a + "attn_res.block_size") != BLK || g.i(a + "shortconv.l_cache") != NCONV || g.i(a + "loop.gate_rank") != RANK)
            throw std::runtime_error("geometry differs from v1's constants");
        eps = (float) g.f(a + "attention.layer_norm_rms_epsilon");
        qk_clip = (float) g.f(a + "attention.qk_clip");
        rel_clip_thr = (float) g.f(a + "attention.relative_clip");
        log_alpha = (float) g.f(a + "attention.log_scale_alpha");
        log_floor = (float) g.i(a + "attention.log_scale_floor");
        situ_b = (float) g.f(a + "activation.situ_beta");
        situ_lb = (float) g.f(a + "activation.situ_linear_beta");
        wscale = (float) g.f(a + "expert_weights_scale");
        softcap = (float) g.f(a + "final_logit_softcapping");
        tok = &g.t("token_embd.weight");
        if (tok->type != G_Q8_0) throw std::runtime_error("token_embd is not Q8_0");
        int ndev = 0;
        CK(cudaGetDeviceCount(&ndev));
        if (ndev < NC) throw std::runtime_error("needs two cards");
        for (int c = 0; c < NC; ++c) {
            cards[c].dev = c;
            cards[c].c = c;
        }
        const double t0 = now_s();
        std::thread t1([&]() { load_card(cards[1]); });
        load_card(cards[0]);
        t1.join();
        for (int c = 0; c < NC; ++c) {
            for (int p = 0; p < 2; ++p) CK(cudaHostAlloc((void**) &X.buf[c][p], (size_t) MAXT * N * 4, cudaHostAllocMapped | cudaHostAllocPortable));
            unsigned* f = nullptr;
            CK(cudaHostAlloc((void**) &f, 64, cudaHostAllocMapped | cudaHostAllocPortable));
            *f = 0;
            X.flag[c] = f;
        }
        CK(cudaHostAlloc((void**) &h_logits, (size_t) MAXT * NV * 4, cudaHostAllocPortable));
        if (getenv("INK2_PF_CHUNK")) pfc = std::max(MAXT, atoi(getenv("INK2_PF_CHUNK")));
        if (getenv("INK2_PF_MIN")) pf_min = std::max(1, atoi(getenv("INK2_PF_MIN")));
        if (getenv("INK2_PF_OFF")) pf_min = 1 << 30;
        if (!ilv && (pf_deq || stare || rows2 || getenv("INK2_ROWS3")))
            throw std::runtime_error("INK2_PF_DEQ / INK2_STARE / INK2_ROWS2 / INK2_ROWS3 read interleaved experts: add INK2_FP4_ILV=1");
        CK(cudaHostAlloc((void**) &h_x0, (size_t) std::max(MAXT, pfc) * N * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &h_pos, (size_t) std::max(MAXT, pfc) * 4, cudaHostAllocPortable));
        CK(cudaHostAlloc((void**) &h_tau, (size_t) std::max(MAXT, pfc) * 4, cudaHostAllocPortable));
        for (int c = 0; c < NC; ++c) pf_rec[c] = 0;
        if (pf_min < (1 << 30)) {
            std::thread t1([&]() { alloc_pf(cards[1]); });
            alloc_pf(cards[0]);
            t1.join();
        }
        if (!mtp.empty()) {   // p6
            dsg = new GgufMin(mtp);
            if (dsg->s("general.architecture") != "dflash" || dsg->i("dflash.block_size") != DBLK || dsg->i("dflash.block_count") != DSL)
                throw std::runtime_error("the drafter is not Inkling 2's DSpark");
            if (!g_draft_vocab.empty()) {   // p13: card c takes the c-th half of the (sorted) draft vocabulary
                FILE* f = std::fopen(g_draft_vocab.c_str(), "rb");
                int32_t n = 0;
                if (!f || std::fread(&n, 4, 1, f) != 1 || n <= 1) throw std::runtime_error("draft vocab: " + g_draft_vocab);
                std::vector<int> ids(n);
                if (std::fread(ids.data(), 4, n, f) != (size_t) n) throw std::runtime_error("draft vocab: " + g_draft_vocab);
                std::fclose(f);
                std::sort(ids.begin(), ids.end());
                for (int c = 0; c < NC; ++c) dvmap[c].assign(ids.begin() + (size_t) c * n / NC, ids.begin() + (size_t) (c + 1) * n / NC);
            }
            std::thread t1([&]() { ds_load(cards[1]); });
            ds_load(cards[0]);
            t1.join();
            if (!g_draft_vocab.empty()) {
                auto pick = [&](int c) {   // own threads: both() needs card 1's worker, which starts after the constructor
                    CK(cudaSetDevice(cards[c].dev));
                    cards[c].D.dvout = q8_pick(cards[c], "output.weight", dvmap[c]);
                };
                std::thread tp(pick, 1);
                pick(0);
                tp.join();
                std::fprintf(stderr, "ink2_v1: slovar risarja %zu + %zu zetonov (%s)\n", dvmap[0].size(), dvmap[1].size(), g_draft_vocab.c_str());
            }
            CK(cudaHostAlloc((void**) &h_amv_pinned, NC * 4, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &h_ams_pinned, NC * 4, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &h_ami_pinned, NC * 4, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &h_prev_pinned, NC * 4, cudaHostAllocPortable));
            ds_on = true;
        }
        std::fprintf(stderr, "ink2_v1: naloženo v %.1f s, VRAM %.2f / %.2f GiB, ctx %d, kos branja %d\n", now_s() - t0,
                     cards[0].vram / 1073741824.0, cards[1].vram / 1073741824.0, ctx, pf_min < (1 << 30) ? pfc : 0);
        w1 = std::thread([this]() {
            CK(cudaSetDevice(cards[1].dev));
            int seen = 0;
            for (;;) {
                while (go.load(std::memory_order_acquire) == seen) std::this_thread::yield();
                seen = go.load();
                if (quit) return;
                job(1);
                done.fetch_add(1, std::memory_order_release);
            }
        });
    }
    ~Engine() {
        quit = true;
        go.fetch_add(1);
        if (w1.joinable()) w1.join();
    }
    void both(std::function<void(int)> f) {   // f(0) here and f(1) on card 1's thread, at the same time
        job = f;
        const int d = done.load();
        go.fetch_add(1, std::memory_order_release);
        CK(cudaSetDevice(cards[0].dev));
        f(0);
        while (done.load(std::memory_order_acquire) == d) std::this_thread::yield();
    }

    // ------------------------------------------------------------------------------------------------ the card's program
    void gemv8(Card& K, const Mat& m, const float* x, int ldx, float* y, int ldy, int T) {
        k_gemv8<<<(m.rows + GW_BR - 1) / GW_BR, 256, (size_t) T * 512 * 4, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, ldx, y, ldy, T);
    }
    // p4a: T = 1, the rows of a (view of a) Q8S matrix to the segments' outputs
    void gemv8_t1(Card& K, const Mat& m, const float* x, const Segs& sg) {
        const int blocks = (m.rows + G1_BR - 1) / G1_BR, nch = (m.cols + 511) / 512;
        const int S = blocks >= 112 ? 1 : std::min(nch, (112 + blocks - 1) / blocks);
        k_gemv8_t1<<<dim3(blocks, S), 256, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, sg, S);
    }
    // p6c: T rows with k_gemv8_t1's sums (the same parts S as at T = 1)
    template <int TT>
    void t1tt(Card& K, const Mat& m, const float* x, int ldx, const SegsT& sg, dim3 g, int S) {
        k_gemv8_t1TT<TT><<<g, 256, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, ldx, sg, S);
    }
    void gemv8_t1T(Card& K, const Mat& m, const float* x, int ldx, const SegsT& sg, int T) {
        const int blocks = (m.rows + G1_BR - 1) / G1_BR, nch = (m.cols + 511) / 512;
        const int S = blocks >= 112 ? 1 : std::min(nch, (112 + blocks - 1) / blocks);
        const dim3 g(blocks, S);
        if (t1tt_on) switch (T) {   // p9
            case 2: return t1tt<2>(K, m, x, ldx, sg, g, S);
            case 3: return t1tt<3>(K, m, x, ldx, sg, g, S);
            case 4: return t1tt<4>(K, m, x, ldx, sg, g, S);
            case 5: return t1tt<5>(K, m, x, ldx, sg, g, S);
            case 6: return t1tt<6>(K, m, x, ldx, sg, g, S);
            case 7: return t1tt<7>(K, m, x, ldx, sg, g, S);
            case 8: return t1tt<8>(K, m, x, ldx, sg, g, S);
            case 9: return t1tt<9>(K, m, x, ldx, sg, g, S);
            case 10: return t1tt<10>(K, m, x, ldx, sg, g, S);
            case 11: return t1tt<11>(K, m, x, ldx, sg, g, S);
            case 12: return t1tt<12>(K, m, x, ldx, sg, g, S);
            case 13: return t1tt<13>(K, m, x, ldx, sg, g, S);
            case 14: return t1tt<14>(K, m, x, ldx, sg, g, S);
            case 15: return t1tt<15>(K, m, x, ldx, sg, g, S);
            case 16: return t1tt<16>(K, m, x, ldx, sg, g, S);
            default: break;
        }
        k_gemv8_t1T<<<g, 256, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, ldx, sg, S, T);
    }
    // p9: the drafter's matrices (DBLK = 8 rows, ingest windows) by the window GEMV instead of k_gemv8 (~85-186 GB/s)
    void dsgemv(Card& K, const Mat& m, const float* x, int ldx, float* y, int ldy, int T) {
        if (ds_t1 && T > 1 && T <= MAXT) gemv8_t1T(K, m, x, ldx, segs1(ldy, y), T);
        else gemv8(K, m, x, ldx, y, ldy, T);
    }
    static SegsT segs1(int ld, float* d) {
        SegsT sg{};
        sg.n = 1;
        sg.r0[0] = 0;
        sg.ld[0] = ld;
        sg.dst[0] = d;
        return sg;
    }
    static Segs segs(std::initializer_list<std::pair<int, float*>> l) {   // (rows, output) one after another
        Segs sg{};
        int r = 0;
        for (auto& [n, d] : l) {
            sg.r0[sg.n] = r;
            sg.dst[sg.n++] = d;
            r += n;
        }
        return sg;
    }
    static Mat rows_of(const Mat& m, int r0, int n) {
        Mat v = m;
        v.rows = n;
        v.q = m.q + (size_t) r0 * m.cols;
        v.d = (const __half*) m.d + (size_t) r0 * (m.cols / 32);
        return v;
    }
    void gemv32(Card& K, const float* W, int rows, int cols, const float* x, int ldx, float* y, int ldy, int T) {
        k_gemv32<<<(rows + GW_BR - 1) / GW_BR, 256, (size_t) T * 128 * 4, K.st>>>(W, rows, cols, x, ldx, y, ldy, T);
    }
    void rms(Card& K, const float* x, const float* w, float* y, int T, float e) { k_rms<<<T, 256, 0, K.st>>>(x, N, w, y, N, N, e); }
    void copy(Card& K, float* dst, const float* src, int T) { CK(cudaMemcpyAsync(dst, src, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, K.st)); }
    void xch(Card& K, float* y, int n) {   // p4c: sequence numbers from the card's forward counter (dtok)
        const unsigned i = K.xidx++;
        const int par = i & 1, c = K.c;
        k_xput2<<<1, 1024, 0, K.st>>>(y, n, X.buf[c][par], X.flag[c], K.dtok, i);
        k_xget2<<<1, 1024, 0, K.st>>>(y, n, X.buf[1 - c][par], X.flag[1 - c], K.dtok, i, c);
    }
    void control(Card& K, const LW& L, int j, const float* x, const float* m, float* out, int T) {
        if (fuse && !stare && !dumping) {   // p4e: the rms of x and m inside the down projection (p6b: T rows)
            if (T == 1) k_ctrl_down1<<<RANK, 256, 0, K.st>>>(L.c_down[j], x, m, K.g64);
            else k_ctrl_downT<<<dim3(RANK, T), 256, 0, K.st>>>(L.c_down[j], x, m, K.g64);
            k_ctrl_gf<<<N / 8, 256, 0, K.st>>>(L.c_gain[j], L.c_forget[j], K.g64, x, m, N, out, T);
            return;
        }
        k_ctrl_in<<<T, 256, 0, K.st>>>(x, m, N, K.xm);
        if (!stare) {
            k_gemv32_rows<<<RANK, 256, 0, K.st>>>(L.c_down[j], 2 * N, K.xm, 2 * N, K.g64, RANK, T);
            k_ctrl_gf<<<N / 8, 256, 0, K.st>>>(L.c_gain[j], L.c_forget[j], K.g64, x, m, N, out, T);
            return;
        }
        gemv32(K, L.c_down[j], RANK, 2 * N, K.xm, 2 * N, K.g64, RANK, T);
        gemv32(K, L.c_gain[j], N, RANK, K.g64, RANK, K.gp, N, T);
        gemv32(K, L.c_forget[j], N, RANK, K.g64, RANK, K.fpp, N, T);
        k_ctrl_out<<<(T * N + 255) / 256, 256, 0, K.st>>>(x, m, N, K.gp, K.fpp, out, T);
    }
    void mix(Card& K, const States& S, const float* q, int nq, float* out, int T) {
        if (stare) k_mix<<<T, 256, 0, K.st>>>(S, q, nq, out);
        else k_mix2<<<T, 768, 0, K.st>>>(S, q, nq, out);
    }
    States states(Card& K, const float* last) {
        States S{};
        for (int i = 0; i < K.nb; ++i) S.p[i] = K.B[i];
        S.p[K.nb] = last;
        S.n = K.nb + 1;
        return S;
    }
    void dump_rows(std::vector<float>& v, const float* d, int T) {
        const size_t o = v.size();
        v.resize(o + (size_t) T * N);
        CK(cudaMemcpy(v.data() + o, d, (size_t) T * N * 4, cudaMemcpyDeviceToHost));
    }

    void layer(Card& K, int p, int slot, bool loop, int T, int nparts) {
        const LW& L = K.L[p];
        const int c = K.c, T_N = T * N;
        const float* xin[3] = {K.cur, K.cur, K.cur};
        if (L.attn_rq) {
            mix(K, states(K, K.cur), L.attn_rq, L.nrq, K.X3, T);
            for (int j = 0; j < 3; ++j) xin[j] = K.X3 + (size_t) (L.full ? j : 0) * T_N;
        }
        if (loop) {
            States S{};
            S.p[0] = K.bankS;
            S.p[1] = K.bankO;
            S.n = 2;
            mix(K, S, L.msg_q, 4, K.MS, T);
            for (int j = 0; j < 3; ++j) {
                control(K, L, j, xin[j], K.MS + (size_t) j * T_N, K.XC[j], T);
                xin[j] = K.XC[j];
            }
        }
        const float *nq, *nk, *nv;
        if (!L.full && !loop) {
            rms(K, xin[0], L.attn_norm, K.NQ, T, eps);
            nq = nk = nv = K.NQ;
        } else {
            rms(K, xin[0], L.full ? L.q_norm : L.attn_norm, K.NQ, T, eps);
            rms(K, xin[1], L.full ? L.k_norm : L.attn_norm, K.NK, T, eps);
            rms(K, xin[2], L.full ? L.v_norm : L.attn_norm, K.NV, T, eps);
            nq = K.NQ;
            nk = K.NK;
            nv = K.NV;
        }
        const bool fzT = fuse && !stare && !dumping;   // p6b: the fused path for any T <= MAXT
        if (T > 1 && fzT) {   // p6b: the matrices of one input in one launch, T rows
            const int nqr = 2 * QC + NH_C * NREL + NH_C, nkr = KVC + (L.full ? NKV_C * NCONV : 0);
            auto st = [&](std::initializer_list<std::pair<int, float*>> l) {
                SegsT sg{};
                int r = 0;
                for (auto& [n, d] : l) {
                    sg.r0[sg.n] = r;
                    sg.ld[sg.n] = n;
                    sg.dst[sg.n++] = d;
                    r += n;
                }
                return sg;
            };
            auto run = [&](const Mat& m, const float* x, const SegsT& sg) {   // p6c: the bits of T = 1 (INK2_WIN_SEG: k_gemv8_seg, ~4 % faster)
                if (win_seg)
                    k_gemv8_seg<<<(m.rows + GW_BR - 1) / GW_BR, 256, (size_t) T * 512 * 4, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, x, N,
                                                                                                  sg, T);
                else
                    gemv8_t1T(K, m, x, N, sg, T);
            };
            const SegsT sq = st({{QC, K.q}, {QC, K.gate}, {NH_C * NREL, K.r}, {NH_C, K.sink}});
            const SegsT sk = L.full ? st({{KVC, K.k}, {NKV_C * NCONV, K.dk}}) : st({{KVC, K.k}});
            const SegsT sv = L.full ? st({{KVC, K.v}, {NKV_C * NCONV, K.dv}}) : st({{KVC, K.v}});
            if (nq == nk && nk == nv) {
                SegsT all = sq;
                for (int i = 0; i < sk.n; ++i) { all.r0[all.n] = nqr + sk.r0[i]; all.ld[all.n] = sk.ld[i]; all.dst[all.n++] = sk.dst[i]; }
                for (int i = 0; i < sv.n; ++i) { all.r0[all.n] = nqr + nkr + sv.r0[i]; all.ld[all.n] = sv.ld[i]; all.dst[all.n++] = sv.dst[i]; }
                run(L.cat, nq, all);
            } else {
                run(rows_of(L.cat, 0, nqr), nq, sq);
                run(rows_of(L.cat, nqr, nkr), nk, sk);
                run(rows_of(L.cat, nqr + nkr, nkr), nv, sv);
            }
        } else if (T == 1 && !stare) {   // p4a: the matrices of one input in one launch
            const int nqr = 2 * QC + NH_C * NREL + NH_C, nkr = KVC + (L.full ? NKV_C * NCONV : 0);
            const Segs sq = segs({{QC, K.q}, {QC, K.gate}, {NH_C * NREL, K.r}, {NH_C, K.sink}});
            const Segs sk = L.full ? segs({{KVC, K.k}, {NKV_C * NCONV, K.dk}}) : segs({{KVC, K.k}});
            const Segs sv = L.full ? segs({{KVC, K.v}, {NKV_C * NCONV, K.dv}}) : segs({{KVC, K.v}});
            if (nq == nk && nk == nv) {
                Segs all = sq;
                for (int i = 0; i < sk.n; ++i) {
                    all.r0[all.n] = nqr + sk.r0[i];
                    all.dst[all.n++] = sk.dst[i];
                }
                for (int i = 0; i < sv.n; ++i) {
                    all.r0[all.n] = nqr + nkr + sv.r0[i];
                    all.dst[all.n++] = sv.dst[i];
                }
                gemv8_t1(K, L.cat, nq, all);
            } else {
                gemv8_t1(K, rows_of(L.cat, 0, nqr), nq, sq);
                gemv8_t1(K, rows_of(L.cat, nqr, nkr), nk, sk);
                gemv8_t1(K, rows_of(L.cat, nqr + nkr, nkr), nv, sv);
            }
        } else {
            gemv8(K, L.wq, nq, N, K.q, QC, T);
            gemv8(K, L.wk, nk, N, K.k, KVC, T);
            gemv8(K, L.wv, nv, N, K.v, KVC, T);
            gemv8(K, L.wrel, nq, N, K.r, NH_C * NREL, T);
            gemv8(K, L.wsink, nq, N, K.sink, NH_C, T);
            gemv8(K, L.wgate, nq, N, K.gate, QC, T);
            if (L.full) {
                gemv8(K, L.wkd, nk, N, K.dk, NKV_C * NCONV, T);
                gemv8(K, L.wvd, nv, N, K.dv, NKV_C * NCONV, T);
            }
        }
        const float kconv = 1.0f / sqrtf((float) ((L.full ? 2 : 1) * NCONV + 1));
        const bool fz = fzT;   // p4e (p6b: any T)
        float* hk = K.keep_hist ? K.hist + (size_t) slot * HISTS : nullptr;   // p5a: the window's inputs for the rollback
        if (fz && T > 1) {
            k_post_qkvT<<<dim3(2 * NH_C + 2 * NKV_C, T), 128, 0, K.st>>>(
                K.q, K.r, K.k, K.v, L.full ? K.dk : nullptr, L.full ? K.dv : nullptr, K.sk[slot], K.sv[slot], L.k_conv, L.v_conv, L.q_clip,
                L.rel_clip, c * NH_C, qk_clip, rel_clip_thr, L.full ? K.tau : nullptr, eps, kconv, K.pos, !L.full, K.kc[slot], K.vc[slot], hk,
                hk ? hk + (size_t) MAXT * KVC : nullptr);
            k_post_state<<<(2 * KVC + 255) / 256, 256, 0, K.st>>>(K.k, K.v, K.sk[slot], K.sv[slot], T);
        } else if (fz) {
            k_post_qkv<<<2 * NH_C + 2 * NKV_C, 128, 0, K.st>>>(K.q, K.r, K.k, K.v, L.full ? K.dk : nullptr, L.full ? K.dv : nullptr, K.sk[slot],
                                                               K.sv[slot], L.k_conv, L.v_conv, L.q_clip, L.rel_clip, c * NH_C, qk_clip,
                                                               rel_clip_thr, L.full ? K.tau : nullptr, eps, kconv, K.pos, !L.full,
                                                               K.kc[slot], K.vc[slot]);
        } else {
        if (K.keep_hist)
            CK(cudaMemcpyAsync(K.hist + (size_t) slot * HISTS + 0, K.k, (size_t) T * KVC * 4, cudaMemcpyDeviceToDevice, K.st));
        if (K.keep_hist)
            CK(cudaMemcpyAsync(K.hist + (size_t) slot * HISTS + (size_t) MAXT * KVC, K.v, (size_t) T * KVC * 4, cudaMemcpyDeviceToDevice, K.st));
        k_conv<<<(KVC + 127) / 128, 128, 0, K.st>>>(K.k, KVC, K.sk[slot], L.k_conv, L.full ? K.dk : nullptr, HD, kconv, K.kcv, T);
        k_conv<<<(KVC + 127) / 128, 128, 0, K.st>>>(K.v, KVC, K.sv[slot], L.v_conv, L.full ? K.dv : nullptr, HD, kconv, K.vcv, T);
        const float* tau = L.full ? K.tau : nullptr;
        k_headnorm<<<(T * NH_C + 7) / 8, 256, 0, K.st>>>(K.q, NH_C, HD, L.q_clip, c * NH_C, qk_clip, tau, 0, eps, T);
        k_headnorm<<<(T * NKV_C + 7) / 8, 256, 0, K.st>>>(K.kcv, NKV_C, HD, nullptr, 0, qk_clip, nullptr, 0, eps, T);
        k_headnorm<<<(T * NKV_C + 7) / 8, 256, 0, K.st>>>(K.vcv, NKV_C, HD, nullptr, 0, 0.0f, nullptr, 1, eps, T);
        k_headnorm<<<(T * NH_C + 7) / 8, 256, 0, K.st>>>(K.r, NH_C, NREL, L.rel_clip, c * NH_C, rel_clip_thr, tau, 0, eps, T);
        k_kv_store<<<(T * KVC + 255) / 256, 256, 0, K.st>>>(K.kcv, K.vcv, K.pos, !L.full, K.kc[slot], K.vc[slot], T);
        }
        if (!attn_old && !stare) {   // p4f: by KV head, 64 keys a block, the 3 query heads together (p5a: any T, grid z)
            const int np = L.full ? 2 * nparts : std::min(2 * nparts, (SWA + MAXT + KC2 - 1) / KC2 + 1);
            k_attn_dec<<<dim3(np, NKV_C, T), 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, !L.full, K.part, np);
            k_attn_comb<<<dim3(NH_C, T), 128, 0, K.st>>>(K.part, np, K.gate, K.ao);
        } else {
            const int np = L.full ? nparts : std::min(nparts, (SWA + KC - 1) / KC + 1);
            k_attn_part<<<dim3(np, NH_C, T), 128, 0, K.st>>>(K.q, K.r, L.rel_bias, K.sink, K.kc[slot], K.vc[slot], K.pos, !L.full, K.part, np);
            k_attn_comb<<<dim3(NH_C, T), 128, 0, K.st>>>(K.part, np, K.gate, K.ao);
        }
        if (T == 1 && !stare) gemv8_t1(K, L.wo, K.ao, segs({{N, K.ab}}));
        else if (fzT) gemv8_t1T(K, L.wo, K.ao, QC, segs1(N, K.ab), T);   // p6c
        else gemv8(K, L.wo, K.ao, QC, K.ab, N, T);
        const int ip = slot < NPHYS ? slot : LS + (slot - NPHYS) % NCORE;
        if (fz) {   // p4e: the exchange, the post-convolution and the residual (or the block push) in one block
            const unsigned xi = K.xidx++;
            const int par = xi & 1;
            if (T > 1)
                k_xchg_convT<<<1, 1024, 0, K.st>>>(K.ab, nullptr, nullptr, X.buf[c][par], X.flag[c], X.buf[1 - c][par], X.flag[1 - c], K.dtok, xi,
                                                   c, K.sa[slot], L.a_pconv, 1.0f / sqrtf(NCONV + 1.0f), K.cur,
                                                   ip % BLK == 0 ? K.B[K.nb++] : nullptr, hk ? hk + (size_t) 2 * MAXT * KVC : nullptr, T, K.F2);
            else
            k_xchg_conv<<<1, 1024, 0, K.st>>>(K.ab, nullptr, nullptr, X.buf[c][par], X.flag[c], X.buf[1 - c][par], X.flag[1 - c], K.dtok, xi, c,
                                              K.sa[slot], L.a_pconv, 1.0f / sqrtf(NCONV + 1.0f), K.cur,
                                              ip % BLK == 0 ? K.B[K.nb++] : nullptr);
        } else {
        xch(K, K.ab, T_N);
        if (K.keep_hist)
            CK(cudaMemcpyAsync(K.hist + (size_t) slot * HISTS + (size_t) 2 * MAXT * KVC, K.ab, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, K.st));
        k_conv<<<(N + 127) / 128, 128, 0, K.st>>>(K.ab, N, K.sa[slot], L.a_pconv, nullptr, 1, 1.0f / sqrtf(NCONV + 1.0f), K.A, T);
        if (ip % BLK == 0) {
            copy(K, K.B[K.nb++], K.cur, T);
            copy(K, K.cur, K.A, T);
        } else {
            k_add<<<(T_N + 255) / 256, 256, 0, K.st>>>(K.cur, K.cur, K.A, T_N);
        }
        }
        // MoE
        mix(K, states(K, K.cur), L.ffn_rq, 1, K.F, T);
        const float* f = K.F;
        if (loop) {
            control(K, L, 3, K.F, K.MS + (size_t) 3 * T_N, K.F2, T);
            f = K.F2;
        }
        rms(K, f, L.ffn_norm, K.FN, T, eps);
        if (stare) gemv32(K, L.gate_inp, NE, N, K.FN, N, K.lg, NE, T);
        else k_gemv32_rows<<<NE, 256, 0, K.st>>>(L.gate_inp, N, K.FN, N, K.lg, NE, T);
        if (fz) {
            k_route_jobs<<<T, NE, 0, K.st>>>(K.lg, L.probs_b, wscale, K.topi, K.topw, K.jobs);
        } else {
            k_route<<<T, NE, 0, K.st>>>(K.lg, L.probs_b, wscale, K.topi, K.topw);
            k_jobs<<<1, MAXT * TOPK, 0, K.st>>>(K.topi, K.jobs, T);
        }
        const int nj = T * TOPK;
        const int* perm = nullptr;
        if (T > 1 && fz && perm_on) {   // p11: a window's jobs in the kernels' block order by expert
            k_job_perm<<<1, MAXT * TOPK, 0, K.st>>>(K.jobs, nj, K.perm);
            perm = K.perm;
        }
        if (L.fp4 && !ilv && T > 1 && !mt_off) {   // p5b: a window: each picked expert once for all its picks
            k_mt_jobs<<<1, 128, 0, K.st>>>(K.topi, nj, K.mjobs, K.nmjobs);
            k_fp4_mt<36, 4, 256><<<dim3(2 * FF_C / 64, nj), 256, 0, K.st>>>(L.gu, (size_t) 2 * FF_C * (N / 64) * 36, 2 * FF_C, K.FN, N, 0, L.gu_s,
                                                                         K.mjobs, K.nmjobs, K.GU, 2 * FF_C);
            k_situ<<<(nj * FF_C + 255) / 256, 256, 0, K.st>>>(K.GU, K.HH, nj, situ_b, situ_lb);
            k_fp4_mt<9, 1, 128><<<dim3(N / 128, nj), 128, 0, K.st>>>(L.dn, (size_t) N * (FF_C / 64) * 36, N, K.HH, FF_C, 1, L.dn_s, K.mjobs,
                                                                  K.nmjobs, K.EY, N);
        } else if (L.fp4 && !ilv) {   // p4d: planar experts, gate_up by rows of 4 lanes, down by 288-block tiles
            k_fp4_rows7<36, 4, 256><<<dim3(2 * FF_C / 64, nj), 256, 0, K.st>>>(L.gu, (size_t) 2 * FF_C * (N / 64) * 36, 2 * FF_C, K.FN, N, 0,
                                                                            L.gu_s, K.jobs, K.GU, 2 * FF_C, perm);
            k_situ<<<(nj * FF_C + 255) / 256, 256, 0, K.st>>>(K.GU, K.HH, nj, situ_b, situ_lb);
            k_fp4_rows6<9, 32><<<dim3(N / 128, nj), 128, 0, K.st>>>(L.dn, (size_t) N * (FF_C / 64) * 36, N, K.HH, FF_C, 1, L.dn_s, K.jobs, K.EY,
                                                                 N, perm);
        } else if (L.fp4 && !stare && !rows2) {   // p4b: contiguous tiles of 288 blocks a warp (INK2_FP4_ILV)
            k_fp4_rows3<36, 8><<<dim3(2 * FF_C / 32, nj), 128, 0, K.st>>>(L.gu, (size_t) 2 * FF_C * (N / 64) * 36, 2 * FF_C, K.FN, N, 0, L.gu_s,
                                                                       K.jobs, K.GU, 2 * FF_C);
            k_situ<<<(nj * FF_C + 255) / 256, 256, 0, K.st>>>(K.GU, K.HH, nj, situ_b, situ_lb);
            k_fp4_rows3<9, 32><<<dim3(N / 128, nj), 128, 0, K.st>>>(L.dn, (size_t) N * (FF_C / 64) * 36, N, K.HH, FF_C, 1, L.dn_s, K.jobs, K.EY,
                                                                 N);
        } else if (L.fp4 && !stare) {
            k_fp4_rows2<<<dim3(2 * FF_C / 8, nj), 256, N * 4, K.st>>>(L.gu, (size_t) 2 * FF_C * (N / 64) * 36, 2 * FF_C, N, K.FN, N, 0, L.gu_s,
                                                                    K.jobs, K.GU, 2 * FF_C);
            k_situ<<<(nj * FF_C + 255) / 256, 256, 0, K.st>>>(K.GU, K.HH, nj, situ_b, situ_lb);
            k_fp4_rows2<<<dim3(N / 8, nj), 256, FF_C * 4, K.st>>>(L.dn, (size_t) N * (FF_C / 64) * 36, N, FF_C, K.HH, FF_C, 1, L.dn_s, K.jobs,
                                                               K.EY, N);
        } else if (L.fp4) {
            k_exp_rows<true><<<dim3(2 * FF_C / 8, nj), 256, N * 4, K.st>>>(L.gu, (size_t) 2 * FF_C * (N / 64) * 36, nullptr, 0, 2 * FF_C, N, K.FN, N,
                                                                         0, L.gu_s, K.jobs, K.GU, 2 * FF_C);
            k_situ<<<(nj * FF_C + 255) / 256, 256, 0, K.st>>>(K.GU, K.HH, nj, situ_b, situ_lb);
            k_exp_rows<true><<<dim3(N / 8, nj), 256, FF_C * 4, K.st>>>(L.dn, (size_t) N * (FF_C / 64) * 36, nullptr, 0, N, FF_C, K.HH, FF_C, 1,
                                                                    L.dn_s, K.jobs, K.EY, N);
        } else {
            k_exp_rows<false><<<dim3(2 * FF_C / 8, nj), 256, N * 4, K.st>>>(L.gu, (size_t) 2 * FF_C * N, L.gu_d, (size_t) 2 * FF_C * (N / 32),
                                                                          2 * FF_C, N, K.FN, N, 0, nullptr, K.jobs, K.GU, 2 * FF_C, perm);
            k_situ<<<(nj * FF_C + 255) / 256, 256, 0, K.st>>>(K.GU, K.HH, nj, situ_b, situ_lb);
            k_exp_rows<false><<<dim3(N / 8, nj), 256, FF_C * 4, K.st>>>(L.dn, (size_t) N * FF_C, L.dn_d, (size_t) N * (FF_C / 32), N, FF_C,
                                                                     K.HH, FF_C, 1, nullptr, K.jobs, K.EY, N, perm);
        }
        if (fz) {   // p4e: the experts' weighted sum, the exchange, the post-convolution and the residual in one block
            const unsigned xi = K.xidx++;
            const int par = xi & 1;
            if (T > 1) k_moe_sum<<<(T_N + 255) / 256, 256, 0, K.st>>>(K.EY, K.topw, K.MO, T);   // p6e: the experts' sum in parallel
            if (T > 1)
                k_xchg_convT<<<1, 1024, 0, K.st>>>(K.MO, nullptr, nullptr, X.buf[c][par], X.flag[c], X.buf[1 - c][par], X.flag[1 - c], K.dtok, xi,
                                                   c, K.sf[slot], L.f_pconv, 1.0f / sqrtf(NCONV + 1.0f), K.cur, nullptr,
                                                   hk ? hk + (size_t) 2 * MAXT * KVC + (size_t) MAXT * N : nullptr, T, K.F2);
            else
            k_xchg_conv<<<1, 1024, 0, K.st>>>(nullptr, K.EY, K.topw, X.buf[c][par], X.flag[c], X.buf[1 - c][par], X.flag[1 - c], K.dtok, xi, c,
                                              K.sf[slot], L.f_pconv, 1.0f / sqrtf(NCONV + 1.0f), K.cur, nullptr);
        } else {
            k_moe_sum<<<(T_N + 255) / 256, 256, 0, K.st>>>(K.EY, K.topw, K.MO, T);
            xch(K, K.MO, T_N);
            if (K.keep_hist)
                CK(cudaMemcpyAsync(K.hist + (size_t) slot * HISTS + (size_t) 2 * MAXT * KVC + (size_t) MAXT * N, K.MO, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, K.st));
            k_conv<<<(N + 127) / 128, 128, 0, K.st>>>(K.MO, N, K.sf[slot], L.f_pconv, nullptr, 1, 1.0f / sqrtf(NCONV + 1.0f), K.F2, T);
            k_add<<<(T_N + 255) / 256, 256, 0, K.st>>>(K.cur, K.cur, K.F2, T_N);
        }
        if (ds_on && ds_feat_slot(slot) >= 0) ds_feat(K, K.D.feat, K.cur, ds_feat_slot(slot), T);   // p6
        if (dumping && c == 0) {
            CK(cudaStreamSynchronize(K.st));
            dump_rows(d_lout[slot], K.cur, T);
            dump_rows(d_aout[slot], K.A, T);
            dump_rows(d_fout[slot], K.F2, T);
        }
    }


    // ------------------------------------------------------------------------------------------------ p6: the DSpark drafter
    const uint16_t* bf16(Card& K, const std::string& name) {
        const GTensor& t = dsg->t(name);
        if (t.type != G_BF16) throw std::runtime_error(name + " is not BF16");
        void* p = alloc(K, t.nbytes);
        CK(cudaMemcpy(p, t.data, t.nbytes, cudaMemcpyHostToDevice));
        return (const uint16_t*) p;
    }
    void ds_load(Card& K) {
        CK(cudaSetDevice(K.dev));
        DSW& D = K.D;
        GgufMin* G = dsg;
        for (int il = 0; il < DSL; ++il) {
            const std::string b = "blk." + std::to_string(il) + ".";
            D.wq[il] = q8(K, b + "attn_q.weight", 0, -1, 0, -1, G);
            D.wk[il] = q8(K, b + "attn_k.weight", 0, -1, 0, -1, G);
            D.wv[il] = q8(K, b + "attn_v.weight", 0, -1, 0, -1, G);
            D.wo[il] = q8(K, b + "attn_output.weight", 0, -1, 0, -1, G);
            D.kctx[il] = q8(K, b + "attn_k_ctx.weight", 0, -1, 0, -1, G);
            D.vctx[il] = q8(K, b + "attn_v_ctx.weight", 0, -1, 0, -1, G);
            D.gate[il] = q8(K, b + "ffn_gate.weight", 0, -1, 0, -1, G);
            D.up[il] = q8(K, b + "ffn_up.weight", 0, -1, 0, -1, G);
            D.down[il] = q8(K, b + "ffn_down.weight", 0, -1, 0, -1, G);
            D.attn_norm[il] = f32(K, b + "attn_norm.weight", 0, 0, G);
            D.ffn_norm[il] = f32(K, b + "ffn_norm.weight", 0, 0, G);
            D.dflare[il] = f32(K, b + "dflare_norm.weight", 0, 0, G);
            D.qn[il] = f32(K, b + "attn_q_norm.weight", 0, 0, G);
            D.kn[il] = f32(K, b + "attn_k_norm.weight", 0, 0, G);
            D.kcn[il] = f32(K, b + "attn_k_ctx_norm.weight", 0, 0, G);
            const size_t dr = ds_ring ? (size_t) (il == 1 ? DS_RING1 : DS_RING) : (size_t) ctx;   // p14
            D.kc[il] = (__half*) alloc(K, dr * NKVD * 2);
            D.vc[il] = (__half*) alloc(K, dr * NKVD * 2);
        }
        D.fuse = f32(K, "dspark_fuse.weight", 0, 0, G);
        D.mask = f32(K, "mask_embd.weight", 0, 0, G);
        D.dsnorm = f32(K, "dspark_norm.weight", 0, 0, G);
        D.outnorm = f32(K, "output_norm.weight", 0, 0, G);
        D.jb = f32(K, "joint_proj.bias", 0, 0, G);
        D.hb = f32(K, "dspark_hidden_out.bias", 0, 0, G);
        D.w1 = bf16(K, "markov_w1.weight");
        D.jw = bf16(K, "joint_proj.weight");
        D.hw = bf16(K, "dspark_hidden_out.weight");
        const size_t TN = (size_t) MAXT * N;
        D.feat = dev(K, 4 * TN);
        D.frnn = dev(K, TN);
        D.ctxv = dev(K, N);
        D.cbuf = dev(K, TN);
        D.kb = dev(K, (size_t) MAXT * NKVD);
        D.vb = dev(K, (size_t) MAXT * NKVD);
        D.x = dev(K, (size_t) DBLK * N);
        D.xn = dev(K, (size_t) DBLK * N);
        D.q = dev(K, (size_t) DBLK * NQ);
        D.k = dev(K, (size_t) DBLK * NKVD);
        D.v = dev(K, (size_t) DBLK * NKVD);
        D.ao = dev(K, (size_t) DBLK * NQ);
        D.o = dev(K, (size_t) DBLK * N);
        D.g = dev(K, (size_t) DBLK * DFF);
        D.u = dev(K, (size_t) DBLK * DFF);
        D.h = dev(K, (size_t) DBLK * DFF);
        D.nparts = ((ds_ring ? std::max(DS_WIN1, SWA + 1) : ctx) + DBLK + KC2 - 1) / KC2 + 1;   // p14
        D.part = dev(K, (size_t) DBLK * NH * D.nparts * (HD + 2));
        D.hid = dev(K, (size_t) DBLK * N);
        D.state = dev(K, DRANK);
        D.jin = dev(K, DJ);
        D.jout = dev(K, 3 * DRANK);
        D.lat = dev(K, DRANK);
        D.y = dev(K, N);
        D.amv = dev(K, 1);
        D.ams = dev(K, 1);
        D.ami = (int*) alloc(K, 4);
        D.prev = (int*) alloc(K, 4);
        D.bpos = (int*) alloc(K, DBLK * 4);
        if (pf_min < (1 << 30)) {
            const size_t CN = (size_t) pfc * N;
            D.pfeat = dev(K, 4 * CN);
            D.pfrnn = dev(K, CN);
            D.pcbuf = dev(K, CN);
            D.pkb = dev(K, (size_t) pfc * NKVD);
            D.pvb = dev(K, (size_t) pfc * NKVD);
        }
    }
    // the target's features of T rows (feat [T][4][N]) into the drafter's caches at pos[0..T); f_rnn of every row; ctxv = the last row's
    void ds_ingest(Card& K, float* feat, float* frnn, float* cbuf, float* kb, float* vb, const int* pos, int T, bool big) {
        DSW& D = K.D;
        for (int il = 0; il < DSL; ++il) {
            k_ds_mixfeat<<<T, 256, 0, K.st>>>(feat, D.fuse + il * 4, D.dflare[il], eps, cbuf);
            if (big) {
                PFB& P = K.P;
                deq8(K, D.kctx[il], P.Wf);
                sgemm(K, P.Wf, NKVD, N, cbuf, N, kb, NKVD, T);
                deq8(K, D.vctx[il], P.Wf);
                sgemm(K, P.Wf, NKVD, N, cbuf, N, vb, NKVD, T);
            } else {
                dsgemv(K, D.kctx[il], cbuf, N, kb, NKVD, T);
                dsgemv(K, D.vctx[il], cbuf, N, vb, NKVD, T);
            }
            const int rg = ds_ring ? (il == 1 ? DS_RING1 : DS_RING) : 0;   // p14
            k_ds_headrope<<<dim3(NKV, T), 128, 0, K.st>>>(kb, NKV, D.kcn[il], eps, pos, 0, 1e6f, D.kc[il], NKVD, rg);
            k_ds_vstore<<<(T * NKVD + 255) / 256, 256, 0, K.st>>>(vb, pos, 0, D.vc[il], T, rg);
        }
        k_ds_mixfeat<<<T, 256, 0, K.st>>>(feat, D.fuse + 3 * 4, D.dsnorm, eps, frnn);
        CK(cudaMemcpyAsync(D.ctxv, frnn + (size_t) (T - 1) * N, N * 4, cudaMemcpyDeviceToDevice, K.st));
    }
    void ds_feat(Card& K, float* feat, const float* rows, int j, int T) {   // feature j of T rows into feat [T][4][N]
        CK(cudaMemcpy2DAsync(feat + (size_t) j * N, (size_t) 4 * N * 4, rows, (size_t) N * 4, (size_t) N * 4, T, cudaMemcpyDeviceToDevice, K.st));
    }
    static int ds_feat_slot(int slot) { return slot == 4 ? 0 : slot == 28 ? 1 : slot == 46 ? 2 : -1; }
    // the drafter's block at positions P..P+7 for the anchor (the pending token): the 3 layers, then the final norm into hid (8 rows)
    void ds_block(int c, int anchor_row_ready, int P, int R = DBLK) {   // p10: R rows of the block (DBLK = as trained)
        Card& K = cards[c];
        DSW& D = K.D;
        CK(cudaSetDevice(K.dev));
        int hp[DBLK];
        for (int i = 0; i < DBLK; ++i) hp[i] = P + i;
        CK(cudaMemcpyAsync(D.bpos, hp, R * sizeof(int), cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(K.x0, h_x0, (size_t) N * 4, cudaMemcpyHostToDevice, K.st));   // the anchor's embedding row (embed(0, a))
        k_rms<<<1, 256, 0, K.st>>>(K.x0, N, K.tok_norm, D.x, N, N, eps);
        for (int i = 1; i < R; ++i) CK(cudaMemcpyAsync(D.x + (size_t) i * N, D.mask, N * 4, cudaMemcpyDeviceToDevice, K.st));
        for (int il = 0; il < DSL; ++il) {
            k_rms<<<R, 256, 0, K.st>>>(D.x, N, D.attn_norm[il], D.xn, N, N, eps);
            dsgemv(K, D.wq[il], D.xn, N, D.q, NQ, R);
            dsgemv(K, D.wk[il], D.xn, N, D.k, NKVD, R);
            dsgemv(K, D.wv[il], D.xn, N, D.v, NKVD, R);
            k_ds_headrope<<<dim3(NH, R), 128, 0, K.st>>>(D.q, NH, D.qn[il], eps, D.bpos, 0, 1e6f, nullptr, 0);
            const int rg = ds_ring ? (il == 1 ? DS_RING1 : DS_RING) : 0;   // p14
            k_ds_headrope<<<dim3(NKV, R), 128, 0, K.st>>>(D.k, NKV, D.kn[il], eps, D.bpos, 0, 1e6f, D.kc[il], NKVD, rg);
            k_ds_vstore<<<(R * NKVD + 255) / 256, 256, 0, K.st>>>(D.v, D.bpos, 0, D.vc[il], R, rg);
            const int lo = il == 1 ? (ds_ring ? std::max(0, P - DS_WIN1) : 0) : std::max(0, P - (SWA - 1) - 1), hi = P + R;   // window n_swa + 1 = 1025: keys from P - 1024
            const int np = (hi - lo + KC2 - 1) / KC2;
            k_ds_attn<<<dim3(np, NH, R), 128, 0, K.st>>>(D.q, D.kc[il], D.vc[il], lo, hi, D.part, np, rg);
            k_ds_comb<<<dim3(NH, R), 128, 0, K.st>>>(D.part, np, D.ao);
            dsgemv(K, D.wo[il], D.ao, NQ, D.o, N, R);
            k_add<<<(R * N + 255) / 256, 256, 0, K.st>>>(D.x, D.x, D.o, (size_t) R * N);
            k_rms<<<R, 256, 0, K.st>>>(D.x, N, D.ffn_norm[il], D.xn, N, N, eps);
            dsgemv(K, D.gate[il], D.xn, N, D.g, DFF, R);
            dsgemv(K, D.up[il], D.xn, N, D.u, DFF, R);
            k_ds_swiglu<<<(R * DFF + 255) / 256, 256, 0, K.st>>>(D.g, D.u, D.h, R * DFF);
            dsgemv(K, D.down[il], D.h, DFF, D.o, N, R);
            k_add<<<(R * N + 255) / 256, 256, 0, K.st>>>(D.x, D.x, D.o, (size_t) R * N);
        }
        k_rms<<<R, 256, 0, K.st>>>(D.x, N, D.outnorm, D.hid, N, N, eps);
        CK(cudaMemsetAsync(D.state, 0, DRANK * 4, K.st));
        CK(cudaStreamSynchronize(K.st));
        (void) anchor_row_ready;
    }
    // one Markov step at block position i: the card's half of output . y and its argmax
    float h_amv[NC];
    int h_ami[NC];
    void ds_step(int c, int i, int prev) {
        Card& K = cards[c];
        DSW& D = K.D;
        CK(cudaSetDevice(K.dev));
        CK(cudaMemcpyAsync(D.prev, &h_prev_pinned[c], 4, cudaMemcpyHostToDevice, K.st));
        (void) prev;
        k_ds_joint_in<<<(DJ + 255) / 256, 256, 0, K.st>>>(D.state, D.w1, D.prev, D.hid + (size_t) i * N, D.ctxv, D.jin);
        k_gemv_bf16<<<(3 * DRANK + 7) / 8, 256, 0, K.st>>>(D.jw, 3 * DRANK, DJ, D.jin, D.jb, D.jout);
        k_ds_gates<<<1, DRANK, 0, K.st>>>(D.jout, D.state, D.lat);
        k_gemv_bf16<<<N / 8, 256, 0, K.st>>>(D.hw, N, DRANK, D.lat, D.hb, D.y);
        k_add<<<(N + 255) / 256, 256, 0, K.st>>>(D.y, D.y, D.hid + (size_t) i * N, N);
        const bool dv = !dvmap[c].empty();   // p13
        if (dv) gemv8_t1(K, D.dvout, D.y, segs({{D.dvout.rows, K.logits}}));
        else gemv8_t1(K, K.out, D.y, segs({{NV_C, K.logits}}));
        const int valid = dv ? D.dvout.rows : std::min(NV_C, NVU - c * NV_C);
        k_argmax_n<<<1, 1024, 0, K.st>>>(K.logits, valid, D.amv, D.ami, ds_pmin > 0.0f ? D.ams : nullptr, softcap);
        CK(cudaMemcpyAsync(&h_amv_pinned[c], D.amv, 4, cudaMemcpyDeviceToHost, K.st));
        if (ds_pmin > 0.0f) CK(cudaMemcpyAsync(&h_ams_pinned[c], D.ams, 4, cudaMemcpyDeviceToHost, K.st));
        CK(cudaMemcpyAsync(&h_ami_pinned[c], D.ami, 4, cudaMemcpyDeviceToHost, K.st));
        CK(cudaStreamSynchronize(K.st));
    }
    float* h_amv_pinned = nullptr;
    float* h_ams_pinned = nullptr;   // p12
    std::vector<int> dvmap[NC];      // p13: the draft vocabulary's ids of each card's head rows (empty = the whole vocabulary)
    // p12: drafts stop at the first one whose probability is below this (FNX_DRAFT_MIN_P in Flash-Next); 0 = off
    float ds_pmin = getenv("INK2_DS_PMIN") ? (float) atof(getenv("INK2_DS_PMIN")) : 0.7f;   // 21. 11.: 0.7 best of 0 / 0.3 / 0.5 / 0.7
    int* h_ami_pinned = nullptr;
    int* h_prev_pinned = nullptr;
    // up to k drafts after the anchor (the pending token at position P = n_past)
    std::vector<int> ds_draft(int anchor, int k) {
        std::vector<int> d;
        if (!ds_on || k <= 0 || n_past + DBLK > ctx) return d;
        k = std::min(k, DBLK - 1);
        embed(0, anchor);
        const int P = n_past;
        static const int rows_env = getenv("INK2_DS_ROWS") ? atoi(getenv("INK2_DS_ROWS")) : 1;   // p10: k + 1 rows (21. 11.: drafter 4.9 -> 3.2 ms a round, same acceptance); 0 = the block of DBLK
        const int R = rows_env > 0 ? std::min(DBLK, std::max(k + 1, rows_env)) : DBLK;
        both([&](int c) { ds_block(c, 0, P, R); });
        int prev = anchor;
        for (int i = 1; i <= k; ++i) {
            for (int c = 0; c < NC; ++c) h_prev_pinned[c] = prev;
            both([&](int c) { ds_step(c, i, prev); });
            const int t0 = dvmap[0].empty() ? h_ami_pinned[0] : dvmap[0][h_ami_pinned[0]];   // p13: card 0's ids are the smaller
            const int t1 = dvmap[1].empty() ? h_ami_pinned[1] + NV_C : dvmap[1][h_ami_pinned[1]];
            prev = h_amv_pinned[0] >= h_amv_pinned[1] ? t0 : t1;   // the first maximum: card 0's tokens come first
            if (ds_pmin > 0.0f) {   // p12: p(top) = 1 / sum over both cards (each card's sum is relative to its own max)
                const float m0 = softcap * tanhf(h_amv_pinned[0] / softcap), m1 = softcap * tanhf(h_amv_pinned[1] / softcap);
                const float m = std::max(m0, m1);
                const float tot = h_ams_pinned[0] * expf(m0 - m) + h_ams_pinned[1] * expf(m1 - m);
                if (1.0f / tot < ds_pmin) break;
            }
            d.push_back(prev);
        }
        return d;
    }

    // ------------------------------------------------------------------------------------------------ p3: the reading path
    void alloc_pf(Card& K) {
        CK(cudaSetDevice(K.dev));
        PFB& P = K.P;
        const size_t C = pfc, CN = C * N;
        P.C = pfc;
        if (cublasCreate(&P.cb) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate");
        cublasSetStream(P.cb, K.st);
        P.x0 = dev(K, CN);
        P.cur = dev(K, CN);
        for (int i = 0; i < MAXB; ++i) P.B[i] = dev(K, CN);
        P.prel = dev(K, CN);
        P.bankS = dev(K, CN);
        P.bankO = dev(K, CN);
        P.X3 = dev(K, 4 * CN);
        P.MS = dev(K, 4 * CN);
        for (int i = 0; i < 4; ++i) P.XC[i] = dev(K, CN);
        P.xm = dev(K, 2 * CN);
        P.g64 = dev(K, C * RANK);
        P.gp = dev(K, CN);
        P.fpp = dev(K, CN);
        P.NQ = dev(K, CN);
        P.NK = dev(K, CN);
        P.NV = dev(K, CN);
        P.q = dev(K, C * QC);
        P.k = dev(K, C * KVC);
        P.v = dev(K, C * KVC);
        P.kcv = dev(K, C * KVC);
        P.vcv = dev(K, C * KVC);
        P.r = dev(K, C * NH_C * NREL);
        P.sink = dev(K, C * NH_C);
        P.gate = dev(K, C * QC);
        P.dk = dev(K, C * NKV_C * NCONV);
        P.dv = dev(K, C * NKV_C * NCONV);
        P.ao = dev(K, C * QC);
        P.ab = dev(K, CN);
        P.A = dev(K, CN);
        P.F = dev(K, CN);
        P.F2 = dev(K, CN);
        P.FN = dev(K, CN);
        P.lg = dev(K, C * NE);
        P.topw = dev(K, C * TOPK);
        P.MO = dev(K, CN);
        P.tau = dev(K, C);
        P.tmp = dev(K, CN);
        P.hx1 = dev(K, N);
        P.Xg = dev(K, CN * TOPK);
        P.GU = dev(K, C * TOPK * 2 * FF_C);
        P.HH = dev(K, C * TOPK * FF_C);
        P.EY = dev(K, CN * TOPK);
        P.Wf = dev(K, (size_t) K.L[5].cat.rows * N);   // a full layer's concatenated matrix is the largest
        P.Wo = dev(K, (size_t) N * QC);
        P.Egu = dev(K, (size_t) EB * 2 * FF_C * N);
        P.Edn = dev(K, (size_t) EB * N * FF_C);
        const size_t kfr = (size_t) std::max(std::min(ctx, KFB), (int) C + SWA);   // p3f: a window chunk reads from p0 - (SWA - 1); p14: KFB
        P.Kf = dev(K, kfr * KVC);
        P.Vf = dev(K, kfr * KVC);
        const size_t s_full = (size_t) 3 * pf_qs_full * (ctx + pf_qs_full), s_win = (size_t) (C / QS) * NH_C * QS * (QS + SWA - 1),
                     s_tail = (size_t) 3 * QS * (QS + SWA);
        P.S_n = std::max(s_full, std::max(s_win, s_tail));
        P.S = dev(K, P.S_n);
        P.pv_nsb = (ctx + PVK - 1) / PVK;   // p14b
        P.PV = dev(K, (size_t) 3 * P.pv_nsb * pf_qs_full * HD);
        P.RB = dev(K, C * NH_C * (EXT + 1));
        {   // p3f: query block b, head h = g * 3 + a: entry b * NH_C + h; S of a window block QS x (QS + SWA - 1)
            const int nb = (int) C / QS, nk = QS + SWA - 1, n = nb * NH_C;
            if ((size_t) n * QS * nk > P.S_n) throw std::runtime_error("p3f: S premajhen");
            std::vector<float*> a(n), b2(n), c(n), d(n), e(n);
            for (int bb = 0; bb < nb; ++bb)
                for (int h = 0; h < NH_C; ++h) {
                    const int i = bb * NH_C + h, g = h / (NH / NKV);
                    a[i] = P.Kf + (size_t) bb * QS * KVC + g * HD;
                    b2[i] = P.q + (size_t) bb * QS * QC + h * HD;
                    c[i] = P.S + (size_t) i * QS * nk;
                    d[i] = P.Vf + (size_t) bb * QS * KVC + g * HD;
                    e[i] = P.ao + (size_t) bb * QS * QC + h * HD;
                }
            float*** dst[5] = {&P.pk, &P.pq, &P.ps, &P.pv, &P.pao};
            std::vector<float*>* srcv[5] = {&a, &b2, &c, &d, &e};
            for (int k = 0; k < 5; ++k) {
                *dst[k] = (float**) alloc(K, (size_t) std::max(n, 1) * sizeof(float*));
                CK(cudaMemcpy(*dst[k], srcv[k]->data(), (size_t) n * sizeof(float*), cudaMemcpyHostToDevice));
            }
        }
        P.pos = (int*) alloc(K, C * 4);
        P.topi = (int*) alloc(K, C * TOPK * 4);
        P.cnt = (int*) alloc(K, NE * 4);
        P.offs = (int*) alloc(K, (NE + 1) * 4);
        P.cursor = (int*) alloc(K, NE * 4);
        P.slotrow = (int*) alloc(K, C * TOPK * 4);
        P.rowtok = (int*) alloc(K, C * TOPK * 4);
        P.rowexp = (int*) alloc(K, C * TOPK * 4);
        for (int i = 0; i < 2; ++i) {
            CK(cudaHostAlloc((void**) &P.hbuf[i], CN * 4, cudaHostAllocPortable));
            CK(cudaEventCreateWithFlags(&P.ev[i], cudaEventDisableTiming));
        }
        P.h_cnt.resize(NE);
        P.h_offs.resize(NE + 1);
    }
    // y[C][rows] = x[C][cols] . W^T, W row-major f32 [rows][cols]
    void sgemm(Card& K, const float* W, int rows, int cols, const float* x, int ldx, float* y, int ldy, int C) {
        const float one = 1.0f, zero = 0.0f;
        cublasSgemm(K.P.cb, CUBLAS_OP_T, CUBLAS_OP_N, rows, C, cols, &one, W, cols, x, ldx, &zero, y, ldy);
    }
    void deq8(Card& K, const Mat& m, float* out) {
        const size_t n16 = (size_t) m.rows * m.cols / 16;
        k_q8s_f32<<<(unsigned) ((n16 + 255) / 256), 256, 0, K.st>>>(m.q, (const __half*) m.d, m.rows, m.cols, out);
    }
    void xch_pf(Card& K, float* y, size_t n) {   // through page-locked RAM by the copy engines; host-side order by pf_rec
        PFB& P = K.P;
        const int c = K.c;
        const unsigned sq = P.seq++;
        const int par = sq & 1;
        CK(cudaMemcpyAsync(P.hbuf[par], y, n * 4, cudaMemcpyDeviceToHost, K.st));
        CK(cudaEventRecord(P.ev[par], K.st));
        pf_rec[c].store(sq, std::memory_order_release);
        while (pf_rec[1 - c].load(std::memory_order_acquire) < sq) std::this_thread::yield();
        PFB& O = cards[1 - c].P;
        CK(cudaStreamWaitEvent(K.st, O.ev[par], 0));
        CK(cudaMemcpyAsync(P.tmp, O.hbuf[par], n * 4, cudaMemcpyHostToDevice, K.st));
        k_add_ord<<<(unsigned) ((n + 255) / 256), 256, 0, K.st>>>(y, P.tmp, n, c);
    }
    void conv_pf(Card& K, const float* x, int Cc, float* st, const float* w, const float* dyn, int hs, float k, float* y, int T) {
        const size_t n = (size_t) T * Cc;
        k_conv_par<<<(unsigned) ((n + 255) / 256), 256, 0, K.st>>>(x, Cc, st, w, dyn, hs, k, y, T);
        k_conv_state<<<(Cc + 255) / 256, 256, 0, K.st>>>(x, Cc, st, T);
    }
    void control_pf(Card& K, const LW& L, int j, const float* x, const float* m, float* out, int C) {
        PFB& P = K.P;
        k_ctrl_in<<<C, 256, 0, K.st>>>(x, m, N, P.xm);
        sgemm(K, L.c_down[j], RANK, 2 * N, P.xm, 2 * N, P.g64, RANK, C);
        sgemm(K, L.c_gain[j], N, RANK, P.g64, RANK, P.gp, N, C);
        sgemm(K, L.c_forget[j], N, RANK, P.g64, RANK, P.fpp, N, C);
        k_ctrl_out<<<(C * N + 255) / 256, 256, 0, K.st>>>(x, m, N, P.gp, P.fpp, out, C);
    }
    States states_pf(Card& K, const float* last) {
        States S{};
        for (int i = 0; i < K.nb; ++i) S.p[i] = K.P.B[i];
        S.p[K.nb] = last;
        S.n = K.nb + 1;
        return S;
    }
    // attention of a chunk on one slot: positions p0 .. p0 + C - 1 are in the cache already
    void attn_pf(Card& K, const LW& L, int slot, int p0, int C) {
        PFB& P = K.P;
        const bool win = !L.full;
        const float one = 1.0f, zero = 0.0f, scale = 1.0f / sqrtf((float) HD);
        // R.B for d = 0..EXT: RB[(i * NH_C + h)][d]
        if (pf_rb)
            k_rb<<<(C * NH_C + 7) / 8, 256, 0, K.st>>>(P.r, L.rel_bias_pad, C * NH_C, P.RB);
        else
            cublasSgemm(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, EXT + 1, C * NH_C, NREL, &one, L.rel_bias_pad, NREL, P.r, NREL, &zero, P.RB, EXT + 1);
        const bool bat = win && pf_attb;   // p3f: keys from p0 - (SWA - 1) even below 0 (zero rows, masked), so every block has the same shape
        const int klo = bat ? p0 - (SWA - 1) : win ? std::max(0, p0 - (SWA - 1)) : 0, khi = p0 + C;
        const int nkall = khi - klo;
        if (!win && nkall > KFB) {   // p14: long context: per query block and KV head, Q.K^T by key blocks into S, softmax, P.V by blocks
            for (int i0 = 0; i0 < C; i0 += pf_qs_full) {
                const int qs = std::min(pf_qs_full, C - i0), nk = p0 + i0 + qs;
                for (int g = 0; g < NKV_C; ++g) {
                    const int h0 = g * (NH / NKV);
                    for (int b0 = 0; b0 < nk; b0 += KFB) {
                        const int bn = std::min(KFB, nk - b0);
                        k_kv_f32g<<<(unsigned) (((size_t) bn * HD + 255) / 256), 256, 0, K.st>>>(K.kc[slot], K.vc[slot], b0, bn, g, P.Kf, P.Vf);
                        cublasSgemmStridedBatched(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, bn, qs, HD, &one, P.Kf, HD, 0, P.q + (size_t) i0 * QC + h0 * HD, QC,
                                                  HD, &zero, P.S + b0, nk, (long long) nk * qs, NH / NKV);
                    }
                    k_pf_softmax<<<dim3(qs, NH / NKV), 256, 0, K.st>>>(P.S, nk, 0, P.pos, i0, h0, P.RB, P.sink, 0, scale);
                    for (int b0 = 0; b0 < nk; b0 += KFB) {   // p14b: P.V by key pieces (KFB is a multiple of PVK)
                        const int bn = std::min(KFB, nk - b0);
                        k_kv_f32g<<<(unsigned) (((size_t) bn * HD + 255) / 256), 256, 0, K.st>>>(K.kc[slot], K.vc[slot], b0, bn, g, nullptr, P.Vf);
                        pv_parts(K, P.Vf, HD, b0, bn, nk, qs);
                    }
                    const int nsb = (nk + PVK - 1) / PVK;
                    k_pv_sum<<<(unsigned) ((3 * (size_t) qs * HD + 255) / 256), 256, 0, K.st>>>(P.PV, nsb, P.pv_nsb, qs, P.ao, i0, h0);
                }
            }
            k_sigmul<<<(unsigned) (((size_t) C * QC + 255) / 256), 256, 0, K.st>>>(P.ao, P.gate, (size_t) C * QC);
            return;
        }
        k_kv_f32<<<(unsigned) (((size_t) nkall * KVC + 255) / 256), 256, 0, K.st>>>(K.kc[slot], K.vc[slot], klo, nkall, win, P.Kf, P.Vf);
        int ib = 0;
        if (bat && C / QS > 0) {
            const int nb = C / QS, nk = QS + SWA - 1;
            cublasSgemmBatched(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, nk, QS, HD, &one, (const float**) P.pk, KVC, (const float**) P.pq, QC, &zero,
                               P.ps, nk, nb * NH_C);
            k_pf_softmax_b<<<dim3(QS, NH_C, nb), 256, 0, K.st>>>(P.S, nk, p0, QS, P.pos, P.RB, P.sink, scale);
            cublasSgemmBatched(P.cb, CUBLAS_OP_N, CUBLAS_OP_N, HD, QS, nk, &one, (const float**) P.pv, KVC, (const float**) P.ps, nk, &zero,
                               P.pao, QC, nb * NH_C);
            ib = nb * QS;
        }
        const int QSL = win ? QS : pf_qs_full;   // p3d: bigger query blocks for the full layers (bigger cuBLAS calls)
        for (int i0 = ib; i0 < C; i0 += QSL) {
            const int qs = std::min(QSL, C - i0);
            const int lo = bat ? p0 + i0 - (SWA - 1) : win ? std::max(0, p0 + i0 - (SWA - 1)) : 0, hi = p0 + i0 + qs, nk = hi - lo;
            const float* kf = P.Kf + (size_t) (lo - klo) * KVC;
            const float* vf = P.Vf + (size_t) (lo - klo) * KVC;
            for (int g = 0; g < NKV_C; ++g) {
                const int h0 = g * (NH / NKV);
                cublasSgemmStridedBatched(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, nk, qs, HD, &one, kf + g * HD, KVC, 0,
                                          P.q + (size_t) i0 * QC + h0 * HD, QC, HD, &zero, P.S, nk, (long long) nk * qs, NH / NKV);
                k_pf_softmax<<<dim3(qs, NH / NKV), 256, 0, K.st>>>(P.S, nk, lo, P.pos, i0, h0, P.RB, P.sink, win, scale);
                if (!win && pv_split && nk > 4 * PVK) {   // p14b: many keys, few output tiles: P.V by key pieces
                    pv_parts(K, vf + g * HD, KVC, 0, nk, nk, qs);
                    k_pv_sum<<<(unsigned) ((3 * (size_t) qs * HD + 255) / 256), 256, 0, K.st>>>(P.PV, (nk + PVK - 1) / PVK, P.pv_nsb, qs, P.ao, i0, h0);
                } else
                cublasSgemmStridedBatched(P.cb, CUBLAS_OP_N, CUBLAS_OP_N, HD, qs, nk, &one, vf + g * HD, KVC, 0, P.S, nk, (long long) nk * qs,
                                          &zero, P.ao + (size_t) i0 * QC + h0 * HD, QC, HD, NH / NKV);
            }
        }
        k_sigmul<<<(unsigned) (((size_t) C * QC + 255) / 256), 256, 0, K.st>>>(P.ao, P.gate, (size_t) C * QC);
    }
    // p14b: the P.V of KV head group h0 for keys [b0, b0 + bn) (V rows at vb, row stride ldv) into the parts of pieces b0 / PVK ..
    void pv_parts(Card& K, const float* vb, int ldv, int b0, int bn, int nk, int qs) {
        PFB& P = K.P;
        const float one = 1.0f, zero = 0.0f;
        const int sb0 = b0 / PVK, nfull = bn / PVK, tail = bn % PVK;
        for (int a = 0; a < 3; ++a) {
            const float* Sa = P.S + (size_t) a * nk * qs + b0;
            float* Pa = P.PV + ((size_t) a * P.pv_nsb + sb0) * qs * HD;
            if (nfull)
                cublasSgemmStridedBatched(P.cb, CUBLAS_OP_N, CUBLAS_OP_N, HD, qs, PVK, &one, vb, ldv, (long long) PVK * ldv, Sa, nk, PVK, &zero,
                                          Pa, HD, (long long) qs * HD, nfull);
            if (tail)
                cublasSgemm(P.cb, CUBLAS_OP_N, CUBLAS_OP_N, HD, qs, tail, &one, vb + (size_t) nfull * PVK * ldv, ldv, Sa + (size_t) nfull * PVK, nk,
                            &zero, Pa + (size_t) nfull * qs * HD, HD);
        }
    }
    void experts_pf(Card& K, const LW& L, int C) {
        PFB& P = K.P;
        const int n = C * TOPK;
        const float one = 1.0f, zero = 0.0f;
        CK(cudaMemsetAsync(P.cnt, 0, NE * 4, K.st));
        k_exp_count<<<(n + 255) / 256, 256, 0, K.st>>>(P.topi, n, P.cnt);
        CK(cudaMemcpyAsync(P.h_cnt.data(), P.cnt, NE * 4, cudaMemcpyDeviceToHost, K.st));
        CK(cudaStreamSynchronize(K.st));
        P.h_offs[0] = 0;
        for (int e = 0; e < NE; ++e) P.h_offs[e + 1] = P.h_offs[e] + P.h_cnt[e];
        CK(cudaMemcpyAsync(P.offs, P.h_offs.data(), (NE + 1) * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemsetAsync(P.cursor, 0, NE * 4, K.st));
        k_exp_fill<<<(n + 255) / 256, 256, 0, K.st>>>(P.topi, n, P.offs, P.cursor, P.slotrow, P.rowtok, P.rowexp);
        k_gather_rows<<<(unsigned) (((size_t) n * (N / 4) + 255) / 256), 256, 0, K.st>>>(P.FN, P.rowtok, n, P.Xg);
        const size_t gu_rows = 2 * FF_C, gu_e = gu_rows * N, dn_e = (size_t) N * FF_C;
        if (L.fp4 && !pf_deq) {   // p3c: the NVFP4 tiles straight from the weights (p4d: planar unless INK2_FP4_ILV)
            auto gemm = ilv ? k_fp4_gemm<false> : (gemm2 ? k_fp4_gemm2<true> : k_fp4_gemm<true>);   // p3e: INK2_GEMM2 = 128 x 128 tiles
            gemm<<<dim3(gu_rows / F4M, NE), 256, 0, K.st>>>(L.gu, gu_rows * (N / 64) * 36, (int) gu_rows, N, P.offs, P.Xg, N, P.GU, (int) gu_rows);
            k_situ_s<<<(n * FF_C + 255) / 256, 256, 0, K.st>>>(P.GU, P.HH, n, P.rowexp, L.gu_s, situ_b, situ_lb);
            gemm<<<dim3(N / F4M, NE), 256, 0, K.st>>>(L.dn, (size_t) N * (FF_C / 64) * 36, N, FF_C, P.offs, P.HH, FF_C, P.EY, N);
            k_moe_comb<<<(unsigned) (((size_t) C * N + 255) / 256), 256, 0, K.st>>>(P.EY, P.slotrow, P.topi, P.topw, L.dn_s, P.MO, C);
            return;
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (int e0 = 0; e0 < NE; e0 += EB) {
                if (P.h_offs[e0 + EB] == P.h_offs[e0]) continue;
                float* E = pass == 0 ? P.Egu : P.Edn;
                const size_t rows = (pass == 0 ? gu_rows : (size_t) N) * EB;
                const int cols = pass == 0 ? N : FF_C;
                if (L.fp4) {
                    const uint8_t* W = pass == 0 ? L.gu + (size_t) e0 * gu_rows * (N / 64) * 36 : L.dn + (size_t) e0 * N * (FF_C / 64) * 36;
                    const size_t n4 = rows * (cols / 4);
                    k_fp4_f32b<<<(unsigned) ((n4 + 255) / 256), 256, 0, K.st>>>(W, rows, cols, E);
                } else {
                    const int8_t* q = (const int8_t*) (pass == 0 ? L.gu + (size_t) e0 * gu_e : L.dn + (size_t) e0 * dn_e);
                    const __half* d = pass == 0 ? L.gu_d + (size_t) e0 * gu_rows * (N / 32) : L.dn_d + (size_t) e0 * N * (FF_C / 32);
                    const size_t n16 = rows * cols / 16;
                    k_q8s_f32<<<(unsigned) ((n16 + 255) / 256), 256, 0, K.st>>>(q, d, rows, cols, E);
                }
                for (int e = e0; e < e0 + EB; ++e) {
                    const int ne = P.h_cnt[e], o = P.h_offs[e];
                    if (!ne) continue;
                    if (pass == 0)
                        cublasSgemm(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, (int) gu_rows, ne, N, &one, E + (size_t) (e - e0) * gu_e, N,
                                    P.Xg + (size_t) o * N, N, &zero, P.GU + (size_t) o * gu_rows, (int) gu_rows);
                    else
                        cublasSgemm(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, N, ne, FF_C, &one, E + (size_t) (e - e0) * dn_e, FF_C,
                                    P.HH + (size_t) o * FF_C, FF_C, &zero, P.EY + (size_t) o * N, N);
                }
            }
            if (pass == 0)
                k_situ_s<<<(n * FF_C + 255) / 256, 256, 0, K.st>>>(P.GU, P.HH, n, P.rowexp, L.fp4 ? L.gu_s : nullptr, situ_b, situ_lb);
        }
        k_moe_comb<<<(unsigned) (((size_t) C * N + 255) / 256), 256, 0, K.st>>>(P.EY, P.slotrow, P.topi, P.topw, L.fp4 ? L.dn_s : nullptr, P.MO, C);
    }
    void layer_pf(Card& K, int p, int slot, bool loop, int p0, int C) {
        const LW& L = K.L[p];
        PFB& P = K.P;
        const size_t CN = (size_t) C * N;
        const float* xin[3] = {P.cur, P.cur, P.cur};
        if (L.attn_rq) {
            mix(K, states_pf(K, P.cur), L.attn_rq, L.nrq, P.X3, C);
            for (int j = 0; j < 3; ++j) xin[j] = P.X3 + (L.full ? j : 0) * CN;
        }
        if (loop) {
            States S{};
            S.p[0] = P.bankS;
            S.p[1] = P.bankO;
            S.n = 2;
            mix(K, S, L.msg_q, 4, P.MS, C);
            for (int j = 0; j < 3; ++j) {
                control_pf(K, L, j, xin[j], P.MS + j * CN, P.XC[j], C);
                xin[j] = P.XC[j];
            }
        }
        const float *nq, *nk, *nv;
        auto rmsC = [&](const float* x, const float* w, float* y) { k_rms<<<C, 256, 0, K.st>>>(x, N, w, y, N, N, eps); };
        if (!L.full && !loop) {
            rmsC(xin[0], L.attn_norm, P.NQ);
            nq = nk = nv = P.NQ;
        } else {
            rmsC(xin[0], L.full ? L.q_norm : L.attn_norm, P.NQ);
            rmsC(xin[1], L.full ? L.k_norm : L.attn_norm, P.NK);
            rmsC(xin[2], L.full ? L.v_norm : L.attn_norm, P.NV);
            nq = P.NQ;
            nk = P.NK;
            nv = P.NV;
        }
        deq8(K, L.cat, P.Wf);
        const int nqr = 2 * QC + NH_C * NREL + NH_C, nkr = KVC + (L.full ? NKV_C * NCONV : 0);
        sgemm(K, P.Wf, QC, N, nq, N, P.q, QC, C);
        sgemm(K, P.Wf + (size_t) QC * N, QC, N, nq, N, P.gate, QC, C);
        sgemm(K, P.Wf + (size_t) 2 * QC * N, NH_C * NREL, N, nq, N, P.r, NH_C * NREL, C);
        sgemm(K, P.Wf + (size_t) (2 * QC + NH_C * NREL) * N, NH_C, N, nq, N, P.sink, NH_C, C);
        sgemm(K, P.Wf + (size_t) nqr * N, KVC, N, nk, N, P.k, KVC, C);
        sgemm(K, P.Wf + (size_t) (nqr + nkr) * N, KVC, N, nv, N, P.v, KVC, C);
        if (L.full) {
            sgemm(K, P.Wf + (size_t) (nqr + KVC) * N, NKV_C * NCONV, N, nk, N, P.dk, NKV_C * NCONV, C);
            sgemm(K, P.Wf + (size_t) (nqr + nkr + KVC) * N, NKV_C * NCONV, N, nv, N, P.dv, NKV_C * NCONV, C);
        }
        const float kconv = 1.0f / sqrtf((float) ((L.full ? 2 : 1) * NCONV + 1));
        conv_pf(K, P.k, KVC, K.sk[slot], L.k_conv, L.full ? P.dk : nullptr, HD, kconv, P.kcv, C);
        conv_pf(K, P.v, KVC, K.sv[slot], L.v_conv, L.full ? P.dv : nullptr, HD, kconv, P.vcv, C);
        const float* tau = L.full ? P.tau : nullptr;
        k_headnorm<<<(C * NH_C + 7) / 8, 256, 0, K.st>>>(P.q, NH_C, HD, L.q_clip, K.c * NH_C, qk_clip, tau, 0, eps, C);
        k_headnorm<<<(C * NKV_C + 7) / 8, 256, 0, K.st>>>(P.kcv, NKV_C, HD, nullptr, 0, qk_clip, nullptr, 0, eps, C);
        k_headnorm<<<(C * NKV_C + 7) / 8, 256, 0, K.st>>>(P.vcv, NKV_C, HD, nullptr, 0, 0.0f, nullptr, 1, eps, C);
        k_headnorm<<<(C * NH_C + 7) / 8, 256, 0, K.st>>>(P.r, NH_C, NREL, L.rel_clip, K.c * NH_C, rel_clip_thr, tau, 0, eps, C);
        k_kv_store<<<(C * KVC + 255) / 256, 256, 0, K.st>>>(P.kcv, P.vcv, P.pos, !L.full, K.kc[slot], K.vc[slot], C);
        attn_pf(K, L, slot, p0, C);
        deq8(K, L.wo, P.Wo);
        sgemm(K, P.Wo, N, QC, P.ao, QC, P.ab, N, C);
        xch_pf(K, P.ab, CN);
        conv_pf(K, P.ab, N, K.sa[slot], L.a_pconv, nullptr, 1, 1.0f / sqrtf(NCONV + 1.0f), P.A, C);
        const int ip = slot < NPHYS ? slot : LS + (slot - NPHYS) % NCORE;
        if (ip % BLK == 0) {
            CK(cudaMemcpyAsync(P.B[K.nb++], P.cur, CN * 4, cudaMemcpyDeviceToDevice, K.st));
            CK(cudaMemcpyAsync(P.cur, P.A, CN * 4, cudaMemcpyDeviceToDevice, K.st));
        } else {
            k_add<<<(unsigned) ((CN + 255) / 256), 256, 0, K.st>>>(P.cur, P.cur, P.A, CN);
        }
        mix(K, states_pf(K, P.cur), L.ffn_rq, 1, P.F, C);
        const float* f = P.F;
        if (loop) {
            control_pf(K, L, 3, P.F, P.MS + 3 * CN, P.F2, C);
            f = P.F2;
        }
        rmsC(f, L.ffn_norm, P.FN);
        sgemm(K, L.gate_inp, NE, N, P.FN, N, P.lg, NE, C);
        k_route<<<C, NE, 0, K.st>>>(P.lg, L.probs_b, wscale, P.topi, P.topw);
        experts_pf(K, L, C);
        xch_pf(K, P.MO, CN);
        conv_pf(K, P.MO, N, K.sf[slot], L.f_pconv, nullptr, 1, 1.0f / sqrtf(NCONV + 1.0f), P.F2, C);
        k_add<<<(unsigned) ((CN + 255) / 256), 256, 0, K.st>>>(P.cur, P.cur, P.F2, CN);
        if (ds_on && ds_feat_slot(slot) >= 0) ds_feat(K, K.D.pfeat, P.cur, ds_feat_slot(slot), C);   // p6
    }
    // a chunk of C tokens at positions p0.. through both cards; the last row's logits into h_logits when last
    void pf_card(int c, int C, int p0, bool last) {
        Card& K = cards[c];
        PFB& P = K.P;
        CK(cudaSetDevice(K.dev));
        CK(cudaMemcpyAsync(P.x0, h_x0, (size_t) C * N * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(P.pos, h_pos, C * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(P.tau, h_tau, C * 4, cudaMemcpyHostToDevice, K.st));
        k_rms<<<C, 256, 0, K.st>>>(P.x0, N, K.tok_norm, P.cur, N, N, eps);
        K.nb = 0;
        const size_t CN = (size_t) C * N;
        for (int p = 0; p < LS; ++p) layer_pf(K, p, p, false, p0, C);
        CK(cudaMemcpyAsync(P.prel, P.cur, CN * 4, cudaMemcpyDeviceToDevice, K.st));
        const int nprel = K.nb;
        for (int pass = 0; pass < NPASS; ++pass) {
            if (pass > 0) {
                CK(cudaMemcpyAsync(P.bankS, P.B[nprel + 1], CN * 4, cudaMemcpyDeviceToDevice, K.st));
                CK(cudaMemcpyAsync(P.bankO, P.cur, CN * 4, cudaMemcpyDeviceToDevice, K.st));
                CK(cudaMemcpyAsync(P.cur, P.prel, CN * 4, cudaMemcpyDeviceToDevice, K.st));
                K.nb = nprel;
            }
            for (int ip = LS; ip < LE; ++ip) layer_pf(K, ip, pass == 0 ? ip : NPHYS + (pass - 1) * NCORE + (ip - LS), pass > 0, p0, C);
        }
        for (int p = LE; p < NPHYS; ++p) layer_pf(K, p, p, false, p0, C);
        if (ds_on) {   // p6: every row's final hidden, then the drafter's caches
            mix(K, states_pf(K, P.cur), K.out_rq, 1, P.F, C);
            k_rms<<<C, 256, 0, K.st>>>(P.F, N, K.out_norm, P.F2, N, N, eps);
            ds_feat(K, K.D.pfeat, P.F2, 3, C);
            ds_ingest(K, K.D.pfeat, K.D.pfrnn, K.D.pcbuf, K.D.pkb, K.D.pvb, P.pos, C, true);
        }
        if (last) {   // the last row only: the depth mix, the norm and the head
            States S{};
            for (int i = 0; i < K.nb; ++i) S.p[i] = P.B[i] + (size_t) (C - 1) * N;
            S.p[K.nb] = P.cur + (size_t) (C - 1) * N;
            S.n = K.nb + 1;
            mix(K, S, K.out_rq, 1, P.F, 1);
            rms(K, P.F, K.out_norm, P.hx1, 1, eps);
            gemv8_t1(K, K.out, P.hx1, segs({{NV_C, K.logits}}));
            k_softcap<<<(NV_C + 255) / 256, 256, 0, K.st>>>(K.logits, NV_C, softcap);
            CK(cudaMemcpyAsync(h_logits + (size_t) c * NV_C, K.logits, (size_t) NV_C * 4, cudaMemcpyDeviceToHost, K.st));
        }
        CK(cudaStreamSynchronize(K.st));
    }

    // T tokens at positions n_past.. through both cards; logits of the last row (or of all rows) into h_logits
    void forward_card(int c, int T, int nparts, bool all_rows) {
        Card& K = cards[c];
        CK(cudaSetDevice(K.dev));
        if (T == 1 && !all_rows && !dumping && graf) {   // p4c: replay (or first capture) this parts bucket's graph
            auto it = K.graphs.find(nparts);
            if (it == K.graphs.end()) {
                CK(cudaStreamBeginCapture(K.st, cudaStreamCaptureModeThreadLocal));
                forward_body(c, T, nparts, all_rows);
                cudaGraph_t gr;
                CK(cudaStreamEndCapture(K.st, &gr));
                cudaGraphExec_t ex;
                CK(cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0));
                CK(cudaGraphDestroy(gr));
                it = K.graphs.emplace(nparts, ex).first;
            }
            CK(cudaGraphLaunch(it->second, K.st));
        } else {
            forward_body(c, T, nparts, all_rows);
        }
        CK(cudaStreamSynchronize(K.st));
    }
    void forward_body(int c, int T, int nparts, bool all_rows) {
        Card& K = cards[c];
        K.xidx = 0;
        k_tick<<<1, 1, 0, K.st>>>(K.dtok);
        CK(cudaMemcpyAsync(K.x0, h_x0, (size_t) T * N * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(K.pos, h_pos, T * 4, cudaMemcpyHostToDevice, K.st));
        CK(cudaMemcpyAsync(K.tau, h_tau, T * 4, cudaMemcpyHostToDevice, K.st));
        rms(K, K.x0, K.tok_norm, K.cur, T, eps);
        K.nb = 0;
        for (int p = 0; p < LS; ++p) layer(K, p, p, false, T, nparts);
        copy(K, K.prel, K.cur, T);
        const int nprel = K.nb;
        for (int pass = 0; pass < NPASS; ++pass) {
            if (pass > 0) {
                copy(K, K.bankS, K.B[nprel + 1], T);
                copy(K, K.bankO, K.cur, T);
                copy(K, K.cur, K.prel, T);
                K.nb = nprel;
            }
            for (int ip = LS; ip < LE; ++ip) {
                const int slot = pass == 0 ? ip : NPHYS + (pass - 1) * NCORE + (ip - LS);
                layer(K, ip, slot, pass > 0, T, nparts);
            }
        }
        for (int p = LE; p < NPHYS; ++p) layer(K, p, p, false, T, nparts);
        mix(K, states(K, K.cur), K.out_rq, 1, K.F, T);
        rms(K, K.F, K.out_norm, K.hx, T, eps);
        if (ds_on) {   // p6: the final hidden is feature 3; the drafter's caches get every row
            ds_feat(K, K.D.feat, K.hx, 3, T);
            ds_ingest(K, K.D.feat, K.D.frnn, K.D.cbuf, K.D.kb, K.D.vb, K.pos, T, false);
        }
        const int r0 = all_rows ? 0 : T - 1, nr = T - r0;
        if (nr == 1 && !stare) gemv8_t1(K, K.out, K.hx + (size_t) r0 * N, segs({{NV_C, K.logits}}));
        else if (fuse && !stare && !dumping) gemv8_t1T(K, K.out, K.hx + (size_t) r0 * N, N, segs1(NV_C, K.logits), nr);   // p6c
        else gemv8(K, K.out, K.hx + (size_t) r0 * N, N, K.logits, NV_C, nr);
        k_softcap<<<(nr * NV_C + 255) / 256, 256, 0, K.st>>>(K.logits, nr * NV_C, softcap);
        CK(cudaMemcpy2DAsync(h_logits + (size_t) c * NV_C, (size_t) NV * 4, K.logits, (size_t) NV_C * 4, (size_t) NV_C * 4, nr,
                             cudaMemcpyDeviceToHost, K.st));
    }
    void embed(int t, int id) {
        const size_t rb = tok->row_bytes();
        const uint8_t* row = tok->data + (size_t) id * rb;
        for (int b = 0; b < N / 32; ++b) {
            const uint8_t* blk = row + (size_t) b * 34;
            uint16_t hb;
            std::memcpy(&hb, blk, 2);
            __half_raw hr;
            hr.x = hb;
            const float d = __half2float(__half(hr));
            for (int j = 0; j < 32; ++j) h_x0[(size_t) t * N + b * 32 + j] = d * (float) (int8_t) blk[2 + j];
        }
    }
    // feeds tokens (windows of MAXT); returns the logits rows of the last window (all its rows when all_rows)
    int feed(const int* ids, int n, bool all_rows) {
        int T = 0;
        for (int i = 0; i < n; i += T) {
            if (!all_rows && n - i >= pf_min) {   // p3: the reading path, chunks of up to pfc
                T = std::min(pfc, n - i);
                if (n_past + T > ctx) throw std::runtime_error("context full");
                for (int t = 0; t < T; ++t) {
                    embed(t, ids[i + t]);
                    const int P = n_past + t;
                    h_pos[t] = P;
                    h_tau[t] = logf(fmaxf(((float) P + 1.0f) * (1.0f / log_floor), 1.0f)) * log_alpha + 1.0f;
                }
                const bool last = i + T >= n;
                const int p0 = n_past;
                both([&](int c) { pf_card(c, T, p0, last); });
                n_past += T;
                processed.insert(processed.end(), ids + i, ids + i + T);
                continue;
            }
            T = std::min(feed_t, n - i);
            if (n_past + T > ctx) throw std::runtime_error("context full");
            for (int t = 0; t < T; ++t) {
                embed(t, ids[i + t]);
                const int P = n_past + t;
                h_pos[t] = P;
                h_tau[t] = logf(fmaxf(((float) P + 1.0f) * (1.0f / log_floor), 1.0f)) * log_alpha + 1.0f;
            }
            int nparts = (n_past + T + KC - 1) / KC + 1;
            if (T == 1 && graf) {   // p4c: graphs by buckets of parts (empty parts return at once)
                int b = 4;
                while (b < nparts) b <<= 1;
                nparts = std::min(b, cards[0].nparts_max);
            }
            const bool last = i + T >= n;
            both([&](int c) { forward_card(c, T, nparts, all_rows && last); });
            n_past += T;
            processed.insert(processed.end(), ids + i, ids + i + T);
        }
        return all_rows ? T : 1;
    }
    // p8: the states before the last prompt token, so a request that repeats the prompt (a conversation's next turn re-renders
    // the reply differently, e.g. without its reasoning) reads only from there. The caches need nothing: rows of later positions
    // are written again, earlier rows stay while the window ring was not overrun (RING - SWA - MAXT positions since)
    void snapshot() {
        both([&](int c) {
            Card& K = cards[c];
            CK(cudaSetDevice(K.dev));
            CK(cudaMemcpyAsync(K.stsnap, K.stbuf, (size_t) NSLOT * SLOTST * 4, cudaMemcpyDeviceToDevice, K.st));
            CK(cudaStreamSynchronize(K.st));
        });
        snap_n = n_past;
    }
    bool restore(size_t same) {
        if (snap_n <= 0 || (int) same < snap_n || (int) processed.size() < snap_n || n_past - snap_n > RING - SWA - MAXT) return false;
        both([&](int c) {
            Card& K = cards[c];
            CK(cudaSetDevice(K.dev));
            CK(cudaMemcpyAsync(K.stbuf, K.stsnap, (size_t) NSLOT * SLOTST * 4, cudaMemcpyDeviceToDevice, K.st));
            CK(cudaStreamSynchronize(K.st));
        });
        n_past = snap_n;
        processed.resize(snap_n);
        return true;
    }
    void reset() {
        n_past = 0;
        snap_n = -1;
        processed.clear();
        both([&](int c) {
            Card& K = cards[c];
            CK(cudaSetDevice(K.dev));
            CK(cudaMemsetAsync(K.stbuf, 0, (size_t) NSLOT * SLOTST * 4, K.st));
            CK(cudaStreamSynchronize(K.st));
        });
    }
    // p5a: tokens win[0..T) through the writing path with every row's logits; win[0] is the pending token, win[1..] drafts. Returns
    // how many drafts were right (row i's argmax == win[i + 1]); the caches then hold win[0..keep] (keep = right + 1) and the
    // convolutions' states are set back to that point. out_next = the argmax after the last kept token
    int verify(const int* win, int T, int& out_next) {
        both([&](int c) {
            Card& K = cards[c];
            CK(cudaSetDevice(K.dev));
            CK(cudaMemcpyAsync(K.stbak, K.stbuf, (size_t) NSLOT * SLOTST * 4, cudaMemcpyDeviceToDevice, K.st));
            K.keep_hist = true;
        });
        feed(win, T, true);
        int right = 0;
        while (right + 1 < T && argmax(h_logits + (size_t) right * NV) == win[right + 1]) ++right;
        out_next = argmax(h_logits + (size_t) right * NV);
        const int keep = right + 1;
        if (ds_on)   // p6: the drafter's context is the last kept row's f_rnn (the rejected rows' caches are rewritten later)
            both([&](int c) {
                Card& K = cards[c];
                CK(cudaSetDevice(K.dev));
                CK(cudaMemcpyAsync(K.D.ctxv, K.D.frnn + (size_t) right * N, N * 4, cudaMemcpyDeviceToDevice, K.st));
                CK(cudaStreamSynchronize(K.st));
            });
        both([&](int c) {
            Card& K = cards[c];
            CK(cudaSetDevice(K.dev));
            K.keep_hist = false;
            if (keep < T) {
                const size_t n = (size_t) NSLOT * SLOTST;
                k_conv_roll<<<(unsigned) ((n + 255) / 256), 256, 0, K.st>>>(K.stbuf, K.stbak, K.hist, keep);
            }
            CK(cudaStreamSynchronize(K.st));
        });
        n_past -= T - keep;
        processed.resize(processed.size() - (T - keep));
        return right;
    }
    static int argmax(const float* l) {
        int b = 0;
        for (int i = 1; i < NVU; ++i)
            if (l[i] > l[b]) b = i;
        return b;
    }
};

static void write_dump(const std::string& path, const std::vector<float>& v, int cols) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    const int32_t nd = 2;
    const int64_t ne[4] = {cols, (int64_t) (v.size() / cols), 1, 1};
    std::fwrite(&nd, 4, 1, f);
    std::fwrite(ne, 8, 4, f);
    std::fwrite(v.data(), 4, v.size(), f);
    std::fclose(f);
}

// --serve SOCKET: one JSON line a connection, {"tokens": [...], "max": N, "stop": [...], "noeos": 0} -> {"tokens": [...], "prompt_n", "cached",
// "read_s", "write_s"}. A prompt that continues exactly the tokens already processed reads only the rest; anything else starts over
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

struct Gen {
    std::vector<int> out;
    int cached = 0, drafted = 0, accepted = 0, rounds = 0;
    double read_s = 0, write_s = 0, draft_s = 0, verify_s = 0;   // p6: time in the drafter and in the target's windows
};
// p5a: draft tokens from the history: the latest earlier place where the last 3 tokens (then 2) occurred, and what followed it
// p7: the longest suffix of the history (3..NG_MAX tokens) that occurred before, at its most recent place, and the tokens that
// followed it there (p5a took the latest 3-token match: in code that is often a common trigram elsewhere). *len = match length
static constexpr int NG_MAX = 16;
static std::vector<int> ngram_draft(const std::vector<int>& h, int D, int* len = nullptr) {
    std::vector<int> d;
    if (len) *len = 0;
    const int n = (int) h.size();
    if (D <= 0 || n < 4) return d;
    int best = 0, at = -1;
    for (int p = n - 2; p >= 0 && best < NG_MAX; --p) {
        if (h[p] != h[n - 1]) continue;
        int l = 1;
        while (l < NG_MAX && p - l >= 0 && h[p - l] == h[n - 1 - l]) ++l;
        if (l > best) {
            best = l;
            at = p;
        }
    }
    if (best < 3) return d;
    if (len) *len = best;
    for (int i = at + 1; i < n && (int) d.size() < D; ++i) d.push_back(h[i]);
    return d;
}
// p15: on_out(new tokens) after each step when given (the server streams them); false stops the generation (the client left)
static Gen generate(Engine& e, const std::vector<int>& tok, int max, const std::vector<int>& stop, int eos, int lookup = 0, int mtp = 0,
                    const std::function<bool(const int*, int)>* on_out = nullptr) {
    Gen g;
    size_t sent = 0;
    auto flush = [&]() {
        if (!on_out || sent == g.out.size()) return true;
        const bool ok = (*on_out)(g.out.data() + sent, (int) (g.out.size() - sent));
        sent = g.out.size();
        return ok;
    };
    static const bool snap = getenv("INK2_SNAP") ? atoi(getenv("INK2_SNAP")) != 0 : true;   // p8
    size_t same = 0;
    while (same < tok.size() && same < e.processed.size() && tok[same] == e.processed[same]) ++same;
    if (same != e.processed.size() || same == tok.size()) {   // not a continuation (or nothing new to read)
        if (!(snap && e.restore(same))) {   // p8: back to the last prompt's snapshot, or start over
            e.reset();
            same = 0;
        }
        same = std::min(same, e.processed.size());
    }
    g.cached = (int) same;
    const double t0 = now_s();
    const int n = (int) tok.size();
    if (snap && n - (int) same >= 2) {   // p8: all but the last token, the snapshot, then the last token
        e.feed(tok.data() + same, n - 1 - (int) same, false);
        e.snapshot();
        e.feed(tok.data() + n - 1, 1, false);
    } else {
        e.feed(tok.data() + same, n - (int) same, false);
    }
    g.read_s = now_s() - t0;
    const double t1 = now_s();
    int next = Engine::argmax(e.h_logits);
    auto ends = [&](int t) { return t == eos || std::find(stop.begin(), stop.end(), t) != stop.end(); };
    int dcur = std::min(lookup, 4), cool = 0;   // p5a: the draft length follows the last window; after a miss no drafts for a while
    static const int ng_min = getenv("INK2_NG_MIN") ? atoi(getenv("INK2_NG_MIN")) : 6;   // p7: n-gram match that beats the drafter
    while ((int) g.out.size() < max) {
        g.out.push_back(next);
        if (!flush()) break;
        if (ends(next) || (int) g.out.size() >= max || e.n_past >= e.ctx) break;
        std::vector<int> hist = e.processed;
        hist.push_back(next);
        std::vector<int> d;
        if (lookup > 0) {   // n-grams first (p6d: with mtp too, where the history has a match; p7: with mtp only a match of ng_min)
            int ml = 0;
            if (cool > 0) --cool;
            else d = ngram_draft(hist, std::min(dcur, std::min(MAXT - 1, max - (int) g.out.size())), &ml);
            if (mtp > 0 && ml < ng_min) d.clear();
        }
        bool mtpd = false;
        if (d.empty() && mtp > 0) {   // p6: the DSpark drafter
            mtpd = true;
            const double td = now_s();
            d = e.ds_draft(next, std::min(mtp, max - (int) g.out.size()));
            g.draft_s += now_s() - td;
        }
        if ((int) d.size() > e.ctx - e.n_past - 1) d.resize(std::max(0, e.ctx - e.n_past - 1));
        if (d.empty()) {
            e.feed(&next, 1, false);
            next = Engine::argmax(e.h_logits);
            continue;
        }
        std::vector<int> win{next};
        win.insert(win.end(), d.begin(), d.end());
        int nn = 0;
        const double tv = now_s();
        const int right = e.verify(win.data(), (int) win.size(), nn);
        g.verify_s += now_s() - tv;
        ++g.rounds;
        g.drafted += (int) d.size();
        g.accepted += right;
        if (lookup > 0 && !mtpd) {   // the n-gram draft length follows the last n-gram window (p7: not the drafter's)
            if (right == (int) d.size()) dcur = std::min(lookup, std::max(2, dcur * 2));
            else dcur = std::max(2, right + 1);
            if (right == 0 && mtp == 0) cool = 4;
        }
        bool stop_now = false;
        for (int i = 0; i < right && (int) g.out.size() < max; ++i) {
            g.out.push_back(d[i]);
            if (ends(d[i])) { stop_now = true; break; }
        }
        if (!flush()) break;
        if (stop_now || (int) g.out.size() >= max) break;
        next = nn;
    }
    g.write_s = now_s() - t1;
    return g;
}

static int serve(Engine& e, const std::string& path, int eos) {
    const int s = socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un a{};
    a.sun_family = AF_UNIX;
    std::snprintf(a.sun_path, sizeof a.sun_path, "%s", path.c_str());
    unlink(path.c_str());
    if (bind(s, (sockaddr*) &a, sizeof a) || listen(s, 4)) {
        std::perror("bind");
        return 1;
    }
    std::fprintf(stderr, "ink2_v1: posluša na %s\n", path.c_str());
    for (;;) {
        const int c = accept(s, nullptr, nullptr);
        if (c < 0) continue;
        std::string req;
        char buf[1 << 16];
        for (;;) {
            const ssize_t n = read(c, buf, sizeof buf);
            if (n <= 0) break;
            req.append(buf, (size_t) n);
            if (req.find('\n') != std::string::npos) break;
        }
        if (json_int(req, "exit", 0)) {
            close(c);
            break;
        }
        std::string rep;
        try {
            std::vector<int> tok, stop;
            for (long long x : json_ints(req, "tokens")) tok.push_back((int) x);
            for (long long x : json_ints(req, "stop")) stop.push_back((int) x);
            if (tok.empty()) throw std::runtime_error("no tokens");
            const int max = (int) std::min<long long>(json_int(req, "max", 256), std::max<long long>(1, (long long) e.ctx - (long long) tok.size()));
            // p15: "stream": 1 sends {"t": [ids]} lines as the tokens come; a failed write (the client left) stops the generation
            const std::function<bool(const int*, int)> out_line = [&](const int* t, int n) {
                std::string l = "{\"t\": [";
                for (int i = 0; i < n; ++i) l += (i ? ", " : "") + std::to_string(t[i]);
                l += "]}\n";
                return write(c, l.data(), l.size()) == (ssize_t) l.size();
            };
            const bool streaming = json_int(req, "stream", 0) != 0;
            Gen g = generate(e, tok, max, stop, json_int(req, "noeos", 0) ? -1 : eos, (int) json_int(req, "lookup", 0),
                             (int) json_int(req, "mtp", 0), streaming ? &out_line : nullptr);   // noeos: a fixed length
            rep = "{\"tokens\": [";
            for (size_t i = 0; i < g.out.size(); ++i) rep += (i ? ", " : "") + std::to_string(g.out[i]);
            char hd[512];
            std::snprintf(hd, sizeof hd,
                          "], \"prompt_n\": %zu, \"cached\": %d, \"read_s\": %.4f, \"write_s\": %.4f, \"drafted\": %d, \"accepted\": %d, "
                          "\"rounds\": %d, \"draft_s\": %.4f, \"verify_s\": %.4f}\n",
                          tok.size(), g.cached, g.read_s, g.write_s, g.drafted, g.accepted, g.rounds, g.draft_s, g.verify_s);
            rep += hd;
        } catch (const std::exception& ex) {
            e.reset();
            rep = std::string("{\"error\": \"") + ex.what() + "\"}\n";
        }
        (void) !write(c, rep.data(), rep.size());
        close(c);
    }
    close(s);
    unlink(path.c_str());
    return 0;
}

}  // namespace ink2

int main(int argc, char** argv) {
    using namespace ink2;
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    signal(SIGPIPE, SIG_IGN);   // p15: a client that left makes write() fail instead of ending the server   // lazy loading can sync a context on a kernel's first launch (a cross-card wait deadlocks)
    std::string model, tokens, dump, sock, logits_out, mtp;   // --mtp F: the DSpark drafter (dspark-Inkling-2-Mini-Q8_0.gguf)   // --logits F: the last row's logits after the prompt (logiti.bin format)
    int gen = 0, ctx = 32768;
    bool logits_all = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--model") model = nx();
        else if (a == "--tokens") tokens = nx();
        else if (a == "--gen") gen = std::stoi(nx());
        else if (a == "--dump") dump = nx();
        else if (a == "--ctx") ctx = std::stoi(nx());
        else if (a == "--serve") sock = nx();
        else if (a == "--logits") logits_out = nx();
        else if (a == "--logits-all") { logits_out = nx(); logits_all = true; }   // every row of the last window (writing path)
        else if (a == "--mtp") mtp = nx();
        else if (a == "--draft-vocab") g_draft_vocab = nx();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (model.empty()) { std::fprintf(stderr, "usage: ink2_v1 --model M.gguf (--tokens zetoni.bin [--gen N] [--dump DIR] | --serve SOCK) [--ctx C]\n"); return 1; }
    Engine e(model, ctx, mtp);
    const int eos = (int) e.g.i("tokenizer.ggml.eos_token_id");
    if (!sock.empty()) return serve(e, sock, eos);
    FILE* f = std::fopen(tokens.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot read %s\n", tokens.c_str()); return 1; }
    int32_t n = 0;
    if (std::fread(&n, 4, 1, f) != 1) return 1;
    std::vector<int> ids(n);
    if (std::fread(ids.data(), 4, n, f) != (size_t) n) return 1;
    std::fclose(f);
    if (!dump.empty()) {
        e.dumping = true;
        e.d_lout.resize(NSLOT);
        e.d_aout.resize(NSLOT);
        e.d_fout.resize(NSLOT);
    }
    const double t0 = now_s();
    const int nr = e.feed(ids.data(), n, !dump.empty() || logits_all);
    const double t1 = now_s();
    std::printf("poziv %d zetonov v %.3f s (%.1f tok/s)\n", n, t1 - t0, n / (t1 - t0));
    if (!dump.empty()) {
        for (int s = 0; s < NSLOT; ++s) {
            write_dump(dump + "/l_out-" + std::to_string(s) + ".bin", e.d_lout[s], N);
            write_dump(dump + "/attn_out-" + std::to_string(s) + ".bin", e.d_aout[s], N);
            write_dump(dump + "/ffn_out-" + std::to_string(s) + ".bin", e.d_fout[s], N);
        }
        FILE* lf = std::fopen((dump + "/logiti.bin").c_str(), "wb");
        const int32_t first = n - nr, nv = NVU;
        std::fwrite(&n, 4, 1, lf);
        std::fwrite(&nv, 4, 1, lf);
        std::fwrite(&first, 4, 1, lf);
        for (int r = 0; r < nr; ++r) std::fwrite(e.h_logits + (size_t) r * NV, 4, NVU, lf);
        std::fclose(lf);
        e.dumping = false;
    }
    if (!logits_out.empty()) {
        FILE* lf = std::fopen(logits_out.c_str(), "wb");
        const int32_t first = logits_all ? n - nr : n - 1, nv = NVU;
        std::fwrite(&n, 4, 1, lf);
        std::fwrite(&nv, 4, 1, lf);
        std::fwrite(&first, 4, 1, lf);
        for (int r = logits_all ? 0 : nr - 1; r < nr; ++r) std::fwrite(e.h_logits + (size_t) r * NV, 4, NVU, lf);
        std::fclose(lf);
    }
    int next = Engine::argmax(e.h_logits + (size_t) (nr - 1) * NV);
    std::printf("prvi %d\n", next);
    if (gen > 0) {
        std::vector<int> out{next};
        const double t2 = now_s();
        for (int i = 1; i < gen && next != eos; ++i) {
            e.feed(&next, 1, false);
            next = Engine::argmax(e.h_logits);
            out.push_back(next);
        }
        const double t3 = now_s();
        std::printf("pisanje %zu zetonov v %.3f s (%.2f tok/s):", out.size(), t3 - t2, (out.size() - 1) / std::max(1e-9, t3 - t2));
        for (int x : out) std::printf(" %d", x);
        std::printf("\n");
    }
    return 0;
}
