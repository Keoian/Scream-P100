    // ===================================================================================================== the prompt path
    // (included in struct Engine) A chunk of up to HY4_PFC tokens through every layer, both cards, then the last row's head.
    struct Pf {
        cublasHandle_t cb;
        float *R, *x, *xn, *qr, *q, *qabs, *qa, *kvr, *krows, *o, *av, *gt, *ao, *fo, *fo2, *lat, *gu_g, *gu_u, *hh, *ey, *moe, *logits,
            *dg, *du, *iq, *ik, *iv, *ig, *iw, *acc, *tmp, *erows, *ekv, *deq, *post;
        int *pos, *topi, *sel, *nsel, *groups, *valid;
        float* topw;
        __half* kbuf = nullptr;   // the window layers' K rows of a chunk (f16 like the cache)
        float *kf = nullptr, *swS = nullptr;   // v1l: the window attention's f32 rows and its tiles' scores (k_swa_softmax)
        float* selS = nullptr;                 // v1o: SELB tokens' scores in the sparse layers' list mode (swS holds their rows)
        const uint8_t** h_sptr = nullptr;      // v1q: each layer's sptr, page-locked, and on the card (k_pf_lists)
        __half *qh = nullptr, *ph = nullptr;   // v1t: the list mode's Q and P in f16 (its rows: swS as f16)
        const uint8_t** d_sptr = nullptr;
        PJob* jobs;
        int2* lists;
        float* xbuf[2];          // this card's exchange buffers (page-locked, mapped), by parity
        unsigned* xflag;         // PXB flags
        int ng_max = 0;
        // HY4_PF_STAGE: two staging buffers of NEXP expert halves (a layer's), filled a layer ahead on the copy stream
        uint8_t* stage[2] = {};
        cudaEvent_t st_ready[2], st_free[2];
        std::vector<const uint8_t*> sptr[2];   // per expert of the staged layer: its slot in the cache or its place in the buffer
        size_t staged_copies = 0;
        uint8_t* embq = nullptr;     // v2a: the chunk's token rows (Q8_0) on the card
        unsigned* nonfin = nullptr;  // HY4_PF_CHECK: non-finite values in R after each layer
    } pf[NC];
    bool pf_ready = false;
    bool pf_gemm_on = true;   // a request's "pfg" (serve)
    bool pf_selh_on = true;   // a request's "selh" (serve)
    // v2 (21 Nov), a request's "host", "selk", "skip", "xs" and "f16" (serve; the defaults from HY4_PF_HOST, HY4_PF_SELK,
    // HY4_PF_SKIP, HY4_PF_XS, HY4_PF_F16). dolg2 (11,048 tokens) read at 284-292 tok/s with v1; the exact switches 325-335, all f16 433-438
    //   host: the token rows expanded on the cards, the Engram rows on host threads, the indexer's groups on the card: no
    //         pageable copy (each waits for the stream). The same bits. v2f: two sets of these buffers; while a chunk runs, a host
    //         thread fills the other set for the next chunk (token rows, positions, the Engram rows in ESPLIT slices a layer)
    //   selk: the sparse layers' list attention in one pass a token (k_sel_attn), f32 math on the cache's f16 rows (v1t rounded
    //         Q and P to f16: KL 0.018 between the two). +6 %
    //   skip: the expert kernels' warps with no tokens of a tile skip the products (k_iq3_gemm3 / 5 <.., true>). The same bits, +7 %
    //   xs:   the cross-card sum in one kernel (k_xsum_pf). The same bits, no faster: off
    //   f16:  bits, f16 sums (not bit-equal to the f32 path; KL 0.024 for all three, the first token the same): 1 dense matrices
    //         as HGEMM (not the router, Engram or the indexer), 2 the experts in half2 (k_iq3_gemm5), 4 the window attention as
    //         HGEMM on its f16 rows
    bool pf_host = true, pf_skip = true, pf_xs = false;   // xs: measured no faster (21 Nov), off by default
    int pf_selk = 1;   // 1 k_sel_attn, 2 / 3 k_sel_attn2 at 2 / 1 blocks an SM (v2j), 0 v1t's path
    int pf_f16 = 0;
    static constexpr int NENG = 4, ESPLIT = 8;   // Engram layers; host threads a layer
    uint8_t* h_embq[2] = {};                     // v2a: page-locked, shared by the cards
    int* h_pos[2] = {};
    float* h_erows[2][NENG] = {};
    int* h_eok[2][NENG] = {};
    std::vector<std::shared_future<void>> eng_fut[2][NENG];
    int pf_buf = 0, prep_p0[2] = {-1, -1}, prep_C[2] = {0, 0};
    const int* prep_tok[2] = {};
    int pf_lo = 0;          // v1k: slots [pf_lo, slots) hold the prompt path's scratch during a chunk
    size_t pf_bytes = 0;

    // v2f: chunk (tok, p0, C) into buffer set b; the Engram rows on ESPLIT threads a layer (prefill waits for them a layer at a time)
    void pf_prepare(int b, const int* tok, int p0, int C) {
        for (int e = 0; e < NENG; ++e)
            for (auto& f : eng_fut[b][e]) f.wait();
        const size_t rb = M.tok->row_bytes();
        for (int t = 0; t < C; ++t) {
            std::memcpy(h_embq[b] + (size_t) t * rb, M.tok->data + (size_t) tok[t] * rb, rb);
            h_pos[b][t] = p0 + t;
        }
        for (int e = 0; e < NENG; ++e) {
            eng_fut[b][e].clear();
            for (int s = 0; s < ESPLIT; ++s) {
                const int a = C * s / ESPLIT, z = C * (s + 1) / ESPLIT;
                if (z > a)
                    eng_fut[b][e].push_back(std::async(std::launch::async, [this, b, e, tok, a, z]() {
                        engram_rows(e, tok + a, h_pos[b] + a, z - a, h_erows[b][e] + (size_t) a * E_W, h_eok[b][e] + a);
                    }).share());
            }
        }
        prep_p0[b] = p0;
        prep_C[b] = C;
        prep_tok[b] = tok;
    }

    void pf_alloc() {
        const int HM = 32;
        const size_t C = HY4_PFC;
        for (int b = 0; b < 2; ++b) {
            CK(cudaHostAlloc((void**) &h_embq[b], C * M.tok->row_bytes(), cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &h_pos[b], C * 4, cudaHostAllocPortable));
            for (int e = 0; e < NENG; ++e) {
                CK(cudaHostAlloc((void**) &h_erows[b][e], C * E_W * 4, cudaHostAllocPortable));
                CK(cudaHostAlloc((void**) &h_eok[b][e], C * 4, cudaHostAllocPortable));
            }
        }
        for (int c = 0; c < NC; ++c) {
            Card& K = M.cards[c];
            Pf& P = pf[c];
            CK(cudaSetDevice(c));
            if (cublasCreate(&P.cb) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate");
            cublasSetStream(P.cb, K.st);
            cublasSetMathMode(P.cb, CUBLAS_DEFAULT_MATH);
            // v1k: the chunk's scratch (~1 GB) lives in the top slots of the expert cache, taken from it for each chunk like the staging
            // slots (pf_take_slots, which also zeroes it): ~750 more slots for writing. Two passes: the sizes, then the pointers
            uint8_t* base = nullptr;
            for (int pass = 0; pass < 2; ++pass) {
            size_t off = 0;
            auto carve = [&](size_t bytes) -> void* {
                void* q = base ? (void*) (base + off) : nullptr;
                off += (bytes + 255) & ~(size_t) 255;
                return q;
            };
            auto f = [&](size_t n) { return (float*) carve(n * 4); };
            auto i32 = [&](size_t n) { return (int*) carve(n * 4); };
            P.ng_max = M.ctx / KPOOL + 1;
            P.R = f(C * D); P.x = f(C * N); P.xn = f(C * N); P.qr = f(C * QLR); P.q = f(C * HM * HDQ); P.qabs = f(C * HM * KVL);
            P.qa = f(C * HM * KROW); P.kvr = f(C * KROW); P.krows = f(C * KROW); P.kbuf = (__half*) f(((C + SWA) * KROW + 1) / 2); P.o = f(C * HM * KVL);
            P.av = f(C * HM * HDV); P.gt = f(C * HM * HDV); P.ao = f(C * N); P.fo = f(C * N); P.fo2 = f(C * N); P.lat = f(C * LAT);
            P.gu_g = f(C * TOPK * FFE_C); P.gu_u = f(C * TOPK * FFE_C); P.hh = f(C * TOPK * FFE_C); P.ey = f(C * TOPK * LAT);
            P.moe = f(C * LAT); P.logits = f(C * NEXP); P.dg = f(C * FFD_C); P.du = f(C * FFD_C); P.iq = f(C * IDXH * IDXD);
            P.ik = f(C * IDXD); P.iv = f(C * IDXD); P.ig = f(C * IDXD); P.iw = f(C * IDXH); P.acc = f(C * P.ng_max);
            P.tmp = f(C * P.ng_max); P.erows = f(C * E_W); P.ekv = f(C * (HC + 1) * N); P.deq = f((size_t) 8192 * 3072 + 1024);
            P.post = f(C * HC);
            P.kf = f((C + SWA) * KROW);
            P.swS = f(std::max((C + SWTT - 1) / SWTT * (size_t) SWTT * HM * SWT, (size_t) SELB * MAXSEL * KROW));
            P.selS = f((size_t) SELB * 16 * MAXSEL);
            P.d_sptr = (const uint8_t**) carve(sizeof(void*) * NL * NEXP);
            P.qh = (__half*) carve(sizeof(__half) * SELB * 16 * KROW);
            P.ph = (__half*) carve(sizeof(__half) * SELB * 16 * MAXSEL);
            P.pos = i32(C); P.topi = i32(C * TOPK); P.sel = i32(C * MAXSEL); P.nsel = i32(C); P.groups = i32(C); P.valid = i32(C);
            P.topw = f(C * TOPK);
            P.jobs = (PJob*) carve(sizeof(PJob) * NEXP * 3);
            P.lists = (int2*) carve(sizeof(int2) * C * TOPK);
            P.embq = (uint8_t*) carve(C * M.tok->row_bytes());
            P.nonfin = (unsigned*) carve(4 * sizeof(unsigned));   // [0] HY4_PF_CHECK, [2], [3] the compare hooks
            if (pass == 0) {
                const int n = (int) ((off + HALF_EXP - 1) / HALF_EXP);
                if (slots - n < 3 * NEXP) throw std::runtime_error("prompt path: too few cache slots for its scratch");
                pf_lo = slots - n;
                pf_bytes = off;
                base = K.pool + (size_t) pf_lo * HALF_EXP;
            }
            }
            for (int p = 0; p < 2; ++p) CK(cudaHostAlloc((void**) &P.xbuf[p], C * N * 4, cudaHostAllocMapped | cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &P.h_sptr, sizeof(void*) * NL * NEXP, cudaHostAllocPortable));
            CK(cudaHostAlloc((void**) &P.xflag, 64 * 4, cudaHostAllocMapped | cudaHostAllocPortable));
            std::memset(P.xflag, 0, 64 * 4);
            std::fprintf(stderr, "hy4_v1: card %d prompt path %d tokens a chunk, scratch in cache slots %d .. %d (%.2f GiB), VRAM in all %.2f GiB\n", c,
                         HY4_PFC, pf_lo, slots - 1, pf_bytes / 1073741824.0, K.vram / 1073741824.0);
        }
        if (pf_stage_on())
            for (int c = 0; c < NC; ++c) {
                Card& K = M.cards[c];
                Pf& P = pf[c];
                CK(cudaSetDevice(c));
                for (int b = 0; b < 2; ++b) {
                    // the first 2 x NEXP slots of the windows' cache (given back to it after each chunk, see pf_take_slots)
                    P.stage[b] = K.pool + (size_t) b * NEXP * HALF_EXP;
                    CK(cudaEventCreateWithFlags(&P.st_ready[b], cudaEventDisableTiming));
                    CK(cudaEventCreateWithFlags(&P.st_free[b], cudaEventDisableTiming));
                    CK(cudaEventRecord(P.st_free[b], K.st));
                    P.sptr[b].assign(NEXP, nullptr);
                }
                std::fprintf(stderr, "hy4_v1: card %d prompt-path staging in cache slots 0 .. 2 x 512 (2 x %.2f GiB), VRAM in all %.2f GiB\n", c,
                             NEXP * (double) HALF_EXP / 1073741824.0, K.vram / 1073741824.0);
            }
        pf_ready = true;
    }

    // the staging slots leave the cache for a chunk: whatever they held is no longer cached (slots of both cards: the same LRU)
    void pf_take_slots(int c) {
        if ((size_t) slots < 3 * (size_t) NEXP) throw std::runtime_error("prompt-path staging needs more than 1536 slots");
        Lru& U = lru[c];
        auto take = [&](int s) {   // empty, and first in line for the windows afterwards
            if (U.owner[s] >= 0) { U.slot_of.erase(U.owner[s]); U.owner[s] = -1; }
            U.lru.splice(U.lru.begin(), U.lru, U.lru_it[s]);
        };
        if (pf_stage_on())
            for (int s = 0; s < 2 * NEXP; ++s) take(s);
        for (int s = pf_lo; s < slots; ++s) take(s);
        CK(cudaSetDevice(c));
        CK(cudaMemsetAsync(M.cards[c].pool + (size_t) pf_lo * HALF_EXP, 0, pf_bytes, M.cards[c].st));
    }
    // after a chunk the staging slots hold the last two MoE layers' copied experts: they join the cache
    void pf_give_slots(int c) {
        Lru& U = lru[c];
        Pf& P = pf[c];
        for (int il = NL - 2; il < NL; ++il) {
            const int b = il & 1;
            for (int e = 0; e < NEXP; ++e) {
                const uint8_t* p = P.sptr[b][e];
                if (p < P.stage[b] || p >= P.stage[b] + (size_t) NEXP * HALF_EXP) continue;
                const int s = (int) ((p - M.cards[c].pool) / HALF_EXP), key = il * NEXP + e;
                U.owner[s] = key;
                U.slot_of[key] = s;
                U.lru.erase(U.lru_it[s]);
                U.lru_it[s] = U.lru.insert(U.lru.end(), s);
            }
        }
    }
    static bool pf_stage_on() {
        const char* v = getenv("HY4_PF_STAGE");
        return !v || std::string(v) != "0";
    }
    // card c: layer il's experts into staging buffer il & 1 on the copy stream, after the kernels that used that buffer last; the
    // ones in the windows' cache are read from their slots (the prompt path does not change the cache, so they stay)
    void pf_stage(int c, int il) {
        if (il >= NL || M.cards[c].L[il].dense) return;
        Card& K = M.cards[c];
        Pf& P = pf[c];
        const int b = il & 1;
        CK(cudaStreamWaitEvent(K.cst, P.st_free[b], 0));
        // v1o: one copy a run of missing experts (their halves are contiguous in RAM and in the buffer), runs closer than 4 cached
        // ones joined: 512 copies a layer kept the host thread waiting in cudaMemcpyAsync (~6 s a 2048-token chunk, nvprof)
        std::vector<char> need(NEXP);
        for (int e = 0; e < NEXP; ++e) {
            auto it = lru[c].slot_of.find(il * NEXP + e);
            need[e] = it == lru[c].slot_of.end();
            P.sptr[b][e] = need[e] ? P.stage[b] + (size_t) e * HALF_EXP : K.pool + (size_t) it->second * HALF_EXP;
            P.staged_copies += need[e];
        }
        for (int e = 0; e < NEXP;) {
            if (!need[e]) { ++e; continue; }
            int e1 = e + 1, gap = 0;
            for (int k = e + 1; k < NEXP && gap < 4; ++k) {
                if (need[k]) { e1 = k + 1; gap = 0; }
                else ++gap;
            }
            CK(cudaMemcpyAsync(P.stage[b] + (size_t) e * HALF_EXP, host_half(c, il, e), (size_t) (e1 - e) * HALF_EXP, cudaMemcpyHostToDevice, K.cst));
            e = e1;
        }
        CK(cudaEventRecord(P.st_ready[b], K.cst));
    }

    // v2c (f16 bit 1): y = x . W^T as HGEMM. W dequantized to f16 into deq, x to f16 and the f16 result into the attention's score
    // buffer swS (not in use between attention blocks): x at its start, y from F16_YOFF halves on (both up to C x 16384 halves)
    static constexpr size_t F16_YOFF = (size_t) HY4_PFC * 16384;
    void hgemm(int c, const Mat& W, const float* x, int ldx, float* y, int ldy, int C) {
        Pf& P = pf[c];
        cudaStream_t st = M.cards[c].st;
        __half *wh = (__half*) P.deq, *xh = (__half*) P.swS, *yh = xh + F16_YOFF;
        k_deq8h<<<1024, 256, 0, st>>>(W.q, (const __half*) W.d, (size_t) W.rows * W.cols, W.cols, wh);
        k_f2h_2d<<<std::min(C, 2048), 256, 0, st>>>(x, ldx, C, W.cols, xh);
        const __half one = __float2half(1.0f), zero = __float2half(0.0f);
        if (cublasGemmEx(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, W.rows, C, W.cols, &one, wh, CUDA_R_16F, W.cols, xh, CUDA_R_16F, W.cols, &zero, yh,
                         CUDA_R_16F, W.rows, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("hgemm");
        k_h2f_2d<<<std::min(C, 2048), 256, 0, st>>>(yh, C, W.rows, y, ldy);
    }
    void hgemm_heads(int c, const Mat& W, int H, const float* x, int ldx, int xs, float* y, int ldy, int ys, int C) {
        Pf& P = pf[c];
        cudaStream_t st = M.cards[c].st;
        __half *wh = (__half*) P.deq, *xh = (__half*) P.swS, *yh = xh + F16_YOFF;
        k_deq8h<<<1024, 256, 0, st>>>(W.q, (const __half*) W.d, (size_t) W.rows * W.cols, W.cols, wh);
        k_f2h_2d<<<std::min(C, 2048), 256, 0, st>>>(x, ldx, C, ldx, xh);   // the whole rows, the heads' strides as they are
        const int rh = W.rows / H;
        const __half one = __float2half(1.0f), zero = __float2half(0.0f);
        if (cublasGemmStridedBatchedEx(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, rh, C, W.cols, &one, wh, CUDA_R_16F, W.cols, (long long) rh * W.cols, xh,
                                       CUDA_R_16F, ldx, xs, &zero, yh, CUDA_R_16F, ldy, ys, H, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT) !=
            CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("hgemm_heads");
        k_h2f_2d<<<std::min(C, 2048), 256, 0, st>>>(yh, C, ldy, y, ldy);
    }

    // y [C][rows] = x [C][cols] . W^T (W dequantized to f32 first when it is Q8S); h16: may go through hgemm (f16 bit 1)
    void sgemm(int c, const Mat& W, const float* x, int ldx, float* y, int ldy, int C, bool h16 = false) {
        if (h16 && (pf_f16 & 1) && W.type == 8) return hgemm(c, W, x, ldx, y, ldy, C);
        Pf& P = pf[c];
        Card& K = M.cards[c];
        const float* w = (const float*) W.d;
        if (W.type == 8) {
            const size_t n = (size_t) W.rows * W.cols;
            if (n > (size_t) 8192 * 3072 + 1024) throw std::runtime_error("sgemm: matrix larger than the dequant buffer");
            k_deq8<<<1024, 256, 0, K.st>>>(W.q, (const __half*) W.d, n, W.cols, P.deq);
            w = P.deq;
        } else if (W.type != 0) {
            throw std::runtime_error("sgemm: F32 or Q8_0 only");
        }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, W.rows, C, W.cols, &one, w, W.cols, x, ldx, &zero, y, ldy) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasSgemm");
    }
    // per head h < H: y_h [C][rows_h] = x_h [C][cols] . W_h^T, W [H][rows_h][cols] (Q8S), x_h at x + h * xs (row stride ldx), y_h at
    // y + h * ys (row stride ldy)
    void sgemm_heads(int c, const Mat& W, int H, const float* x, int ldx, int xs, float* y, int ldy, int ys, int C) {
        if (pf_f16 & 1) return hgemm_heads(c, W, H, x, ldx, xs, y, ldy, ys, C);
        Pf& P = pf[c];
        Card& K = M.cards[c];
        const size_t n = (size_t) W.rows * W.cols;
        k_deq8<<<1024, 256, 0, K.st>>>(W.q, (const __half*) W.d, n, W.cols, P.deq);
        const int rh = W.rows / H;
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemmStridedBatched(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, rh, C, W.cols, &one, P.deq, W.cols, (long long) rh * W.cols, x, ldx, xs,
                                      &zero, y, ldy, ys, H) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasSgemmStridedBatched");
    }
    void xsum_pf(int c, float* y, int C) {
        Card& K = M.cards[c];
        const unsigned s = seq[c]++;
        const int par = s & 1;
        if (pf_xs) {   // v2i: one kernel, both directions of the link at once (a request's "xs", HY4_PF_XS)
            k_xsum_pf<<<PXB2, 1024, 0, K.st>>>(y, C * N, pf[c].xbuf[par], pf[c].xflag, pf[1 - c].xbuf[par], pf[1 - c].xflag, s);
            return;
        }
        k_xput_pf<<<PXB, 1024, 0, K.st>>>(y, C * N, pf[c].xbuf[par], pf[c].xflag, s);
        k_xget_pf<<<PXB, 1024, 0, K.st>>>(y, C * N, pf[1 - c].xbuf[par], pf[1 - c].xflag, s);
    }

    // a prompt chunk at positions p0 .. p0 + C - 1 (C > MAXT); returns the last row's argmax
    int prefill(const int* tok, int p0, int C, const int* ntok = nullptr, int nC = 0) {   // ntok, nC: the next chunk (v2f)
        if (!pf_ready) pf_alloc();
        if (trace) {   // the prompt path: its staging empties slots 0 .. 2 NEXP - 1
            const int hd[2] = {-1, C};
            std::fwrite(hd, 4, 2, trace);
        }
        if (getenv("HY4_DEBUG")) std::fprintf(stderr, "prefill: %d tokens at %d (HY4_PFC %d)\n", C, p0, (int) HY4_PFC);
        w_tok = tok;
        w_T = C;
        w_pos.resize(C);
        for (int t = 0; t < C; ++t) {
            w_pos[t] = p0 + t;
            if ((int) hist.size() <= p0 + t) hist.resize((size_t) p0 + t + 1);
            hist[(size_t) p0 + t] = tok[t];
        }
        const size_t rb = M.tok->row_bytes();
        std::future<void> next;
        if (pf_host) {   // v2a: the rows as they are into page-locked memory (k_embed_q8 expands them), the Engram rows on threads
            const int b = pf_buf;
            if (prep_p0[b] != p0 || prep_C[b] != C || prep_tok[b] != tok) pf_prepare(b, tok, p0, C);
            if (ntok && nC > 0) {   // v2f: the next chunk's history now (its Engram rows read it), its buffers on a thread
                if (hist.size() < (size_t) (p0 + C + nC)) hist.resize((size_t) p0 + C + nC);
                for (int t = 0; t < nC; ++t) hist[(size_t) p0 + C + t] = ntok[t];
                next = std::async(std::launch::async, [this, b, ntok, p0, C, nC]() { pf_prepare(1 - b, ntok, p0 + C, nC); });
            }
        } else {
        w_emb.assign((size_t) C * D, 0.0f);
        for (int t = 0; t < C; ++t) {
            const uint8_t* r = M.tok->data + (size_t) tok[t] * rb;
            for (int b = 0; b < N / 32; ++b) {
                __half hh;
                std::memcpy(&hh, r + b * 34, 2);
                const float d = __half2float(hh);
                for (int j = 0; j < 32; ++j) {
                    const float v = d * (float) (int8_t) r[b * 34 + 2 + j];
                    for (int cc = 0; cc < HC; ++cc) w_emb[(size_t) t * D + cc * N + b * 32 + j] = v;
                }
            }
        }
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int c = 0; c < NC; ++c) pf_take_slots(c);
        card1(2);
        run_card_pf(0);
        card1_wait();
        if (pf_stage_on()) for (int c = 0; c < NC; ++c) pf_give_slots(c);
        prep_p0[pf_buf] = -1;   // used: a later request with the same pointer, start and length must not take it
        if (next.valid()) {   // long done (a chunk takes seconds): nothing of it may outlive the request's tokens
            next.get();
            for (int e = 0; e < NENG; ++e)
                for (auto& f : eng_fut[1 - pf_buf][e]) f.wait();
            pf_buf = 1 - pf_buf;
        }
        if (getenv("HY4_DEBUG")) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "prefill: %d tokens at %d in %.2f s (%.1f tok/s), hits %zu misses %zu\n", C, p0, s, C / s, hits(), misses());
        }
        return out_v[1][0] > out_v[0][0] ? out_i[1][0] + NV_C : out_i[0][0];
    }

    void run_card_pf(int c) {
        Card& K = M.cards[c];
        Pf& P = pf[c];
        const int C = w_T;
        CK(cudaSetDevice(c));
        if (pf_host) {
            const size_t rb = M.tok->row_bytes();
            CK(cudaMemcpyAsync(P.pos, h_pos[pf_buf], C * 4, cudaMemcpyHostToDevice, K.st));
            CK(cudaMemcpyAsync(P.embq, h_embq[pf_buf], C * rb, cudaMemcpyHostToDevice, K.st));
            k_embed_q8<<<C, 256, 0, K.st>>>(P.embq, rb, P.R);
        } else {
            CK(cudaMemcpyAsync(P.pos, w_pos.data(), C * 4, cudaMemcpyHostToDevice, K.st));
            CK(cudaMemcpyAsync(P.R, w_emb.data(), w_emb.size() * 4, cudaMemcpyHostToDevice, K.st));
        }
        static const bool check = getenv("HY4_PF_CHECK") != nullptr;
        if (check) CK(cudaMemsetAsync(P.nonfin, 0, sizeof(unsigned), K.st));
        if (pf_stage_on()) pf_stage(c, 1);
        for (int il = 0; il < NL; ++il) {
            if (pf_stage_on() && il >= 1) pf_stage(c, il + 1);   // the next layer's copies run under this layer's work
            layer_pf(c, il);
            if (check) k_nonfinite<<<256, 256, 0, K.st>>>(P.R, (size_t) C * D, P.nonfin);
        }
        // the head on the last row (the window buffers: R of one row)
        CK(cudaMemcpyAsync(K.R, P.R + (size_t) (C - 1) * D, D * 4, cudaMemcpyDeviceToDevice, K.st));
        hc_pre(K.R, K.head_fn, K.head_scale, K.head_base, HC, M.eps, M.hc_eps, 0.0f, K.x, nullptr, 1, K.st);
        rms(K.x, N, K.out_norm, K.xn, N, 1, N, M.eps, K.st);
        gemv(K.out, K.xn, N, K.head_x, NV_C, 1, K.st);
        k_argmax<<<1, 1024, 0, K.st>>>(K.head_x, NV_C, K.best, K.bestv);
        CK(cudaMemcpyAsync(out_i[c], K.best, 4, cudaMemcpyDeviceToHost, K.st));
        CK(cudaMemcpyAsync(out_v[c], K.bestv, 4, cudaMemcpyDeviceToHost, K.st));
        CK(cudaStreamSynchronize(K.st));
        if (check) {
            unsigned nf = 0;
            CK(cudaMemcpy(&nf, P.nonfin, sizeof nf, cudaMemcpyDeviceToHost));
            if (nf) std::fprintf(stderr, "hy4_v1: card %d, chunk at %d: %u non-finite values in R (f16 %d)\n", c, w_pos[0], nf, pf_f16);
        }
        // HY4_PF_LOGITS=FILE: the chunk's last-row logits, card c's half of the vocabulary, into FILE.c<c> (overwritten each chunk)
        if (const char* lf = getenv("HY4_PF_LOGITS")) {
            std::vector<float> lg(NV_C);
            CK(cudaMemcpy(lg.data(), K.head_x, NV_C * 4, cudaMemcpyDeviceToHost));
            if (FILE* f = std::fopen((std::string(lf) + ".c" + std::to_string(c)).c_str(), "wb")) {
                std::fwrite(lg.data(), 4, NV_C, f);
                std::fclose(f);
            }
        }
    }

    void layer_pf(int c, int il) {
        Card& K = M.cards[c];
        Pf& P = pf[c];
        const CLayer& l = K.L[il];
        cudaStream_t st = K.st;
        const int C = w_T, H = l.H, p0 = w_pos[0], plast = w_pos[C - 1];
        const int* hp = w_pos.data();
        if (c == 0) prof.start();
        if (l.engram && pf_host) {   // v2a: computed since the chunk's start (v2f: during the chunk before)
            for (auto& f : eng_fut[pf_buf][l.e_index]) f.wait();
            CK(cudaMemcpyAsync(P.erows, h_erows[pf_buf][l.e_index], (size_t) C * E_W * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(P.valid, h_eok[pf_buf][l.e_index], C * 4, cudaMemcpyHostToDevice, st));
            k_engram_rows<<<dim3(C, E_COLS), 128, 0, st>>>(P.erows, l.e_lnorm);
            sgemm(c, l.e_kv, P.erows, E_W, P.ekv, (HC + 1) * N, C);
            k_engram_apply<<<C, 256, 0, st>>>(P.R, P.ekv, l.e_kv_b, l.e_gate, l.e_gamma, l.e_hcs, P.valid);
        } else if (l.engram) {
            std::vector<float> rows;
            std::vector<int> ok;
            engram_rows(l.e_index, w_tok, hp, C, rows, ok);
            CK(cudaMemcpyAsync(P.erows, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(P.valid, ok.data(), C * 4, cudaMemcpyHostToDevice, st));
            k_engram_rows<<<dim3(C, E_COLS), 128, 0, st>>>(P.erows, l.e_lnorm);
            sgemm(c, l.e_kv, P.erows, E_W, P.ekv, (HC + 1) * N, C);
            k_engram_apply<<<C, 256, 0, st>>>(P.R, P.ekv, l.e_kv_b, l.e_gate, l.e_gamma, l.e_hcs, P.valid);
        }
        hc_pre(P.R, l.hc_attn_fn, l.hc_attn_scale, l.hc_attn_base, 2 * HC, M.eps, M.hc_eps, M.hc_mag, P.x, P.post, C, st);
        rms(P.x, N, l.attn_norm, P.xn, N, C, N, M.eps, st);
        sgemm(c, l.q_a, P.xn, N, P.qr, QLR, C, true);
        rms(P.qr, QLR, l.q_a_norm, P.qr, QLR, C, QLR, M.eps, st);
        sgemm(c, l.q_b, P.qr, QLR, P.q, H * HDQ, C, true);
        k_rope<<<dim3(C, H), ROT / 2, 0, st>>>(P.q, H * HDQ, H, HDQ, NOPE, P.pos, l.rope_base);
        sgemm(c, l.kv_a, P.xn, N, P.kvr, KROW, C, true);
        k_rope<<<dim3(C, 1), ROT / 2, 0, st>>>(P.kvr, KROW, 1, KROW, KVL, P.pos, l.rope_base);
        k_kv_rows<<<C, 256, 0, st>>>(P.kvr, l.kv_a_norm, M.eps, P.krows);
        const __half* Kp;
        int kbase;
        if (l.swa) {   // the window's 511 rows before the chunk from the ring, then the chunk's; the ring takes the chunk's after
            const int npre = std::min(SWA - 1, p0);
            if (npre) k_swa_gather<<<npre, 256, 0, st>>>(l.kv, p0, npre, P.kbuf);
            k_f2h<<<256, 256, 0, st>>>(P.krows, P.kbuf + (size_t) npre * KROW, (size_t) C * KROW);
            k_rows_to_cache<<<C, 256, 0, st>>>(P.krows, l.kv, P.pos, 1);
            Kp = P.kbuf;
            kbase = p0 - npre;
        } else {
            k_rows_to_cache<<<C, 256, 0, st>>>(P.krows, l.kv, P.pos, 0);
            Kp = l.kv;
            kbase = 0;
        }
        sgemm_heads(c, l.k_b, H, P.q, H * HDQ, HDQ, P.qabs, H * KVL, KVL, C);
        k_q_assemble<<<dim3(C, H), 256, 0, st>>>(P.q, P.qabs, P.qa, H);
        if (c == 0) prof.stop("pf_projekcije", st);
        bool use_list = false;
        if (l.sparse_idx) {
            sgemm(c, l.idx_k, P.xn, N, P.ik, IDXD, C);
            rms(P.ik, IDXD, l.idx_k_norm, P.ik, IDXD, C, IDXD, M.eps, st);
            sgemm(c, l.idx_cv, P.ik, IDXD, P.iv, IDXD, C);
            sgemm(c, l.idx_cg, P.ik, IDXD, P.ig, IDXD, C);
            k_idx_store<<<C, IDXD, 0, st>>>(P.iv, P.ig, l.ivg, P.pos);
            if (pf_host) {   // v2a: the chunk's positions are p0 .. plast, so its complete groups are consecutive: made on the card
                const int pa = p0 + (KPOOL - 1 - p0 % KPOOL), pb = plast - (plast + 1) % KPOOL;   // the first and last p % 4 == 3
                const int n = pa <= plast ? (pb - pa) / KPOOL + 1 : 0;
                if (n > 0) {
                    k_iota<<<(n + 255) / 256, 256, 0, st>>>(P.groups, pa / KPOOL, n);
                    k_idx_pool<<<n, IDXD, 0, st>>>(l.ivg, l.idx_ape, l.idx_cnorm, M.eps, l.rope_base, P.groups, l.pooled);
                }
            } else {
            std::vector<int> done;
            for (int t = 0; t < C; ++t)
                if (hp[t] % KPOOL == KPOOL - 1) done.push_back(hp[t] / KPOOL);
            if (!done.empty()) {
                CK(cudaMemcpyAsync(P.groups, done.data(), done.size() * 4, cudaMemcpyHostToDevice, st));
                k_idx_pool<<<(int) done.size(), IDXD, 0, st>>>(l.ivg, l.idx_ape, l.idx_cnorm, M.eps, l.rope_base, P.groups, l.pooled);
            }
            }
            const int ng = plast / KPOOL;   // the most visible groups of the chunk's tokens
            if (ng > TOPPOOL) {
                use_list = true;
                sgemm(c, l.idx_qb, P.qr, QLR, P.iq, IDXH * IDXD, C);
                k_rope<<<dim3(C, IDXH), ROT / 2, 0, st>>>(P.iq, IDXH * IDXD, IDXH, IDXD, IDXD - ROT, P.pos, l.rope_base);
                sgemm(c, l.idx_proj, P.xn, N, P.iw, IDXH, C);
                k_scale<<<64, 256, 0, st>>>(P.iw, (size_t) C * IDXH, 1.0f / sqrtf((float) (IDXD * IDXH)));
                const float one = 1.0f, zero = 0.0f;
                for (int h = 0; h < IDXH; ++h) {
                    if (cublasSgemm(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, ng, C, IDXD, &one, l.pooled, IDXD, P.iq + h * IDXD, IDXH * IDXD, &zero, P.tmp, ng) !=
                        CUBLAS_STATUS_SUCCESS)
                        throw std::runtime_error("indexer sgemm");
                    k_idx_acc<<<dim3((ng + 255) / 256, C), 256, 0, st>>>(P.tmp, ng, P.iw, h, P.pos, P.acc);
                }
                k_idx_topk<<<C, 256, 0, st>>>(P.acc, ng, P.pos, P.sel, P.nsel, 1);
            }
        }
        const float scale = 1.0f / sqrtf((float) HDQ);
        // the cuBLAS attention: HY4_PF_SWAG=0 (window layers) or HY4_PF_SPG=0 (sparse) turn it off, a request's "pfg": 0 both
        static const bool swa_env = !getenv("HY4_PF_SWAG") || std::string(getenv("HY4_PF_SWAG")) != "0";
        static const bool sp_env = !getenv("HY4_PF_SPG") || std::string(getenv("HY4_PF_SPG")) != "0";
        const bool swa_gemm = swa_env && pf_gemm_on, sp_gemm = sp_env && pf_gemm_on;
        static const bool sel_env = !getenv("HY4_PF_SELH") || std::string(getenv("HY4_PF_SELH")) != "0";   // v1t: 0 = the f32 list mode
        const bool sel_h = sel_env && pf_selh_on;
        if (l.swa && swa_gemm && (pf_f16 & 4)) {   // v2e: HGEMM on the f16 rows; Q into qabs (free after k_q_assemble), S then O in swS
            const int npre = p0 - kbase, nt = (C + SWTT - 1) / SWTT, Mq = SWTT * H;
            __half *kfh = (__half*) P.kf, *qh = (__half*) P.qabs, *Sh = (__half*) P.swS, *Oh = Sh + (size_t) nt * Mq * SWT;
            k_kfh_fill<<<C + SWA, 72, 0, st>>>(P.kbuf, npre, npre + C, kfh);
            k_f2h<<<1024, 256, 0, st>>>(P.qa, qh, (size_t) C * H * KROW);
            const __half hs = __float2half(scale), one = __float2half(1.0f), zero = __float2half(0.0f);
            if (cublasGemmStridedBatchedEx(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, SWT, Mq, KROW, &hs, kfh, CUDA_R_16F, KROW, (long long) SWTT * KROW, qh,
                                           CUDA_R_16F, KROW, (long long) Mq * KROW, &zero, Sh, CUDA_R_16F, SWT, (long long) Mq * SWT, nt,
                                           CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("window attention f16: scores");
            k_swa_softmax_h<<<(nt * Mq + 7) / 8, 256, 0, st>>>(Sh, Mq, H, C, p0, nt * Mq, l.sinks);
            if (cublasGemmStridedBatchedEx(P.cb, CUBLAS_OP_N, CUBLAS_OP_N, KVL, Mq, SWT, &one, kfh, CUDA_R_16F, KROW, (long long) SWTT * KROW, Sh,
                                           CUDA_R_16F, SWT, (long long) Mq * SWT, &zero, Oh, CUDA_R_16F, KVL, (long long) Mq * KVL, nt,
                                           CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("window attention f16: values");
            k_h2f_2d<<<1024, 256, 0, st>>>(Oh, C * H, KVL, P.o, KVL);
        } else if (l.swa && swa_gemm) {
            const int npre = p0 - kbase, nt = (C + SWTT - 1) / SWTT, Mq = SWTT * H;
            k_kf_fill<<<C + SWA, 256, 0, st>>>(P.kbuf, npre, npre + C, P.kf);
            const float one = 1.0f, zero = 0.0f;
            if (cublasSgemmStridedBatched(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, SWT, Mq, KROW, &one, P.kf, KROW, (long long) SWTT * KROW, P.qa, KROW,
                                          (long long) Mq * KROW, &zero, P.swS, SWT, (long long) Mq * SWT, nt) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("window attention: scores");
            k_swa_softmax<<<(nt * Mq + 7) / 8, 256, 0, st>>>(P.swS, Mq, H, C, p0, nt * Mq, l.sinks, scale);
            if (cublasSgemmStridedBatched(P.cb, CUBLAS_OP_N, CUBLAS_OP_N, KVL, Mq, SWT, &one, P.kf, KROW, (long long) SWTT * KROW, P.swS, SWT,
                                          (long long) Mq * SWT, &zero, P.o, KVL, (long long) Mq * KVL, nt) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("window attention: values");
        } else if (!l.swa && sp_gemm && !use_list) {   // v1o: before position 2048, tiles against all positions 0 .. np - 1
            const int np = p0 + C, nt = (C + SWTT - 1) / SWTT, Mq = SWTT * H;
            k_kf_range<<<np, 128, 0, st>>>(l.kv, P.kf);
            const size_t cap = std::max((C + SWTT - 1) / SWTT * (size_t) SWTT * 32 * SWT, (size_t) SELB * MAXSEL * KROW);
            const float one = 1.0f, zero = 0.0f;
            for (int i0 = 0; i0 < nt;) {
                int ng = 1;
                while (i0 + ng < nt && (size_t) (ng + 1) * Mq * std::min(np, p0 + SWTT * (i0 + ng + 1)) <= cap) ++ng;
                const int len = std::min(np, p0 + SWTT * (i0 + ng));
                if (cublasSgemmStridedBatched(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, len, Mq, KROW, &one, P.kf, KROW, 0, P.qa + (size_t) i0 * Mq * KROW, KROW,
                                              (long long) Mq * KROW, &zero, P.swS, len, (long long) Mq * len, ng) != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("sparse attention: scores");
                k_causal_softmax<<<(ng * Mq + 7) / 8, 256, 0, st>>>(P.swS, len, Mq, H, C, p0, i0, ng * Mq, l.sinks, scale);
                if (cublasSgemmStridedBatched(P.cb, CUBLAS_OP_N, CUBLAS_OP_N, KVL, Mq, len, &one, P.kf, KROW, 0, P.swS, len, (long long) Mq * len, &zero,
                                              P.o + (size_t) i0 * Mq * KVL, KVL, (long long) Mq * KVL, ng) != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("sparse attention: values");
                i0 += ng;
            }
        } else if (!l.swa && sp_gemm && use_list && pf_selk && H == 16 && !getenv("HY4_PF_SELCMP")) {   // v2b: one pass a token
            if (pf_selk == 2) k_sel_attn2<2><<<C, 32 * H, 0, st>>>(P.qa, l.kv, P.sel, P.nsel, H, l.sinks, scale, P.o);
            else if (pf_selk == 3) k_sel_attn2<1><<<C, 32 * H, 0, st>>>(P.qa, l.kv, P.sel, P.nsel, H, l.sinks, scale, P.o);
            else k_sel_attn<<<C, 32 * H, 0, st>>>(P.qa, l.kv, P.sel, P.nsel, H, l.sinks, scale, P.o);
        } else if (!l.swa && sp_gemm && use_list && sel_h) {   // v1t: the same in f16 (rows, Q, P), f32 sums
            const float one = 1.0f, zero = 0.0f;
            __half* kg = (__half*) P.swS;
            for (int t0 = 0; t0 < C; t0 += SELB) {
                const int nb = std::min(SELB, C - t0);
                k_sel_gather_h<<<dim3(MAXSEL, SELB), 96, 0, st>>>(l.kv, P.sel, P.nsel, t0, C, kg);
                k_f2h<<<64, 256, 0, st>>>(P.qa + (size_t) t0 * H * KROW, P.qh, (size_t) nb * H * KROW);
                if (cublasGemmStridedBatchedEx(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, MAXSEL, H, KROW, &one, kg, CUDA_R_16F, KROW, (long long) MAXSEL * KROW,
                                               P.qh, CUDA_R_16F, KROW, (long long) H * KROW, &zero, P.selS, CUDA_R_32F, MAXSEL, (long long) MAXSEL * H, nb,
                                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("sparse list attention f16: scores");
                k_sel_softmax_h<<<(SELB * H + 7) / 8, 256, 0, st>>>(P.selS, H, t0, C, P.nsel, l.sinks, scale, P.ph);
                if (cublasGemmStridedBatchedEx(P.cb, CUBLAS_OP_N, CUBLAS_OP_N, KVL, H, MAXSEL, &one, kg, CUDA_R_16F, KROW, (long long) MAXSEL * KROW,
                                               P.ph, CUDA_R_16F, MAXSEL, (long long) MAXSEL * H, &zero, P.o + (size_t) t0 * H * KVL, CUDA_R_32F, KVL,
                                               (long long) H * KVL, nb, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("sparse list attention f16: values");
            }
        } else if (!l.swa && sp_gemm && use_list) {   // v1o: SELB tokens at a time, each against its own gathered rows
            const float one = 1.0f, zero = 0.0f;
            for (int t0 = 0; t0 < C; t0 += SELB) {
                const int nb = std::min(SELB, C - t0);
                k_sel_gather<<<dim3(MAXSEL, SELB), 128, 0, st>>>(l.kv, P.sel, P.nsel, t0, C, P.swS);
                if (cublasSgemmStridedBatched(P.cb, CUBLAS_OP_T, CUBLAS_OP_N, MAXSEL, H, KROW, &one, P.swS, KROW, (long long) MAXSEL * KROW,
                                              P.qa + (size_t) t0 * H * KROW, KROW, (long long) H * KROW, &zero, P.selS, MAXSEL, (long long) MAXSEL * H,
                                              nb) != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("sparse list attention: scores");
                k_sel_softmax<<<(SELB * H + 7) / 8, 256, 0, st>>>(P.selS, H, t0, C, P.nsel, l.sinks, scale);
                if (cublasSgemmStridedBatched(P.cb, CUBLAS_OP_N, CUBLAS_OP_N, KVL, H, MAXSEL, &one, P.swS, KROW, (long long) MAXSEL * KROW, P.selS, MAXSEL,
                                              (long long) MAXSEL * H, &zero, P.o + (size_t) t0 * H * KVL, KVL, (long long) H * KVL, nb) != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("sparse list attention: values");
            }
        } else if (use_list) k_mla_pf<1, 8><<<dim3(C, H / 8), 256, 0, st>>>(P.qa, Kp, kbase, P.pos, C, H, 0, P.sel, P.nsel, l.sinks, scale, P.o);
        else k_mla_pf<4, 2><<<dim3((C + 3) / 4, H / 2), 256, 0, st>>>(P.qa, Kp, kbase, P.pos, C, H, l.swa ? 1 : 0, nullptr, nullptr, l.sinks, scale, P.o);
        if (getenv("HY4_PF_SELCMP") && !l.swa && use_list && H == 16 && c == 0) {   // k_sel_attn (into av) against the path above (o)
            CK(cudaMemsetAsync(P.nonfin + 2, 0, 8, st));
            if (pf_selk >= 2) k_sel_attn2<1><<<C, 32 * H, 0, st>>>(P.qa, l.kv, P.sel, P.nsel, H, l.sinks, scale, P.av);
            else k_sel_attn<<<C, 32 * H, 0, st>>>(P.qa, l.kv, P.sel, P.nsel, H, l.sinks, scale, P.av);
            k_cmp_max<<<256, 256, 0, st>>>(P.av, P.o, (size_t) C * H * KVL, P.nonfin + 2);
            unsigned r[2];
            CK(cudaMemcpyAsync(r, P.nonfin + 2, 8, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            float d, m;
            std::memcpy(&d, r, 4);
            std::memcpy(&m, r + 1, 4);
            std::fprintf(stderr, "selcmp layer %d at %d: max |k_sel_attn - v1| %.3e, max |v1| %.3e\n", il, p0, d, m);
        }
        if (c == 0) prof.stop(l.swa ? "pf_mla_okno" : "pf_mla_redka", st);
        sgemm_heads(c, l.v_b, H, P.o, H * KVL, KVL, P.av, H * HDV, HDV, C);
        sgemm(c, l.gate, P.xn, N, P.gt, H * HDV, C, true);
        k_sigmul<<<dim3(64, C), 256, 0, st>>>(P.av, P.gt, H * HDV);
        sgemm(c, l.wo, P.av, H * HDV, P.ao, N, C, true);
        if (c == 0) prof.stop("pf_izhod_pozornosti", st);
        xsum_pf(c, P.ao, C);
        rms(P.ao, N, l.post_attn_norm, P.ao, N, C, N, M.eps, st);
        k_hc_post<<<dim3(C, HCB), 256, 0, st>>>(P.R, P.ao, P.post);
        hc_pre(P.R, l.hc_ffn_fn, l.hc_ffn_scale, l.hc_ffn_base, 2 * HC, M.eps, M.hc_eps, M.hc_mag, P.x, P.post, C, st);
        rms(P.x, N, l.ffn_norm, P.xn, N, C, N, M.eps, st);
        if (c == 0) prof.stop("pf_izmenjava_hc", st);
        if (l.dense) {
            sgemm(c, l.ffn_gate, P.xn, N, P.dg, FFD_C, C, true);
            sgemm(c, l.ffn_up, P.xn, N, P.du, FFD_C, C, true);
            k_swiglu<<<dim3(64, C), 256, 0, st>>>(P.dg, P.du, P.dg, FFD_C, 0.0f);
            sgemm(c, l.ffn_down, P.dg, FFD_C, P.fo, N, C, true);
        } else {
            sgemm(c, l.gate_inp, P.xn, N, P.logits, NEXP, C);
            k_route4<<<C, 256, 0, st>>>(P.logits, l.probs_b, M.wscale, P.topi, P.topw);   // k_route's bits
            static const bool dev_lists = !getenv("HY4_PF_HOSTLISTS");   // v1q: the lists on the card (HY4_PF_HOSTLISTS=1: v1's)
            const bool dev = dev_lists && pf_stage_on();
            int nu = NEXP;
            if (dev) {
                std::memcpy(P.h_sptr + (size_t) il * NEXP, P.sptr[il & 1].data(), NEXP * sizeof(void*));
                CK(cudaMemcpyAsync(P.d_sptr + (size_t) il * NEXP, P.h_sptr + (size_t) il * NEXP, NEXP * sizeof(void*), cudaMemcpyHostToDevice, st));
                k_pf_lists<<<1, 1024, 0, st>>>(P.topi, C, P.d_sptr + (size_t) il * NEXP, HALF_GU, P.jobs, P.lists);
            } else {
            std::vector<int> ids((size_t) C * TOPK);
            CK(cudaMemcpyAsync(ids.data(), P.topi, ids.size() * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (c == 0) prof.stop("pf_usmerjevalnik", st);
            // per expert: its (token, pair) entries; the experts in ascending order (the same on both cards)
            std::vector<std::vector<int2>> by(NEXP);
            for (int j = 0; j < (int) ids.size(); ++j) by[ids[j]].push_back(make_int2(j / TOPK, j));
            std::vector<int> uniq;
            for (int e = 0; e < NEXP; ++e)
                if (!by[e].empty()) uniq.push_back(e);
            // the experts: staged a layer ahead (HY4_PF_STAGE, the default: a cached one from its slot, the rest copied into this layer's
            // staging buffer while the layer before ran), or through the windows' LRU (HY4_PF_STAGE=0)
            const bool staged = pf_stage_on();
            std::vector<int> slot;
            std::vector<char> miss;
            if (!staged) experts(c, il, uniq, slot, miss);
            std::vector<int2> lists;
            std::vector<PJob> jg, ju, jd;
            for (size_t u = 0; u < uniq.size(); ++u) {
                const int off = (int) lists.size(), n = (int) by[uniq[u]].size();
                lists.insert(lists.end(), by[uniq[u]].begin(), by[uniq[u]].end());
                const uint8_t* base = staged ? P.sptr[il & 1][uniq[u]] : K.pool + (size_t) slot[u] * HALF_EXP;
                jg.push_back({base, off, n});
                ju.push_back({base + HALF_GU, off, n});
                jd.push_back({base + 2 * HALF_GU, off, n});
            }
            std::vector<PJob> jall = jg;
            jall.insert(jall.end(), ju.begin(), ju.end());
            jall.insert(jall.end(), jd.begin(), jd.end());
            nu = (int) uniq.size();
            CK(cudaMemcpyAsync(P.lists, lists.data(), lists.size() * sizeof(int2), cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(P.jobs, jall.data(), jall.size() * sizeof(PJob), cudaMemcpyHostToDevice, st));
            }
            const bool staged = pf_stage_on();
            sgemm(c, l.r_down, P.xn, N, P.lat, LAT, C, true);
            CK(cudaStreamWaitEvent(st, staged ? P.st_ready[il & 1] : K.cev, 0));
            // HY4_PFIQ: 3 (default) one row a warp, tokens through shared memory; 2 two rows a warp from registers; 1 the first kernel
            // 4 (v1l, the default): k_iq3_gemm2, tiles of 128 rows by 32 tokens
            // 5 (v1s, the default): k_iq3_gemm3, tiles of 128 rows by 64 tokens
            const int iqv = (pf_f16 & 2) ? 6 : getenv("HY4_PFIQ") ? atoi(getenv("HY4_PFIQ")) : 5;   // 6 (v2d, f16 bit 2): k_iq3_gemm4
            const bool skip = pf_skip;   // v2g: warps without tokens skip (a request's "skip", HY4_PF_SKIP)
            if (iqv == 6 && skip) {   // v2h: k_iq3_gemm5 (the same bits as k_iq3_gemm4, 64-column steps)
                k_iq3_gemm5<LAT / 256, true><<<dim3(FFE_C / G3M, nu), 256, 0, st>>>(P.jobs, P.lists, P.lat, LAT, P.gu_g, FFE_C, FFE_C, 0);
                k_iq3_gemm5<LAT / 256, true><<<dim3(FFE_C / G3M, nu), 256, 0, st>>>(P.jobs + nu, P.lists, P.lat, LAT, P.gu_u, FFE_C, FFE_C, 0);
            } else if (iqv == 6) {
                k_iq3_gemm4<LAT / 256><<<dim3(FFE_C / G3M, nu), 256, 0, st>>>(P.jobs, P.lists, P.lat, LAT, P.gu_g, FFE_C, FFE_C, 0);
                k_iq3_gemm4<LAT / 256><<<dim3(FFE_C / G3M, nu), 256, 0, st>>>(P.jobs + nu, P.lists, P.lat, LAT, P.gu_u, FFE_C, FFE_C, 0);
            } else if (iqv == 5 && skip) {
                k_iq3_gemm3<LAT / 256, true><<<dim3(FFE_C / G3M, nu), 256, 0, st>>>(P.jobs, P.lists, P.lat, LAT, P.gu_g, FFE_C, FFE_C, 0);
                k_iq3_gemm3<LAT / 256, true><<<dim3(FFE_C / G3M, nu), 256, 0, st>>>(P.jobs + nu, P.lists, P.lat, LAT, P.gu_u, FFE_C, FFE_C, 0);
            } else if (iqv == 5) {
                k_iq3_gemm3<LAT / 256><<<dim3(FFE_C / G3M, nu), 256, 0, st>>>(P.jobs, P.lists, P.lat, LAT, P.gu_g, FFE_C, FFE_C, 0);
                k_iq3_gemm3<LAT / 256><<<dim3(FFE_C / G3M, nu), 256, 0, st>>>(P.jobs + nu, P.lists, P.lat, LAT, P.gu_u, FFE_C, FFE_C, 0);
            } else if (iqv == 4) {
                k_iq3_gemm2<LAT / 256><<<dim3(FFE_C / GM, nu), 256, 0, st>>>(P.jobs, P.lists, P.lat, LAT, P.gu_g, FFE_C, FFE_C, 0);
                k_iq3_gemm2<LAT / 256><<<dim3(FFE_C / GM, nu), 256, 0, st>>>(P.jobs + nu, P.lists, P.lat, LAT, P.gu_u, FFE_C, FFE_C, 0);
            } else if (iqv == 1) {
                k_iq3_pf<<<dim3(FFE_C / IQ_BR, nu), 256, 0, st>>>(P.jobs, P.lists, P.lat, LAT, P.gu_g, FFE_C, FFE_C, LAT, 0);
                k_iq3_pf<<<dim3(FFE_C / IQ_BR, nu), 256, 0, st>>>(P.jobs + nu, P.lists, P.lat, LAT, P.gu_u, FFE_C, FFE_C, LAT, 0);
            } else if (iqv == 2) {
                k_iq3_pf2<LAT / 256><<<dim3(FFE_C / IQ_BR, nu), 256, 0, st>>>(P.jobs, P.lists, P.lat, LAT, P.gu_g, FFE_C, FFE_C, 0);
                k_iq3_pf2<LAT / 256><<<dim3(FFE_C / IQ_BR, nu), 256, 0, st>>>(P.jobs + nu, P.lists, P.lat, LAT, P.gu_u, FFE_C, FFE_C, 0);
            } else {
                k_iq3_pf3<LAT / 256, 4><<<dim3(FFE_C / 8, nu), 256, 0, st>>>(P.jobs, P.lists, P.lat, LAT, P.gu_g, FFE_C, FFE_C, 0);
                k_iq3_pf3<LAT / 256, 4><<<dim3(FFE_C / 8, nu), 256, 0, st>>>(P.jobs + nu, P.lists, P.lat, LAT, P.gu_u, FFE_C, FFE_C, 0);
            }
            if (getenv("HY4_PF_IQCMP") && c == 0 && iqv == 5) {   // k_iq3_gemm4 (into hh) against k_iq3_gemm3's gate rows (gu_g)
                CK(cudaMemsetAsync(P.nonfin + 2, 0, 8, st));
                CK(cudaMemsetAsync(P.hh, 0, (size_t) C * TOPK * FFE_C * 4, st));
                k_iq3_gemm4<LAT / 256><<<dim3(FFE_C / G3M, nu), 256, 0, st>>>(P.jobs, P.lists, P.lat, LAT, P.hh, FFE_C, FFE_C, 0);
                k_cmp_max<<<256, 256, 0, st>>>(P.hh, P.gu_g, (size_t) C * TOPK * FFE_C, P.nonfin + 2);
                unsigned r[2];
                CK(cudaMemcpyAsync(r, P.nonfin + 2, 8, cudaMemcpyDeviceToHost, st));
                CK(cudaStreamSynchronize(st));
                float d, m;
                std::memcpy(&d, r, 4);
                std::memcpy(&m, r + 1, 4);
                std::fprintf(stderr, "iqcmp layer %d at %d: max |gemm4 - gemm3| %.3e, max |gemm3| %.3e\n", il, p0, d, m);
            }
            k_swiglu<<<dim3(4, C * TOPK), 256, 0, st>>>(P.gu_g, P.gu_u, P.hh, FFE_C, l.clamp);
            if (iqv == 6 && skip)
                k_iq3_gemm5<FFE_C / 256, true><<<dim3(LAT / G3M, nu), 256, 0, st>>>(P.jobs + 2 * nu, P.lists, P.hh, FFE_C, P.ey, LAT, LAT, 1);
            else if (iqv == 6)
                k_iq3_gemm4<FFE_C / 256><<<dim3(LAT / G3M, nu), 256, 0, st>>>(P.jobs + 2 * nu, P.lists, P.hh, FFE_C, P.ey, LAT, LAT, 1);
            else if (iqv == 5 && skip)
                k_iq3_gemm3<FFE_C / 256, true><<<dim3(LAT / G3M, nu), 256, 0, st>>>(P.jobs + 2 * nu, P.lists, P.hh, FFE_C, P.ey, LAT, LAT, 1);
            else if (iqv == 5)
                k_iq3_gemm3<FFE_C / 256><<<dim3(LAT / G3M, nu), 256, 0, st>>>(P.jobs + 2 * nu, P.lists, P.hh, FFE_C, P.ey, LAT, LAT, 1);
            else if (iqv == 4)
                k_iq3_gemm2<FFE_C / 256><<<dim3(LAT / GM, nu), 256, 0, st>>>(P.jobs + 2 * nu, P.lists, P.hh, FFE_C, P.ey, LAT, LAT, 1);
            else if (iqv == 1)
                k_iq3_pf<<<dim3(LAT / IQ_BR, nu), 256, 0, st>>>(P.jobs + 2 * nu, P.lists, P.hh, FFE_C, P.ey, LAT, LAT, FFE_C, 1);
            else if (iqv == 2)
                k_iq3_pf2<FFE_C / 256><<<dim3(LAT / IQ_BR, nu), 256, 0, st>>>(P.jobs + 2 * nu, P.lists, P.hh, FFE_C, P.ey, LAT, LAT, 1);
            else
                k_iq3_pf3<FFE_C / 256, 8><<<dim3(LAT / 8, nu), 256, 0, st>>>(P.jobs + 2 * nu, P.lists, P.hh, FFE_C, P.ey, LAT, LAT, 1);
            if (staged) CK(cudaEventRecord(P.st_free[il & 1], st));   // the staging buffer may take layer il + 2 now
            if (c == 0) prof.stop("pf_strokovnjaki", st);
            k_moe_sum<<<C, 256, 0, st>>>(P.ey, P.topw, P.moe);
            sgemm(c, l.r_up, P.moe, LAT, P.fo, N, C, true);
            sgemm(c, l.sh_gate, P.xn, N, P.dg, FFE_C, C, true);
            sgemm(c, l.sh_up, P.xn, N, P.du, FFE_C, C, true);
            k_swiglu<<<dim3(4, C), 256, 0, st>>>(P.dg, P.du, P.dg, FFE_C, 0.0f);
            sgemm(c, l.sh_down, P.dg, FFE_C, P.fo2, N, C, true);
            k_add<<<256, 256, 0, st>>>(P.fo, P.fo2, (size_t) C * N);
            if (!dev) CK(cudaStreamSynchronize(st));   // lists and jall are the host path's (pageable copies return early only up to 64 KB)
        }
        xsum_pf(c, P.fo, C);
        rms(P.fo, N, l.post_ffw_norm, P.fo, N, C, N, M.eps, st);
        k_hc_post<<<dim3(C, HCB), 256, 0, st>>>(P.R, P.fo, P.post);
        if (c == 0) prof.stop("pf_deljeni_in_izmenjava", st);
        // HY4_DUMP_LAYER: a comma list of the layers to dump (unset: all)
        const char* dl = getenv("HY4_DUMP_LAYER");
        const bool want = !dl || ("," + std::string(dl) + ",").find("," + std::to_string(il) + ",") != std::string::npos;
        if (c == 0 && !dump.empty() && want) save("l_out-" + std::to_string(il), P.R, (size_t) C * D);
    }
