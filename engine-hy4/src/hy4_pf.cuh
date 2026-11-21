// hy4_pf.cuh (2026-11-20): Hy4's prompt path for hy4_v1 (chunks of up to HY4_PFC tokens). Dense matrices go through cuBLAS SGEMM on
// weights dequantized to f32 (exact like the windows), the routed experts run once per chunk per expert on all its tokens, the latent
// attention reads tiles of positions shared by a tile of tokens and heads, the indexer scores every complete group for every token and
// keeps the best 511 per token.
#pragma once
#include <cublas_v2.h>

namespace hy4 {

constexpr int HY4_PFC = 2048;   // v1l: the largest chunk (the scratch lives in cache slots); HY4_PF_CHUNK picks less

// Q8S -> f32 [rows][cols]: rows over the blocks, columns over the threads (a 64-bit division per value made this the prompt path's
// slowest kernel by far)
__global__ void k_deq8(const int8_t* q, const __half* d, size_t n, int cols, float* out) {
    const int rows = (int) (n / (size_t) cols), nb = cols / 32;
    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        const int8_t* qr = q + (size_t) r * cols;
        const __half* dr = d + (size_t) r * nb;
        float* o = out + (size_t) r * cols;
        for (int c = threadIdx.x; c < cols; c += blockDim.x) o[c] = (float) qr[c] * __half2float(dr[c >> 5]);
    }
}

// ---------------------------------------------------------------------------------------------------------------- attention
// Latent attention of a chunk. A block holds PT tokens x PH heads (8 query rows) and runs over tiles of 8 positions: in range mode
// the union of its tokens' windows [lo(t0), pos(t_last)] with K rows from a contiguous array (row = position - kbase); in list mode
// (PT = 1, the indexer's pick) the token's list, rows from the cache by position. Online softmax per query row; the sink joins the
// denominator at the end. q: [C][H][KROW], o: [C][H][KVL]
constexpr int PQ = 8, PP = 8;
template <int PT, int PH>
__global__ void __launch_bounds__(256) k_mla_pf(const float* qa, const __half* K, int kbase, const int* pos, int C, int H, int window,
                                                const int* sel, const int* nsel, const float* sinks, float scale, float* o) {
    static_assert(PT * PH == PQ, "8 query rows a block");
    __shared__ float qs[PQ][KROW];
    __shared__ float ks[PP][KROW];
    __shared__ float sc[PQ][PP];
    __shared__ float mrow[PQ], lrow[PQ], alpha[PQ];
    const int t0 = blockIdx.x * PT, h0 = blockIdx.y * PH, tid = threadIdx.x;
    for (int i = tid; i < PQ * KROW; i += 256) {
        const int r = i / KROW, c = i % KROW, t = t0 + r / PH, h = h0 + r % PH;
        qs[r][c] = t < C ? qa[((size_t) t * H + h) * KROW + c] : 0.0f;
    }
    if (tid < PQ) { mrow[tid] = -INFINITY; lrow[tid] = 0.0f; }
    const bool list = sel != nullptr;
    const int tl = min(t0 + PT, C) - 1;
    int lo, n;
    if (list) { lo = 0; n = nsel[t0]; }
    else {
        lo = window ? max(0, pos[t0] - SWA + 1) : 0;
        n = pos[tl] - lo + 1;
    }
    // warp w owns query row w: its scores (lanes over the columns: no bank conflicts), its softmax, its output columns lane + 32 j
    const int orow = tid >> 5, lane = tid & 31;
    const int tq = t0 + orow / PH;
    const int ptq = tq < C ? pos[tq] : -1;
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
    float m = -INFINITY, lsum = 0.0f;
    __syncthreads();
    for (int p0 = 0; p0 < n; p0 += PP) {
        for (int i = tid; i < PP * KROW; i += 256) {
            const int r = i / KROW, c = i % KROW, j = p0 + r;
            float v = 0.0f;
            if (j < n) {
                const int pp = list ? sel[(size_t) t0 * MAXSEL + j] : lo + j;
                v = __half2float(list ? K[(size_t) pp * KROW + c] : K[(size_t) (pp - kbase) * KROW + c]);
            }
            ks[r][c] = v;
        }
        __syncthreads();
        float p[PP];
        float mx = m;
#pragma unroll
        for (int k = 0; k < PP; ++k) {
            float s = 0;
#pragma unroll 6
            for (int c = lane; c < KROW; c += 32) s += qs[orow][c] * ks[k][c];
            s = warp_sum(s);
            const int j = p0 + k;
            bool ok = j < n && ptq >= 0;
            if (ok && !list) {
                const int pp = lo + j;
                ok = pp <= ptq && (!window || pp > ptq - SWA);
            }
            p[k] = ok ? s * scale : -INFINITY;
            mx = fmaxf(mx, p[k]);
        }
        const float a = mx == -INFINITY ? 1.0f : __expf(m - mx);
        lsum *= a;
#pragma unroll
        for (int k = 0; k < PP; ++k) {
            p[k] = p[k] == -INFINITY ? 0.0f : __expf(p[k] - mx);
            lsum += p[k];
        }
        m = mx;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            float v = acc[j] * a;
#pragma unroll
            for (int k = 0; k < PP; ++k) v += p[k] * ks[k][lane + 32 * j];
            acc[j] = v;
        }
        __syncthreads();
    }
    const int h = h0 + orow % PH;
    if (tq >= C) return;
    const float sk = sinks[h];
    const float mm = fmaxf(m, sk);
    const float den = lsum * __expf(m - mm) + __expf(sk - mm);
    const float f = __expf(m - mm) / den;
#pragma unroll
    for (int j = 0; j < 16; ++j) o[((size_t) tq * H + h) * KVL + lane + 32 * j] = acc[j] * f;
}

// v1l: the window layers' attention for a chunk as two cuBLAS products per 64-token tile (k_mla_pf ran at ~0.9 TFLOPS). Tile i's
// 64 tokens see positions p0 - 511 + 64 i .. + 575 (SWT rows); kf holds those positions as f32 rows from p0 - 511 on (zeros
// before position 0). S = Q K^T for the tile's 64 H query rows, k_swa_softmax masks, scales and normalises each row with its
// sink, O = P V (V: the rows' first 512 columns)
constexpr int SWTT = 64, SWT = SWTT + SWA;   // tokens a tile, positions a tile (575 used, the last one masked)
__global__ void k_kf_fill(const __half* kbuf, int npre, int nrows, float* kf) {   // kf row r = kbuf row r - (511 - npre)
    const int r = blockIdx.x, src = r - (SWA - 1 - npre);
    for (int c = threadIdx.x; c < KROW; c += blockDim.x)
        kf[(size_t) r * KROW + c] = src >= 0 && src < nrows ? __half2float(kbuf[(size_t) src * KROW + c]) : 0.0f;
}
__global__ void k_swa_softmax(float* S, int Mq, int H, int C, int p0, int nrows_total, const float* sinks, float scale) {
    const int g = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (g >= nrows_total) return;
    const int i = g / Mq, q = g % Mq, lt = q / H, h = q % H, t = i * SWTT + lt;
    float* s = S + (size_t) g * SWT;
    if (t >= C) {
        for (int j = lane; j < SWT; j += 32) s[j] = 0.0f;
        return;
    }
    const int pbase = p0 - (SWA - 1) + i * SWTT;   // the position of column 0
    float v[SWT / 32];
    float m = sinks[h];
#pragma unroll
    for (int k = 0; k < SWT / 32; ++k) {
        const int j = lane + 32 * k;
        const bool ok = j >= lt && j <= lt + SWA - 1 && pbase + j >= 0;
        v[k] = ok ? s[j] * scale : -INFINITY;
        m = fmaxf(m, v[k]);
    }
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float sum = 0;
#pragma unroll
    for (int k = 0; k < SWT / 32; ++k) {
        v[k] = v[k] == -INFINITY ? 0.0f : __expf(v[k] - m);
        sum += v[k];
    }
    sum = warp_sum(sum);
    const float inv = 1.0f / (sum + __expf(sinks[h] - m));
#pragma unroll
    for (int k = 0; k < SWT / 32; ++k) s[lane + 32 * k] = v[k] * inv;
}

// v1o: the sparse layers' attention for a chunk through cuBLAS too (k_mla_pf: 78 ms a layer before position 2048, 211 ms with
// the indexer's lists after). Before 2048 every token sees positions 0 .. its own: tiles of 64 tokens against the f32 rows of
// 0 .. np - 1 (k_kf_range), k_causal_softmax masks the rest. With lists: SELB tokens at a time, each token's selected rows gathered
// as f32 (k_sel_gather), then per token S = K_sel Q^T and O = P V_sel (strided batches), k_sel_softmax in between
constexpr int SELB = 32;
__global__ void k_kf_range(const __half* cache, float* kf) {   // grid rows: kf row r = cache row r
    const int r = blockIdx.x;
    for (int c = threadIdx.x; c < KROW / 2; c += blockDim.x)
        ((float2*) (kf + (size_t) r * KROW))[c] = __half22float2(((const __half2*) (cache + (size_t) r * KROW))[c]);
}
// rows of len scores: row g = (tile i, query q = lt * H + h), token t = (i0 + i) * 64 + lt sees positions 0 .. p0 + t
__global__ void k_causal_softmax(float* S, int len, int Mq, int H, int C, int p0, int i0, int nrows, const float* sinks, float scale) {
    const int g = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (g >= nrows) return;
    const int i = g / Mq, q = g % Mq, lt = q / H, h = q % H, t = (i0 + i) * SWTT + lt;
    float* s = S + (size_t) g * len;
    if (t >= C) {
        for (int j = lane; j < len; j += 32) s[j] = 0.0f;
        return;
    }
    const int last = p0 + t;
    float m = sinks[h];
    for (int j = lane; j <= last && j < len; j += 32) m = fmaxf(m, s[j] * scale);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float sum = 0;
    for (int j = lane; j < len; j += 32) {
        const float e = j <= last ? __expf(s[j] * scale - m) : 0.0f;
        s[j] = e;
        sum += e;
    }
    sum = warp_sum(sum);
    const float inv = 1.0f / (sum + __expf(sinks[h] - m));
    for (int j = lane; j < len; j += 32) s[j] *= inv;
}
__global__ void k_sel_gather(const __half* cache, const int* sel, const int* nsel, int t0, int C, float* kg) {   // grid (MAXSEL, SELB)
    const int b = blockIdx.y, j = blockIdx.x, t = t0 + b;
    float2* d = (float2*) (kg + ((size_t) b * MAXSEL + j) * KROW);
    const bool ok = t < C && j < nsel[t];
    const __half2* s = (const __half2*) (cache + (size_t) (ok ? sel[(size_t) t * MAXSEL + j] : 0) * KROW);
    for (int c = threadIdx.x; c < KROW / 2; c += blockDim.x) d[c] = ok ? __half22float2(s[c]) : make_float2(0.0f, 0.0f);
}
// rows (b, h) of MAXSEL scores: the first nsel[t0 + b] are the token's
__global__ void k_sel_softmax(float* S, int H, int t0, int C, const int* nsel, const float* sinks, float scale) {
    const int g = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int b = g / H, h = g % H, t = t0 + b;
    if (b >= SELB) return;
    float* s = S + (size_t) g * MAXSEL;
    const int n = t < C ? nsel[t] : 0;
    float m = sinks[h];
    for (int j = lane; j < n; j += 32) m = fmaxf(m, s[j] * scale);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float sum = 0;
    for (int j = lane; j < MAXSEL; j += 32) {
        const float e = j < n ? __expf(s[j] * scale - m) : 0.0f;
        s[j] = e;
        sum += e;
    }
    sum = warp_sum(sum);
    const float inv = 1.0f / (sum + __expf(sinks[h] - m));
    for (int j = lane; j < MAXSEL; j += 32) s[j] *= inv;
}

// v1t: the list mode in f16 (the cache rows are f16 already; the two products read half the bytes, f32 sums): the rows gathered
// as they are, Q and P as f16
__global__ void k_sel_gather_h(const __half* cache, const int* sel, const int* nsel, int t0, int C, __half* kg) {   // grid (MAXSEL, SELB)
    const int b = blockIdx.y, j = blockIdx.x, t = t0 + b;
    int4* d = (int4*) (kg + ((size_t) b * MAXSEL + j) * KROW);
    const bool ok = t < C && j < nsel[t];
    const int4* s = (const int4*) (cache + (size_t) (ok ? sel[(size_t) t * MAXSEL + j] : 0) * KROW);
    for (int c = threadIdx.x; c < KROW / 8; c += blockDim.x) d[c] = ok ? s[c] : make_int4(0, 0, 0, 0);
}
__global__ void k_sel_softmax_h(const float* S, int H, int t0, int C, const int* nsel, const float* sinks, float scale, __half* P) {
    const int g = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int b = g / H, h = g % H, t = t0 + b;
    if (b >= SELB) return;
    const float* s = S + (size_t) g * MAXSEL;
    __half* p = P + (size_t) g * MAXSEL;
    const int n = t < C ? nsel[t] : 0;
    float m = sinks[h];
    for (int j = lane; j < n; j += 32) m = fmaxf(m, s[j] * scale);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float sum = 0;
    for (int j = lane; j < n; j += 32) sum += __expf(s[j] * scale - m);
    sum = warp_sum(sum);
    const float inv = 1.0f / (sum + __expf(sinks[h] - m));
    for (int j = lane; j < MAXSEL; j += 32) p[j] = __float2half(j < n ? __expf(s[j] * scale - m) * inv : 0.0f);
}

// the window layers' K rows for a chunk: the 511 positions before it from the ring, then the chunk's own rows (contiguous)
__global__ void k_swa_gather(const __half* ring, int p0, int npre, __half* out) {   // out rows 0 .. npre - 1 = positions p0 - npre ..
    const int r = blockIdx.x;
    const int p = p0 - npre + r;
    for (int c = threadIdx.x; c < KROW; c += blockDim.x) out[(size_t) r * KROW + c] = ring[(size_t) (p % RING) * KROW + c];
}
// a chunk's latent rows: [rms(c) * w | k_pe] into rows of out (row r = token r), and, for the cache, into cache rows
__global__ void k_kv_rows(const float* kv, const float* w, float eps, float* out) {
    __shared__ float red[32];
    const float* r = kv + (size_t) blockIdx.x * KROW;
    float* y = out + (size_t) blockIdx.x * KROW;
    float s = 0;
    for (int i = threadIdx.x; i < KVL; i += blockDim.x) s += r[i] * r[i];
    s = block_sum(s, red);
    const float k = rsqrtf(s / KVL + eps);
    for (int i = threadIdx.x; i < KVL; i += blockDim.x) y[i] = r[i] * k * w[i];
    for (int i = threadIdx.x; i < ROT; i += blockDim.x) y[KVL + i] = r[KVL + i];
}
__global__ void k_f2h(const float* in, __half* out, size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) out[i] = __float2half(in[i]);
}
__global__ void k_rows_to_cache(const float* rows, __half* cache, const int* pos, int ring) {
    const int t = blockIdx.x, p = pos[t];
    if (ring && t + RING < (int) gridDim.x) return;   // v1l: a chunk longer than the ring keeps its last RING rows (no two blocks a slot)
    __half* d = cache + (size_t) (ring ? p % RING : p) * KROW;
    for (int c = threadIdx.x; c < KROW; c += blockDim.x) d[c] = __float2half(rows[(size_t) t * KROW + c]);
}

// ---------------------------------------------------------------------------------------------------------------- indexer
// acc[t][g] (+)= w[t][h] * relu(s[t][g]) over the visible groups (g < pos[t] / 4), -inf elsewhere at h == 0
__global__ void k_idx_acc(const float* s, int ng, const float* w, int h, const int* pos, float* acc) {
    const int t = blockIdx.y;
    const int vis = pos[t] / KPOOL;
    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < ng; g += gridDim.x * blockDim.x) {
        const float v = w[t * IDXH + h] * fmaxf(s[(size_t) t * ng + g], 0.0f);
        float& a = acc[(size_t) t * ng + g];
        if (h == 0) a = g < vis ? v : -INFINITY;
        else if (g < vis) a += v;
    }
}
// one token: <= TOPPOOL visible groups -> every position (nsel -1); else the TOPPOOL best (ties: the lower group) by a binary search
// for the threshold on the scores' order-preserving bits, then their 4 positions each, then the token's own group up to its position
__device__ __forceinline__ unsigned f2key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__global__ void k_idx_topk(const float* acc, int ng, const int* pos, int* sel, int* nsel, int full_list) {
    __shared__ int cnt, outn;
    const int t = blockIdx.x, p = pos[t], vis = p / KPOOL;
    if (vis <= TOPPOOL) {   // every position: -1 for the windows, the list itself for the prompt path's list mode
        if (full_list) {
            for (int q = threadIdx.x; q <= p; q += blockDim.x) sel[(size_t) t * MAXSEL + q] = q;
            if (threadIdx.x == 0) nsel[t] = p + 1;
        } else if (threadIdx.x == 0) nsel[t] = -1;
        return;
    }
    const float* a = acc + (size_t) t * ng;
    unsigned long long lo = 0, hi = 0xffffffffull;   // the largest key k with count(key >= k) >= TOPPOOL
    while (lo < hi) {
        const unsigned long long mid = lo + (hi - lo + 1) / 2;   // 64-bit: hi - lo + 1 overflowed 32 bits on the first step
        if (threadIdx.x == 0) cnt = 0;
        __syncthreads();
        int c = 0;
        for (int g = threadIdx.x; g < vis; g += blockDim.x) c += f2key(a[g]) >= mid;
        atomicAdd(&cnt, c);
        __syncthreads();
        if (cnt >= TOPPOOL) lo = mid; else hi = mid - 1;
        __syncthreads();
    }
    // strictly above the threshold first, then the threshold's ties in group order until TOPPOOL
    if (threadIdx.x == 0) {
        int n = 0;
        for (int g = 0; g < vis; ++g)
            if (f2key(a[g]) > lo) { for (int m = 0; m < KPOOL; ++m) sel[(size_t) t * MAXSEL + n * KPOOL + m] = g * KPOOL + m; ++n; }
        for (int g = 0; g < vis && n < TOPPOOL; ++g)
            if (f2key(a[g]) == lo) { for (int m = 0; m < KPOOL; ++m) sel[(size_t) t * MAXSEL + n * KPOOL + m] = g * KPOOL + m; ++n; }
        int k = n * KPOOL;
        for (int q = vis * KPOOL; q <= p; ++q) sel[(size_t) t * MAXSEL + k++] = q;
        outn = k;
        nsel[t] = k;
    }
}

// ---------------------------------------------------------------------------------------------------------------- experts
// a chunk's routed experts: one job a (expert, row tile), the expert's (token, pair) list in a table; the block loops over the
// expert's tokens in batches of PB, input rows in shared memory, the tile's rows decoded once a batch (L1 holds them)
struct PJob {
    const uint8_t* m;   // the matrix in VRAM
    int list, n;        // the expert's entries in the list table, how many
};
constexpr int PB = 4;
__global__ void __launch_bounds__(256) k_iq3_pf(const PJob* jobs, const int2* lists, const float* x, int ldx, float* y, int ldy, int rows,
                                                int cols, int in_is_pair) {
    __shared__ __align__(16) float xs[PB][IQ_MAXC];
    __shared__ uint32_t grid[256];
    __shared__ uint8_t sg[128];
    const PJob J = jobs[blockIdx.y];
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) sg[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
    const int warp = threadIdx.x >> 5, l = threadIdx.x & 31, ib = l >> 2, l4 = l & 3;
    const int r0 = blockIdx.x * IQ_BR + warp * IQ_R, nb = cols / 256;
    for (int b0 = 0; b0 < J.n; b0 += PB) {
        const int nbat = min(PB, J.n - b0);
        __syncthreads();
        for (int i = threadIdx.x; i < PB * cols; i += 256) {
            const int k = i / cols, c = i % cols;
            float v = 0.0f;
            if (k < nbat) {
                const int2 e = lists[J.list + b0 + k];   // (token, pair)
                v = x[(size_t) (in_is_pair ? e.y : e.x) * ldx + c];
            }
            xs[k][c] = v;
        }
        __syncthreads();
        float acc[IQ_R][PB];
#pragma unroll
        for (int a = 0; a < IQ_R; ++a)
#pragma unroll
            for (int k = 0; k < PB; ++k) acc[a][k] = 0.0f;
        for (int b = 0; b < nb; ++b) {
#pragma unroll
            for (int a = 0; a < IQ_R; ++a) {
                const int r = r0 + a;
                if (r >= rows) break;
                const uint16_t* blk = (const uint16_t*) (J.m + ((size_t) r * nb + b) * 98);
                const float d = __half2float(__ushort_as_half(__ldg(blk)));
                const uint16_t qq = __ldg(blk + 1 + l);
                const uint32_t aux = (uint32_t) __ldg(blk + 33 + 2 * ib) | ((uint32_t) __ldg(blk + 34 + 2 * ib) << 16);
                const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
                const uint32_t s = sg[(aux >> (7 * l4)) & 127];
                const uint32_t g1 = grid[qq & 255], g2 = grid[qq >> 8];
                float w[8];
                w[0] = (float) (g1 & 255) * (s & 1 ? -db : db);
                w[1] = (float) ((g1 >> 8) & 255) * (s & 2 ? -db : db);
                w[2] = (float) ((g1 >> 16) & 255) * (s & 4 ? -db : db);
                w[3] = (float) (g1 >> 24) * (s & 8 ? -db : db);
                w[4] = (float) (g2 & 255) * (s & 16 ? -db : db);
                w[5] = (float) ((g2 >> 8) & 255) * (s & 32 ? -db : db);
                w[6] = (float) ((g2 >> 16) & 255) * (s & 64 ? -db : db);
                w[7] = (float) (g2 >> 24) * (s & 128 ? -db : db);
#pragma unroll
                for (int k = 0; k < PB; ++k) {
                    const float4 xa = *(const float4*) &xs[k][b * 256 + l * 8], xb = *(const float4*) &xs[k][b * 256 + l * 8 + 4];
                    acc[a][k] += w[0] * xa.x + w[1] * xa.y + w[2] * xa.z + w[3] * xa.w + w[4] * xb.x + w[5] * xb.y + w[6] * xb.z + w[7] * xb.w;
                }
            }
        }
#pragma unroll
        for (int a = 0; a < IQ_R; ++a)
#pragma unroll
            for (int k = 0; k < PB; ++k) {
                const float v = warp_sum(acc[a][k]);
                if (l == 0 && r0 + a < rows && k < nbat) y[(size_t) lists[J.list + b0 + k].y * ldy + r0 + a] = v;
            }
    }
}

// v2: each warp decodes its IQ_R rows once into registers (8 values a lane a 256-block: up to 6 blocks, 48 a row), then runs every
// token of the expert: the token's input row through L1 (the block's 8 warps read the same row), one warp sum a row and token
template <int NB>
__global__ void __launch_bounds__(256) k_iq3_pf2(const PJob* jobs, const int2* lists, const float* x, int ldx, float* y, int ldy, int rows,
                                                 int in_is_pair) {
    __shared__ uint32_t grid[256];
    __shared__ uint8_t sg[128];
    const PJob J = jobs[blockIdx.y];
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) sg[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x >> 5, l = threadIdx.x & 31, ib = l >> 2, l4 = l & 3;
    const int r0 = blockIdx.x * IQ_BR + warp * IQ_R;
    float w[IQ_R][NB][8];
#pragma unroll
    for (int a = 0; a < IQ_R; ++a) {
        const int r = min(r0 + a, rows - 1);
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const uint16_t* blk = (const uint16_t*) (J.m + ((size_t) r * NB + b) * 98);
            const float d = __half2float(__ushort_as_half(__ldg(blk)));
            const uint16_t qq = __ldg(blk + 1 + l);
            const uint32_t aux = (uint32_t) __ldg(blk + 33 + 2 * ib) | ((uint32_t) __ldg(blk + 34 + 2 * ib) << 16);
            const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
            const uint32_t s = sg[(aux >> (7 * l4)) & 127];
            const uint32_t g1 = grid[qq & 255], g2 = grid[qq >> 8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                w[a][b][j] = (float) ((g1 >> (8 * j)) & 255) * ((s >> j) & 1 ? -db : db);
                w[a][b][4 + j] = (float) ((g2 >> (8 * j)) & 255) * ((s >> (4 + j)) & 1 ? -db : db);
            }
        }
    }
    for (int k = 0; k < J.n; ++k) {
        const int2 e = lists[J.list + k];
        const float* xr = x + (size_t) (in_is_pair ? e.y : e.x) * ldx;
        float acc[IQ_R];
#pragma unroll
        for (int a = 0; a < IQ_R; ++a) acc[a] = 0.0f;
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const float4 xa = __ldg((const float4*) (xr + b * 256 + l * 8)), xb = __ldg((const float4*) (xr + b * 256 + l * 8 + 4));
#pragma unroll
            for (int a = 0; a < IQ_R; ++a)
                acc[a] += w[a][b][0] * xa.x + w[a][b][1] * xa.y + w[a][b][2] * xa.z + w[a][b][3] * xa.w + w[a][b][4] * xb.x + w[a][b][5] * xb.y +
                          w[a][b][6] * xb.z + w[a][b][7] * xb.w;
        }
#pragma unroll
        for (int a = 0; a < IQ_R; ++a) {
            const float v = warp_sum(acc[a]);
            if (l == 0 && r0 + a < rows) y[(size_t) e.y * ldy + r0 + a] = v;
        }
    }
}

// v3: one row a warp (48 decoded weights a lane: fewer registers, more warps on an SM), the expert's tokens in groups of TB through
// shared memory (loaded once a block a group), one warp sum a token. Grid (rows / 8, experts)
template <int NB, int TB>
__global__ void __launch_bounds__(256) k_iq3_pf3(const PJob* jobs, const int2* lists, const float* x, int ldx, float* y, int ldy, int rows,
                                                 int in_is_pair) {
    __shared__ __align__(16) float xs[TB][NB * 256];
    __shared__ uint32_t grid[256];
    __shared__ uint8_t sg[128];
    const PJob J = jobs[blockIdx.y];
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) sg[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x >> 5, l = threadIdx.x & 31, ib = l >> 2, l4 = l & 3;
    const int r = blockIdx.x * 8 + warp;
    float w[NB][8];
    {
        const int rr = min(r, rows - 1);
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const uint16_t* blk = (const uint16_t*) (J.m + ((size_t) rr * NB + b) * 98);
            const float d = __half2float(__ushort_as_half(__ldg(blk)));
            const uint16_t qq = __ldg(blk + 1 + l);
            const uint32_t aux = (uint32_t) __ldg(blk + 33 + 2 * ib) | ((uint32_t) __ldg(blk + 34 + 2 * ib) << 16);
            const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
            const uint32_t s = sg[(aux >> (7 * l4)) & 127];
            const uint32_t g1 = grid[qq & 255], g2 = grid[qq >> 8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                w[b][j] = (float) ((g1 >> (8 * j)) & 255) * ((s >> j) & 1 ? -db : db);
                w[b][4 + j] = (float) ((g2 >> (8 * j)) & 255) * ((s >> (4 + j)) & 1 ? -db : db);
            }
        }
    }
    for (int k0 = 0; k0 < J.n; k0 += TB) {
        const int nk = min(TB, J.n - k0);
        __syncthreads();
        for (int i = threadIdx.x; i < TB * NB * 64; i += 256) {   // float4 a thread
            const int k = i / (NB * 64), c4 = i % (NB * 64);
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (k < nk) {
                const int2 e = lists[J.list + k0 + k];
                v = __ldg((const float4*) (x + (size_t) (in_is_pair ? e.y : e.x) * ldx) + c4);
            }
            *(float4*) &xs[k][c4 * 4] = v;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TB; ++k) {
            if (k >= nk) break;
            float acc = 0.0f;
#pragma unroll
            for (int b = 0; b < NB; ++b) {
                const float4 xa = *(const float4*) &xs[k][b * 256 + l * 8], xb = *(const float4*) &xs[k][b * 256 + l * 8 + 4];
                acc += w[b][0] * xa.x + w[b][1] * xa.y + w[b][2] * xa.z + w[b][3] * xa.w + w[b][4] * xb.x + w[b][5] * xb.y + w[b][6] * xb.z +
                       w[b][7] * xb.w;
            }
            acc = warp_sum(acc);
            if (l == 0 && r < rows) y[(size_t) lists[J.list + k0 + k].y * ldy + r] = acc;
        }
    }
}

// v1l (nvprof: k_iq3_pf3 ran at ~1.2 TFLOPS, each block of 8 rows reading all its tokens' inputs again): a tiled product. A block
// takes 128 rows of one expert and 32 of its tokens; each 64-column step dequantises the rows' two 32-value sub-blocks into
// shared memory (k-major, a thread a row and sub-block) and the tokens' inputs (k-major), then each thread adds an 8-row by
// 2-token tile. More tokens: the next 32 with the same rows. Grid (rows / 128, jobs)
constexpr int GM = 128, GN = 32, GK = 64;
template <int NB>
__global__ void __launch_bounds__(256) k_iq3_gemm(const PJob* jobs, const int2* lists, const float* x, int ldx, float* y, int ldy,
                                                  int rows, int in_is_pair) {
    __shared__ __align__(16) float ws[GK][GM];
    __shared__ __align__(16) float xs[GK][GN];
    __shared__ uint32_t grid[256];
    __shared__ uint8_t sg[128];
    __shared__ int2 lt[GN];
    const PJob J = jobs[blockIdx.y];
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) sg[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
    const int r0 = blockIdx.x * GM, ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
    const int dr = threadIdx.x & (GM - 1), dsb = threadIdx.x >> 7;   // the row and sub-block this thread dequantises
    for (int t0 = 0; t0 < J.n; t0 += GN) {
        const int nk = min(GN, J.n - t0);
        __syncthreads();
        if (threadIdx.x < GN) lt[threadIdx.x] = threadIdx.x < nk ? lists[J.list + t0 + threadIdx.x] : make_int2(-1, -1);
        float acc[8][2];
#pragma unroll
        for (int a = 0; a < 8; ++a) acc[a][0] = acc[a][1] = 0.0f;
        for (int ks = 0; ks < NB * 4; ++ks) {   // 64 columns: block ks / 4, sub-blocks 2 (ks % 4) and + 1
            __syncthreads();
            {
                const int b = ks >> 2, ib = 2 * (ks & 3) + dsb;
                const uint16_t* blk = (const uint16_t*) (J.m + ((size_t) (r0 + dr) * NB + b) * 98);
                const float d = __half2float(__ushort_as_half(__ldg(blk)));
                const uint32_t aux = (uint32_t) __ldg(blk + 33 + 2 * ib) | ((uint32_t) __ldg(blk + 34 + 2 * ib) << 16);
                const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
#pragma unroll
                for (int l4 = 0; l4 < 4; ++l4) {
                    const uint16_t qq = __ldg(blk + 1 + 4 * ib + l4);
                    const uint32_t s = sg[(aux >> (7 * l4)) & 127];
                    const uint32_t g1 = grid[qq & 255], g2 = grid[qq >> 8];
                    const int k = dsb * 32 + l4 * 8;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        ws[k + j][dr] = (float) ((g1 >> (8 * j)) & 255) * ((s >> j) & 1 ? -db : db);
                        ws[k + 4 + j][dr] = (float) ((g2 >> (8 * j)) & 255) * ((s >> (4 + j)) & 1 ? -db : db);
                    }
                }
            }
            {   // the inputs: token tok's 8 columns seg .. seg + 7 a thread
                const int tok = threadIdx.x >> 3, seg = (threadIdx.x & 7) * 8;
                float4 v0 = make_float4(0.f, 0.f, 0.f, 0.f), v1 = v0;
                if (tok < nk) {
                    const int2 e = lt[tok];
                    const float* xr = x + (size_t) (in_is_pair ? e.y : e.x) * ldx + ks * GK + seg;
                    v0 = __ldg((const float4*) xr);
                    v1 = __ldg((const float4*) (xr + 4));
                }
                xs[seg + 0][tok] = v0.x; xs[seg + 1][tok] = v0.y; xs[seg + 2][tok] = v0.z; xs[seg + 3][tok] = v0.w;
                xs[seg + 4][tok] = v1.x; xs[seg + 5][tok] = v1.y; xs[seg + 6][tok] = v1.z; xs[seg + 7][tok] = v1.w;
            }
            __syncthreads();
#pragma unroll 8
            for (int k = 0; k < GK; ++k) {
                const float4 w0 = *(const float4*) &ws[k][ty * 8], w1 = *(const float4*) &ws[k][ty * 8 + 4];
                const float2 xv = *(const float2*) &xs[k][tx * 2];
                acc[0][0] += w0.x * xv.x; acc[0][1] += w0.x * xv.y;
                acc[1][0] += w0.y * xv.x; acc[1][1] += w0.y * xv.y;
                acc[2][0] += w0.z * xv.x; acc[2][1] += w0.z * xv.y;
                acc[3][0] += w0.w * xv.x; acc[3][1] += w0.w * xv.y;
                acc[4][0] += w1.x * xv.x; acc[4][1] += w1.x * xv.y;
                acc[5][0] += w1.y * xv.x; acc[5][1] += w1.y * xv.y;
                acc[6][0] += w1.z * xv.x; acc[6][1] += w1.z * xv.y;
                acc[7][0] += w1.w * xv.x; acc[7][1] += w1.w * xv.y;
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int tok = tx * 2 + i;
            if (tok >= nk) continue;
            float* yr = y + (size_t) lt[tok].y * ldy + r0 + ty * 8;
#pragma unroll
            for (int a = 0; a < 8; ++a) yr[a] = acc[a][i];
        }
    }
}

// k_iq3_gemm with the next step's raw bytes and inputs loaded into registers while this step's products run (one block an SM fits:
// the loads' latency was in the open)
template <int NB>
__global__ void __launch_bounds__(256) k_iq3_gemm2(const PJob* jobs, const int2* lists, const float* x, int ldx, float* y, int ldy,
                                                   int rows, int in_is_pair) {
    __shared__ __align__(16) float ws[GK][GM];
    __shared__ __align__(16) float xs[GK][GN];
    __shared__ uint32_t grid[256];
    __shared__ uint8_t sg[128];
    __shared__ int2 lt[GN];
    const PJob J = jobs[blockIdx.y];
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) sg[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
    const int r0 = blockIdx.x * GM, ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
    const int dr = threadIdx.x & (GM - 1), dsb = threadIdx.x >> 7;
    const int tok = threadIdx.x >> 3, seg = (threadIdx.x & 7) * 8;
    const uint16_t* rowp = (const uint16_t*) (J.m + (size_t) (r0 + dr) * NB * 98);
    for (int t0 = 0; t0 < J.n; t0 += GN) {
        const int nk = min(GN, J.n - t0);
        __syncthreads();
        if (threadIdx.x < GN) lt[threadIdx.x] = threadIdx.x < nk ? lists[J.list + t0 + threadIdx.x] : make_int2(-1, -1);
        __syncthreads();
        const float* xr = nullptr;
        if (tok < nk) xr = x + (size_t) (in_is_pair ? lt[tok].y : lt[tok].x) * ldx + seg;
        uint16_t rd, ra0, ra1, rq[4];
        float4 v0, v1;
        auto load = [&](int ks) {
            const int b = ks >> 2, ib = 2 * (ks & 3) + dsb;
            const uint16_t* blk = rowp + b * 49;
            rd = __ldg(blk);
            ra0 = __ldg(blk + 33 + 2 * ib);
            ra1 = __ldg(blk + 34 + 2 * ib);
#pragma unroll
            for (int l4 = 0; l4 < 4; ++l4) rq[l4] = __ldg(blk + 1 + 4 * ib + l4);
            v0 = v1 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (xr) {
                v0 = __ldg((const float4*) (xr + ks * GK));
                v1 = __ldg((const float4*) (xr + ks * GK + 4));
            }
        };
        load(0);
        float acc[8][2];
#pragma unroll
        for (int a = 0; a < 8; ++a) acc[a][0] = acc[a][1] = 0.0f;
        for (int ks = 0; ks < NB * 4; ++ks) {
            __syncthreads();
            {
                const uint32_t aux = (uint32_t) ra0 | ((uint32_t) ra1 << 16);
                const float db = __half2float(__ushort_as_half(rd)) * (0.5f + (float) (aux >> 28)) * 0.5f;
#pragma unroll
                for (int l4 = 0; l4 < 4; ++l4) {
                    const uint32_t s = sg[(aux >> (7 * l4)) & 127];
                    const uint32_t g1 = grid[rq[l4] & 255], g2 = grid[rq[l4] >> 8];
                    const int k = dsb * 32 + l4 * 8;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        ws[k + j][dr] = (float) ((g1 >> (8 * j)) & 255) * ((s >> j) & 1 ? -db : db);
                        ws[k + 4 + j][dr] = (float) ((g2 >> (8 * j)) & 255) * ((s >> (4 + j)) & 1 ? -db : db);
                    }
                }
                xs[seg + 0][tok] = v0.x; xs[seg + 1][tok] = v0.y; xs[seg + 2][tok] = v0.z; xs[seg + 3][tok] = v0.w;
                xs[seg + 4][tok] = v1.x; xs[seg + 5][tok] = v1.y; xs[seg + 6][tok] = v1.z; xs[seg + 7][tok] = v1.w;
            }
            __syncthreads();
            if (ks + 1 < NB * 4) load(ks + 1);
#pragma unroll 8
            for (int k = 0; k < GK; ++k) {
                const float4 w0 = *(const float4*) &ws[k][ty * 8], w1 = *(const float4*) &ws[k][ty * 8 + 4];
                const float2 xv = *(const float2*) &xs[k][tx * 2];
                acc[0][0] += w0.x * xv.x; acc[0][1] += w0.x * xv.y;
                acc[1][0] += w0.y * xv.x; acc[1][1] += w0.y * xv.y;
                acc[2][0] += w0.z * xv.x; acc[2][1] += w0.z * xv.y;
                acc[3][0] += w0.w * xv.x; acc[3][1] += w0.w * xv.y;
                acc[4][0] += w1.x * xv.x; acc[4][1] += w1.x * xv.y;
                acc[5][0] += w1.y * xv.x; acc[5][1] += w1.y * xv.y;
                acc[6][0] += w1.z * xv.x; acc[6][1] += w1.z * xv.y;
                acc[7][0] += w1.w * xv.x; acc[7][1] += w1.w * xv.y;
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int tk = tx * 2 + i;
            if (tk >= nk) continue;
            float* yr = y + (size_t) lt[tk].y * ldy + r0 + ty * 8;
#pragma unroll
            for (int a = 0; a < 8; ++a) yr[a] = acc[a][i];
        }
    }
}

// v1s: tiles of 128 rows by 64 tokens (2048-token chunks give an expert ~64 tokens), 32-column steps (one sub-block a row: half
// a thread's 16 values each), each thread an 8-row by 4-token tile: 32 FMAs per 3 shared loads (k_iq3_gemm2: 16 per 3), 24 KB of
// shared memory (two blocks an SM), the next step's bytes and inputs in registers during this step's products
constexpr int G3M = 128, G3N = 64, G3K = 32;
template <int NB, bool SKIP = false>
__global__ void __launch_bounds__(256) k_iq3_gemm3(const PJob* jobs, const int2* lists, const float* x, int ldx, float* y, int ldy,
                                                   int rows, int in_is_pair) {
    __shared__ __align__(16) float ws[G3K][G3M];
    __shared__ __align__(16) float xs[G3K][G3N];
    __shared__ uint32_t grid[256];
    __shared__ uint8_t sg[128];
    __shared__ int2 lt[G3N];
    const PJob J = jobs[blockIdx.y];
    if (J.n == 0) return;
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) sg[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
    // SKIP (v2g): a warp holds 8 tokens (tx = its two token groups), so a warp with no tokens of the tile skips the products
    const int r0 = blockIdx.x * G3M, ty = SKIP ? threadIdx.x & 15 : threadIdx.x >> 4, tx = SKIP ? threadIdx.x >> 4 : threadIdx.x & 15;
    const int wtok = (threadIdx.x >> 5) * 8;
    const int dr = threadIdx.x & (G3M - 1), dh = threadIdx.x >> 7;   // the row and the half sub-block (l4 pairs 2 dh, 2 dh + 1)
    const int tok = threadIdx.x >> 2, seg = (threadIdx.x & 3) * 8;
    const uint16_t* rowp = (const uint16_t*) (J.m + (size_t) (r0 + dr) * NB * 98);
    for (int t0 = 0; t0 < J.n; t0 += G3N) {
        const int nk = min(G3N, J.n - t0);
        __syncthreads();
        if (threadIdx.x < G3N) lt[threadIdx.x] = threadIdx.x < nk ? lists[J.list + t0 + threadIdx.x] : make_int2(-1, -1);
        __syncthreads();
        const float* xr = nullptr;
        if (tok < nk) xr = x + (size_t) (in_is_pair ? lt[tok].y : lt[tok].x) * ldx + seg;
        uint16_t rd, ra0, ra1, rq[2];
        float4 v0, v1;
        auto load = [&](int ks) {   // step ks: block ks / 8, sub-block ks % 8
            const int b = ks >> 3, ib = ks & 7;
            const uint16_t* blk = rowp + b * 49;
            rd = __ldg(blk);
            ra0 = __ldg(blk + 33 + 2 * ib);
            ra1 = __ldg(blk + 34 + 2 * ib);
            rq[0] = __ldg(blk + 1 + 4 * ib + 2 * dh);
            rq[1] = __ldg(blk + 2 + 4 * ib + 2 * dh);
            v0 = v1 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (xr) {
                v0 = __ldg((const float4*) (xr + ks * G3K));
                v1 = __ldg((const float4*) (xr + ks * G3K + 4));
            }
        };
        load(0);
        float acc[8][4];
#pragma unroll
        for (int a = 0; a < 8; ++a)
#pragma unroll
            for (int k = 0; k < 4; ++k) acc[a][k] = 0.0f;
        for (int ks = 0; ks < NB * 8; ++ks) {
            __syncthreads();
            {
                const uint32_t aux = (uint32_t) ra0 | ((uint32_t) ra1 << 16);
                const float db = __half2float(__ushort_as_half(rd)) * (0.5f + (float) (aux >> 28)) * 0.5f;
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const int l4 = 2 * dh + p;
                    const uint32_t s = sg[(aux >> (7 * l4)) & 127];
                    const uint32_t g1 = grid[rq[p] & 255], g2 = grid[rq[p] >> 8];
                    const int k = l4 * 8;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        ws[k + j][dr] = (float) ((g1 >> (8 * j)) & 255) * ((s >> j) & 1 ? -db : db);
                        ws[k + 4 + j][dr] = (float) ((g2 >> (8 * j)) & 255) * ((s >> (4 + j)) & 1 ? -db : db);
                    }
                }
                xs[seg + 0][tok] = v0.x; xs[seg + 1][tok] = v0.y; xs[seg + 2][tok] = v0.z; xs[seg + 3][tok] = v0.w;
                xs[seg + 4][tok] = v1.x; xs[seg + 5][tok] = v1.y; xs[seg + 6][tok] = v1.z; xs[seg + 7][tok] = v1.w;
            }
            __syncthreads();
            if (ks + 1 < NB * 8) load(ks + 1);
            if (SKIP && wtok >= nk) continue;
#pragma unroll 8
            for (int k = 0; k < G3K; ++k) {
                const float4 w0 = *(const float4*) &ws[k][ty * 8], w1 = *(const float4*) &ws[k][ty * 8 + 4];
                const float4 xv = *(const float4*) &xs[k][tx * 4];
                const float w[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w}, xx[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
                for (int a = 0; a < 8; ++a)
#pragma unroll
                    for (int kk = 0; kk < 4; ++kk) acc[a][kk] += w[a] * xx[kk];
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int tk = tx * 4 + i;
            if (tk >= nk) continue;
            float* yr = y + (size_t) lt[tk].y * ldy + r0 + ty * 8;
#pragma unroll
            for (int a = 0; a < 8; ++a) yr[a] = acc[a][i];
        }
    }
}

// v1q: the prompt path's expert lists on the card (v1: the router's picks to the host and back each layer while the host thread
// waits): per expert its (token, pair) entries, then each expert's three jobs; an expert without tokens gets n = 0 (its blocks
// return at once). The order inside an expert's list is the atomics' (each pair's sums do not depend on it)
__global__ void __launch_bounds__(1024) k_pf_lists(const int* topi, int C, const uint8_t* const* sptr, size_t half_gu, PJob* jobs,
                                                   int2* lists) {
    __shared__ int cnt[NEXP], off[NEXP], cur[NEXP];
    for (int e = threadIdx.x; e < NEXP; e += blockDim.x) cnt[e] = cur[e] = 0;
    __syncthreads();
    const int np = C * TOPK;
    for (int j = threadIdx.x; j < np; j += blockDim.x) atomicAdd(&cnt[topi[j]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int s = 0;
        for (int e = 0; e < NEXP; ++e) {
            off[e] = s;
            s += cnt[e];
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < np; j += blockDim.x) {
        const int e = topi[j];
        lists[off[e] + atomicAdd(&cur[e], 1)] = make_int2(j / TOPK, j);
    }
    for (int e = threadIdx.x; e < NEXP; e += blockDim.x) {
        const uint8_t* b = sptr[e];
        jobs[e] = {b, off[e], cnt[e]};
        jobs[NEXP + e] = {b + half_gu, off[e], cnt[e]};
        jobs[2 * NEXP + e] = {b + 2 * half_gu, off[e], cnt[e]};
    }
}

// ---------------------------------------------------------------------------------------------------------------- exchange
// the cross-card sum for a chunk: PXB blocks, each its slice and its own flag (flags [blocks]); seq as for the windows
constexpr int PXB = 32;
__global__ void k_xput_pf(const float* y, int n, float* out, volatile unsigned* flags, unsigned seq) {
    const int per = (n + PXB - 1) / PXB, b0 = blockIdx.x * per, b1 = min(n, b0 + per);
    for (int i = b0 + threadIdx.x; i < b1; i += blockDim.x) out[i] = y[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) flags[blockIdx.x] = seq;
}
__global__ void k_xget_pf(float* y, int n, const volatile float* in, const volatile unsigned* flags, unsigned seq) {
    const int per = (n + PXB - 1) / PXB, b0 = blockIdx.x * per, b1 = min(n, b0 + per);
    if (threadIdx.x == 0)
        while (flags[blockIdx.x] < seq) {}
    __syncthreads();
    __threadfence_system();
    for (int i = b0 + threadIdx.x; i < b1; i += blockDim.x) y[i] = y[i] + in[i];
}

}  // namespace hy4
