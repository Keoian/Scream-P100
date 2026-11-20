// m40_rows_test.cu: the M40's expert kernels (sm_52) against a CPU reference, on made-up weights of every type the M40 model has.
//   rows: the decode window's kernel (m40_moe_rows.cuh, f32 tables and rows) for windows of 1 to 16 rows;
//   gemm: the prompt path's (p100_moe.cu's k_gemm4_job, f16 tiles and f32 sums on Maxwell) for a 96-row job.
// The reference decodes each row with ggml's dequantize_row_* formulas (the tables copied from the device) and sums in double.
// Idle cards only (the production engine unloaded). Build: mikro/m40_test_gradnja.sh. Prints the worst relative error and the time.
#include "p100_moe.cu"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace fnx;
namespace fnx {
int qsa_maxsel() { return QSA_TOP_K; }   // the attention code p100_moe.cu includes asks it (kernels.cu has the engine's); never called here
}

static uint16_t f2h(float f) {   // round to nearest even, normal range only (the test's values)
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int e = (int) ((x >> 23) & 255) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rest = m & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}
static float h2f_host(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) u = s << 31;   // the test's scales are normal
    else u = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Ref {   // ggml's tables, read from the device copies of ggml-common.h
    uint64_t g2xxs[256], g2xs[512], g2s[1024];
    uint32_t g3xxs[256], g3s[512];
    uint8_t ksigns[128];
    int8_t kv[16];
    void load() {
        cudaMemcpyFromSymbol(g2xxs, iq2xxs_grid, sizeof g2xxs);
        cudaMemcpyFromSymbol(g2xs, iq2xs_grid, sizeof g2xs);
        cudaMemcpyFromSymbol(g2s, iq2s_grid, sizeof g2s);
        cudaMemcpyFromSymbol(g3xxs, iq3xxs_grid, sizeof g3xxs);
        cudaMemcpyFromSymbol(g3s, iq3s_grid, sizeof g3s);
        cudaMemcpyFromSymbol(ksigns, ksigns_iq2xs, sizeof ksigns);
        cudaMemcpyFromSymbol(kv, kvalues_iq4nl, sizeof kv);
    }
    static float sg(uint8_t s, int j) { return (s >> j) & 1 ? -1.f : 1.f; }
    static uint8_t b(uint64_t g, int j) { return (uint8_t) (g >> (8 * j)); }
    // one row of K values
    void row(uint32_t ty, const uint8_t* r, int K, float* y) const {
        if (ty == T_IQ4_NL || ty == T_Q4_0) {
            for (int bk = 0; bk < K / 32; ++bk, r += 18, y += 32) {
                uint16_t dh;
                std::memcpy(&dh, r, 2);
                const float d = h2f_host(dh);
                for (int j = 0; j < 16; ++j) {
                    const int lo = r[2 + j] & 15, hi = r[2 + j] >> 4;
                    y[j] = d * (ty == T_Q4_0 ? (float) (lo - 8) : (float) kv[lo]);
                    y[j + 16] = d * (ty == T_Q4_0 ? (float) (hi - 8) : (float) kv[hi]);
                }
            }
            return;
        }
        if (ty == T_Q2_0) {
            for (int bk = 0; bk < K / 64; ++bk, r += 18, y += 64) {
                uint16_t dh;
                std::memcpy(&dh, r, 2);
                const float d = h2f_host(dh);
                for (int j = 0; j < 64; ++j) y[j] = (float) (((r[2 + j / 4] >> (2 * (j % 4))) & 3) - 1) * d;
            }
            return;
        }
        const int bytes = ty == T_IQ2_XXS ? 66 : ty == T_IQ2_XS ? 74 : ty == T_IQ2_S ? 82 : ty == T_IQ3_XXS ? 98 : 110;
        for (int bk = 0; bk < K / 256; ++bk, r += bytes) {
            uint16_t dh;
            std::memcpy(&dh, r, 2);
            const float d = h2f_host(dh);
            const uint8_t* q = r + 2;
            for (int ib = 0; ib < 8; ++ib) {
                float* o = y + 256 * bk + 32 * ib;
                if (ty == T_IQ2_XXS) {
                    uint32_t a0, a1;
                    std::memcpy(&a0, q + 8 * ib, 4);
                    std::memcpy(&a1, q + 8 * ib + 4, 4);
                    const float db = d * (0.5f + (float) (a1 >> 28)) * 0.25f;
                    for (int l = 0; l < 4; ++l)
                        for (int j = 0; j < 8; ++j) o[8 * l + j] = db * b(g2xxs[(a0 >> (8 * l)) & 255], j) * sg(ksigns[(a1 >> (7 * l)) & 127], j);
                } else if (ty == T_IQ2_XS) {
                    const uint8_t sc = q[64 + ib];
                    for (int l = 0; l < 4; ++l) {
                        uint16_t v;
                        std::memcpy(&v, q + 8 * ib + 2 * l, 2);
                        const float db = d * (0.5f + (float) (l < 2 ? sc & 15 : sc >> 4)) * 0.25f;
                        for (int j = 0; j < 8; ++j) o[8 * l + j] = db * b(g2xs[v & 511], j) * sg(ksigns[v >> 9], j);
                    }
                } else if (ty == T_IQ2_S) {
                    const uint8_t sc = q[64 + 8 + ib], qh = q[64 + ib];
                    for (int l = 0; l < 4; ++l) {
                        const float db = d * (0.5f + (float) (l < 2 ? sc & 15 : sc >> 4)) * 0.25f;
                        const int gi = q[4 * ib + l] | ((qh << (8 - 2 * l)) & 0x300);
                        for (int j = 0; j < 8; ++j) o[8 * l + j] = db * b(g2s[gi], j) * sg(q[32 + 4 * ib + l], j);
                    }
                } else if (ty == T_IQ3_XXS) {
                    uint32_t aux;
                    std::memcpy(&aux, q + 64 + 4 * ib, 4);
                    const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
                    for (int l = 0; l < 4; ++l) {
                        const uint8_t s = ksigns[(aux >> (7 * l)) & 127];
                        for (int j = 0; j < 4; ++j) {
                            o[8 * l + j] = db * b(g3xxs[q[8 * ib + 2 * l]], j) * sg(s, j);
                            o[8 * l + 4 + j] = db * b(g3xxs[q[8 * ib + 2 * l + 1]], j) * sg(s, j + 4);
                        }
                    }
                } else {   // IQ3_S: qs[64], qh[8], signs[32], scales[4]
                    const uint8_t qh = q[64 + ib];
                    const float db = d * (float) (1 + 2 * ((q[64 + 8 + 32 + ib / 2] >> (4 * (ib & 1))) & 15));
                    for (int l = 0; l < 4; ++l) {
                        const uint8_t s = q[64 + 8 + 4 * ib + l];
                        const int i1 = q[8 * ib + 2 * l] | ((qh << (8 - 2 * l)) & 256), i2 = q[8 * ib + 2 * l + 1] | ((qh << (7 - 2 * l)) & 256);
                        for (int j = 0; j < 4; ++j) {
                            o[8 * l + j] = db * b(g3s[i1], j) * sg(s, j);
                            o[8 * l + 4 + j] = db * b(g3s[i2], j) * sg(s, j + 4);
                        }
                    }
                }
            }
        }
    }
};

struct Case {
    const char* name;
    uint32_t type;
    int block_values, block_bytes, N, K;
};

static double run_case(const Case& c, const Ref& ref, int T, bool gemm, bool show) {
    std::mt19937 g(c.type * 1000 + T * 7 + gemm);
    const size_t row_bytes = (size_t) c.K / c.block_values * c.block_bytes, stride = (size_t) c.N * row_bytes;
    const int E = 12, C = gemm ? 96 : 16, njobs = gemm ? 4 : 10;
    std::vector<uint8_t> w((size_t) E * stride);
    for (auto& v : w) v = (uint8_t) g();
    for (size_t r = 0; r + c.block_bytes <= w.size(); r += c.block_bytes) {   // every block's d: 0.01 .. 0.03
        const uint16_t d = f2h(0.01f + 0.02f * (float) (g() % 1000) / 1000.0f);
        std::memcpy(&w[r], &d, 2);
    }
    std::vector<float> x((size_t) C * c.K);
    std::vector<uint16_t> xh(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        xh[i] = f2h(((float) (g() % 2001) - 1000.0f) / 500.0f);
        x[i] = h2f_host(xh[i]);   // the same values for the f16 and the f32 kernel
    }
    std::vector<Job> jobs((size_t) njobs);
    std::vector<BigJob> big((size_t) njobs);
    int pair = 0;
    const int rows_job = gemm ? C : T;
    for (int j = 0; j < njobs; ++j) {
        jobs[(size_t) j].w = big[(size_t) j].w = (j * 5 + 3) % E;
        jobs[(size_t) j].n = big[(size_t) j].n = rows_job;
        for (int t = 0; t < rows_job; ++t) {
            const int y = gemm ? t : (int) (g() % C);
            if (gemm) big[(size_t) j].y[t] = y, big[(size_t) j].out[t] = pair++;
            else jobs[(size_t) j].y[t] = y, jobs[(size_t) j].out[t] = pair++;
        }
    }
    cudaStream_t s;
    cudaStreamCreate(&s);
    uint8_t* dw = nullptr;
    void* dx = nullptr;
    float* dy = nullptr;
    void* dj = nullptr;
    int* dn = nullptr;
    cudaMalloc(&dw, w.size());
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dx, x.size() * 4);
    if (gemm) cudaMemcpy(dx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);
    else cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    // the rows kernel of a small window splits gate|up rows by columns into ks partial slices of pair N floats (m40_ks)
    const int ks = gemm || (c.type != T_IQ2_XXS && c.type != T_IQ2_XS && c.type != T_IQ2_S && c.type != T_IQ3_XXS && c.type != T_IQ3_S) ? 1 : m40_ks(T, c.K);
    cudaMalloc(&dy, (size_t) ks * pair * c.N * 4);
    cudaMemset(dy, 0, (size_t) ks * pair * c.N * 4);
    const size_t jb = gemm ? big.size() * sizeof(BigJob) : jobs.size() * sizeof(Job);
    cudaMalloc(&dj, jb);
    cudaMemcpy(dj, gemm ? (const void*) big.data() : (const void*) jobs.data(), jb, cudaMemcpyHostToDevice);
    cudaMalloc(&dn, 4);
    cudaMemcpy(dn, &njobs, 4, cudaMemcpyHostToDevice);
    DW A;
    A.p = dw;
    A.type = c.type;
    A.n_in = c.K;
    A.n_out = c.N;
    A.row_bytes = row_bytes;
    auto launch = [&] {
        if (gemm) p100_moe(A, stride, (const BigJob*) dj, dn, njobs, (const uint16_t*) dx, dy, s);
        else p100_moe_rows_phase(A, stride, (const Job*) dj, dn, pair, T, (const uint16_t*) dx, dy, 0, s);   // njobs_cap: the pairs, as the engine passes P
    };
    launch();
    const cudaError_t e = cudaStreamSynchronize(s);
    if (e != cudaSuccess) {
        std::printf("%-8s %s T=%2d: CUDA %s\n", c.name, gemm ? "gemm" : "rows", T, cudaGetErrorString(e));
        std::exit(1);
    }
    std::vector<float> ys((size_t) ks * pair * c.N), y((size_t) pair * c.N, 0.0f);
    cudaMemcpy(ys.data(), dy, ys.size() * 4, cudaMemcpyDeviceToHost);
    for (int k = 0; k < ks; ++k)
        for (size_t i = 0; i < y.size(); ++i) y[i] += ys[(size_t) k * y.size() + i];
    // the reference: every job's rows, decoded once each
    double worst = 0, scale = 0;
    std::vector<float> wr((size_t) c.K);
    for (int j = 0; j < njobs; ++j) {
        const int ex = gemm ? big[(size_t) j].w : jobs[(size_t) j].w;
        for (int r = 0; r < c.N; ++r) {
            ref.row(c.type, w.data() + (size_t) ex * stride + (size_t) r * row_bytes, c.K, wr.data());
            for (int t = 0; t < rows_job; ++t) {
                const int yy = gemm ? big[(size_t) j].y[t] : jobs[(size_t) j].y[t], o = gemm ? big[(size_t) j].out[t] : jobs[(size_t) j].out[t];
                double acc = 0;
                for (int k = 0; k < c.K; ++k) acc += (double) wr[(size_t) k] * x[(size_t) yy * c.K + k];
                scale = std::max(scale, std::fabs(acc));
                worst = std::max(worst, std::fabs(acc - (double) y[(size_t) o * c.N + r]));
            }
        }
    }
    const double rel = worst / std::max(scale, 1e-30);
    // time: 50 launches
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 50; ++i) launch();
    cudaStreamSynchronize(s);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 50;
    const double gb = (double) njobs * stride / 1e9 / (us / 1e6);
    if (show)
        std::printf("%-8s %s T=%2d: worst abs %.3g of %.3g (rel %.2e) %s, %.1f us a launch, %.0f GB/s of weights\n", c.name, gemm ? "gemm" : "rows",
                    rows_job, worst, scale, rel, rel < (gemm ? 3e-3 : 1e-4) ? "OK" : "NAPAKA", us, gb);
    cudaFree(dw), cudaFree(dx), cudaFree(dy), cudaFree(dj), cudaFree(dn);
    cudaStreamDestroy(s);
    return rel;
}

int main() {
    cudaSetDevice(0);
    Ref ref;
    ref.load();
    const Case cases[] = {
        {"IQ2_XXS", T_IQ2_XXS, 256, 66, 640, 2560}, {"IQ2_XS", T_IQ2_XS, 256, 74, 640, 2560}, {"IQ2_S", T_IQ2_S, 256, 82, 640, 2560},
        {"IQ3_XXS", T_IQ3_XXS, 256, 98, 640, 2560}, {"IQ3_S", T_IQ3_S, 256, 110, 640, 2560}, {"IQ4_NL", T_IQ4_NL, 32, 18, 2560, 320},
        {"Q2_0", T_Q2_0, 64, 18, 2560, 320}, {"Q4_0", T_Q4_0, 32, 18, 2560, 320},
    };
    int bad = 0;
    for (const Case& c : cases) {
        for (int T : {1, 2, 3, 4, 6, 8, 12, 16}) bad += run_case(c, ref, T, false, true) >= 1e-4;
        if (c.type != T_Q4_0) bad += run_case(c, ref, 0, true, true) >= 3e-3;   // the draft layer's Q4_0 has no prompt-path job
    }
    std::printf(bad ? "NAPAKE: %d\n" : "vse OK\n", bad);
    return bad ? 1 : 0;
}
