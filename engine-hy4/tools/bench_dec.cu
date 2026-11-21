// bench_dec.cu (2026-11-20): v1's writing kernels against hy4_dec.cuh's on random data, one card: same results, time a call
#include "hy4_kern.cuh"
#include "hy4_pf.cuh"
#include "hy4_dec.cuh"
#include "hy4_pf2.cuh"

#include <random>

using namespace hy4;

static std::mt19937 rng(7);
static float urand() { return std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng); }
template <class T> static T* up(const std::vector<T>& v) {
    T* d = nullptr;
    CK(cudaMalloc(&d, v.size() * sizeof(T)));
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static std::vector<T> down(const T* d, size_t n) {
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
template <class F> static double time_us(F f, int reps = 200) {
    f();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i = 0; i < reps; ++i) f();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    return ms * 1000.0 / reps;
}
static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, (double) std::fabs(a[i]));
        d = std::max(d, (double) std::fabs(a[i] - b[i]));
    }
    return d / std::max(m, 1e-30);
}
static std::vector<uint8_t> iq3_bytes(size_t rows, int cols) {   // random IQ3_XXS rows with sane scales
    std::vector<uint8_t> v(rows * (cols / 256) * 98);
    for (auto& c : v) c = (uint8_t) rng();
    for (size_t b = 0; b < v.size() / 98; ++b) {
        const __half h = __float2half(0.002f * (1.0f + urand()));
        std::memcpy(&v[b * 98], &h, 2);
    }
    return v;
}

// what holds k_iq3_rows2 at ~50 us (the plain read of its bytes takes 15): MODE 0 as it is, 1 no shared-memory tables (the grid
// word and the signs made from the index), 2 no weight loads (the bytes made from the row and block), 3 neither
template <int MODE>
__global__ void __launch_bounds__(256) k_iq3_x(const EJob* jobs, const float* __restrict__ x, int ldx, float* y, int ldy, int rows, int cols) {
    __shared__ uint32_t grid[256];
    __shared__ uint8_t sg[128];
    const EJob J = jobs[blockIdx.y];
    grid[threadIdx.x] = d_iq3xxs_grid[threadIdx.x];
    if (threadIdx.x < 128) sg[threadIdx.x] = d_ksigns_iq2xs[threadIdx.x];
    __syncthreads();
    const float* xr = x + (size_t) J.in * ldx;
    const int warp = threadIdx.x >> 5, l = threadIdx.x & 31, ib = l >> 2, l4 = l & 3;
    const int r0 = blockIdx.x * IQ2_BR + warp * IQ2_R, nb = cols / 256;
    float acc[IQ2_R];
#pragma unroll
    for (int a = 0; a < IQ2_R; ++a) acc[a] = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float4 xa = __ldg((const float4*) (xr + b * 256 + l * 8)), xb = __ldg((const float4*) (xr + b * 256 + l * 8 + 4));
        uint16_t dq[IQ2_R], qq[IQ2_R], a0[IQ2_R], a1[IQ2_R];
#pragma unroll
        for (int a = 0; a < IQ2_R; ++a) {
            const int r = min(r0 + a, rows - 1);
            if (MODE & 2) {
                dq[a] = 0x1400;
                qq[a] = (uint16_t) (r * 131 + b * 17 + l);
                a0[a] = (uint16_t) (r * 7 + b);
                a1[a] = (uint16_t) (r + ib);
            } else {
                const uint16_t* blk = (const uint16_t*) (J.m + ((size_t) r * nb + b) * 98);
                dq[a] = __ldg(blk);
                qq[a] = __ldg(blk + 1 + l);
                a0[a] = __ldg(blk + 33 + 2 * ib);
                a1[a] = __ldg(blk + 34 + 2 * ib);
            }
        }
#pragma unroll
        for (int a = 0; a < IQ2_R; ++a) {
            const uint32_t aux = (uint32_t) a0[a] | ((uint32_t) a1[a] << 16);
            const float db = __half2float(__ushort_as_half(dq[a])) * (0.5f + (float) (aux >> 28)) * 0.5f;
            uint32_t s, g1, g2;
            if (MODE & 1) {
                s = (aux >> (7 * l4)) & 255;
                g1 = (qq[a] & 255) * 0x01010101u;
                g2 = (qq[a] >> 8) * 0x01010101u;
            } else {
                s = sg[(aux >> (7 * l4)) & 127];
                g1 = grid[qq[a] & 255];
                g2 = grid[qq[a] >> 8];
            }
            float v = 0;
            v += (float) (g1 & 255) * (s & 1 ? -xa.x : xa.x);
            v += (float) ((g1 >> 8) & 255) * (s & 2 ? -xa.y : xa.y);
            v += (float) ((g1 >> 16) & 255) * (s & 4 ? -xa.z : xa.z);
            v += (float) (g1 >> 24) * (s & 8 ? -xa.w : xa.w);
            v += (float) (g2 & 255) * (s & 16 ? -xb.x : xb.x);
            v += (float) ((g2 >> 8) & 255) * (s & 32 ? -xb.y : xb.y);
            v += (float) ((g2 >> 16) & 255) * (s & 64 ? -xb.z : xb.z);
            v += (float) (g2 >> 24) * (s & 128 ? -xb.w : xb.w);
            acc[a] += db * v;
        }
    }
#pragma unroll
    for (int a = 0; a < IQ2_R; ++a) {
        const float s = warp_sum(acc[a]);
        if (l == 0 && r0 + a < rows) y[(size_t) J.out * ldy + r0 + a] = s;
    }
}
static void bench_iq3x() {
    const int rows = FFE / 2, cols = LAT, nj = 16;
    const size_t mb = (size_t) rows * (cols / 256) * 98;
    uint8_t* m = up(iq3_bytes((size_t) rows * nj, cols));
    std::vector<EJob> jh(nj);
    for (int j = 0; j < nj; ++j) jh[j] = {m + (size_t) j * mb, 0, j};
    EJob* jobs = up(jh);
    std::vector<float> xh((size_t) cols);
    for (auto& v : xh) v = urand();
    float* x = up(xh);
    float* y;
    CK(cudaMalloc(&y, (size_t) nj * rows * 4));
    const double t0 = time_us([&]() { k_iq3_x<0><<<dim3(rows / IQ2_BR, nj), 256>>>(jobs, x, cols, y, rows, rows, cols); });
    const double t1 = time_us([&]() { k_iq3_x<1><<<dim3(rows / IQ2_BR, nj), 256>>>(jobs, x, cols, y, rows, rows, cols); });
    const double t2 = time_us([&]() { k_iq3_x<2><<<dim3(rows / IQ2_BR, nj), 256>>>(jobs, x, cols, y, rows, rows, cols); });
    const double t3 = time_us([&]() { k_iq3_x<3><<<dim3(rows / IQ2_BR, nj), 256>>>(jobs, x, cols, y, rows, rows, cols); });
    std::printf("iq3 trial (us, card time below in nvprof): as is %.1f, no tables %.1f, no reads %.1f, neither %.1f\n", t0, t1, t2, t3);
    // the half2 kernel against v1's (gate shape and down shape, real inputs in a plausible range)
    for (int pass = 0; pass < 2; ++pass) {
        const int rr = pass ? LAT : FFE / 2, cc = pass ? FFE / 2 : LAT;
        const size_t mb2 = (size_t) rr * (cc / 256) * 98;
        uint8_t* m2 = up(iq3_bytes((size_t) rr * nj, cc));
        std::vector<EJob> jh2(nj);
        for (int j = 0; j < nj; ++j) jh2[j] = {m2 + (size_t) j * mb2, pass ? j : 0, j};
        EJob* jobs2 = up(jh2);
        std::vector<float> xh2((size_t) nj * cc);
        for (auto& v : xh2) v = urand() * 3.0f;
        xh2[5] = 40.0f;   // an outlier in block 0
        float* x2 = up(xh2);
        float *ya, *yb;
        CK(cudaMalloc(&ya, (size_t) nj * rr * 4));
        CK(cudaMalloc(&yb, (size_t) nj * rr * 4));
        const double ta = time_us([&]() { k_iq3_rows<<<dim3(rr / IQ_BR, nj), 256>>>(jobs2, x2, cc, ya, rr, rr, cc); });
        const double tb = time_us([&]() { k_iq3_rows5<4><<<dim3(rr / 32, nj), 256>>>(jobs2, x2, cc, yb, rr, rr, cc); });
        const double tc = time_us([&]() { k_iq3_rows_h2<4><<<dim3(rr / 32, nj), 256>>>(jobs2, x2, cc, yb, rr, rr, cc); });
        const double rc = rel(down(ya, (size_t) nj * rr), down(yb, (size_t) nj * rr));
        const double td = time_us([&]() { k_iq3_rows_h2<8><<<dim3(rr / 64, nj), 256>>>(jobs2, x2, cc, yb, rr, rr, cc); });
        const double tf = time_us([&]() { k_iq3_rows_h2<2><<<dim3(rr / 16, nj), 256>>>(jobs2, x2, cc, yb, rr, rr, cc); });
        std::printf("iq3 half2 %s: v1 %.1f, rows5 %.1f, h2<4> %.1f (rel %.2e), h2<8> %.1f, h2<2> %.1f us\n", pass ? "down" : "gate", ta, tb, tc, rc, td, tf);
    }
}

// the prompt path's experts: E experts of n tokens each out of C, gate shape (768 x 1536, input = token row) and down shape
// (1536 x 768, input = pair row)
static void bench_pf(int E, int nlo, int nhi) {
    const int C = 2048;
    for (int pass = 0; pass < 2; ++pass) {
        const int rows = pass ? LAT : FFE / 2, cols = pass ? FFE / 2 : LAT;
        const size_t mb = (size_t) rows * (cols / 256) * 98;
        uint8_t* m = up(iq3_bytes((size_t) rows * E, cols));
        std::vector<int2> lh;
        std::vector<PJob> jh(E);
        int pair = 0;
        for (int j = 0; j < E; ++j) {
            const int n = j == 0 ? 150 : nlo + (int) (rng() % (unsigned) (nhi - nlo));
            jh[j] = {m + (size_t) j * mb, (int) lh.size(), n};
            for (int k = 0; k < n; ++k) lh.push_back(make_int2((int) (rng() % C), pair++));
        }
        PJob* jobs = up(jh);
        int2* lists = up(lh);
        const int xrows = pass ? pair : C;
        std::vector<float> xh((size_t) xrows * cols);
        for (auto& v : xh) v = urand();
        float* x = up(xh);
        float *y1, *y2;
        CK(cudaMalloc(&y1, (size_t) pair * rows * 4));
        CK(cudaMalloc(&y2, (size_t) pair * rows * 4));
        CK(cudaMemset(y1, 0, (size_t) pair * rows * 4));
        CK(cudaMemset(y2, 0, (size_t) pair * rows * 4));
        double t1, t2;
        if (pass) {
            t1 = time_us([&]() { k_iq3_pf3<3, 8><<<dim3(rows / 8, E), 256>>>(jobs, lists, x, cols, y1, rows, rows, 1); }, 20);
            t2 = time_us([&]() { k_iq3_gemm<3><<<dim3(rows / GM, E), 256>>>(jobs, lists, x, cols, y2, rows, rows, 1); }, 20);
        } else {
            t1 = time_us([&]() { k_iq3_pf3<6, 4><<<dim3(rows / 8, E), 256>>>(jobs, lists, x, cols, y1, rows, rows, 0); }, 20);
            t2 = time_us([&]() { k_iq3_gemm<6><<<dim3(rows / GM, E), 256>>>(jobs, lists, x, cols, y2, rows, rows, 0); }, 20);
        }
        const double fl = 2.0 * pair * rows * cols / 1e6;   // MFLOP
        const double t3 = time_us([&]() { if (pass) k_iq3_gemm2<3><<<dim3(rows / GM, E), 256>>>(jobs, lists, x, cols, y2, rows, rows, 1); else k_iq3_gemm2<6><<<dim3(rows / GM, E), 256>>>(jobs, lists, x, cols, y2, rows, rows, 0); }, 20);
        std::printf("gemm2 %.0f us (%.2f TFLOPS), rel %.2e\n", t3, fl / t3, rel(down(y1, (size_t) pair * rows), down(y2, (size_t) pair * rows)));
        const double t4 = time_us([&]() { if (pass) k_iq3_gemm3<3><<<dim3(rows / G3M, E), 256>>>(jobs, lists, x, cols, y2, rows, rows, 1); else k_iq3_gemm3<6><<<dim3(rows / G3M, E), 256>>>(jobs, lists, x, cols, y2, rows, rows, 0); }, 20);
        std::printf("gemm3 %.0f us (%.2f TFLOPS), rel %.2e\n", t4, fl / t4, rel(down(y1, (size_t) pair * rows), down(y2, (size_t) pair * rows)));
        std::printf("pf %s: pf3 %.0f us (%.2f TFLOPS), gemm %.0f us (%.2f TFLOPS), rel %.2e\n", pass ? "down" : "gate", t1, fl / t1,
                    t2, fl / t2, rel(down(y1, (size_t) pair * rows), down(y2, (size_t) pair * rows)));
        cudaFree(m); cudaFree(jobs); cudaFree(lists); cudaFree(x); cudaFree(y1); cudaFree(y2);
    }
}

// v2: gemm3 / gemm3 skip / gemm4 / gemm4 skip / gemm5 on experts with nlo .. nhi tokens each: time, useful TFLOPS, error against
// gemm3 (f32) and gemm5 against gemm4 (should be 0: the same f16 sums)
static void bench_pf4(int E, int nlo, int nhi) {
    const int C = 2048;
    for (int pass = 0; pass < 2; ++pass) {
        const int rows = pass ? LAT : FFE / 2, cols = pass ? FFE / 2 : LAT;
        const size_t mb = (size_t) rows * (cols / 256) * 98;
        uint8_t* m = up(iq3_bytes((size_t) rows * E, cols));
        std::vector<int2> lh;
        std::vector<PJob> jh(E);
        int pair = 0;
        for (int j = 0; j < E; ++j) {
            const int n = nlo + (int) (rng() % (unsigned) (nhi - nlo));
            jh[j] = {m + (size_t) j * mb, (int) lh.size(), n};
            for (int k = 0; k < n; ++k) lh.push_back(make_int2((int) (rng() % C), pair++));
        }
        PJob* jobs = up(jh);
        int2* lists = up(lh);
        const int xrows = pass ? pair : C;
        std::vector<float> xh((size_t) xrows * cols);
        for (auto& v : xh) v = urand() * (pass ? 4.0f : 1.0f);
        float* x = up(xh);
        float *y[5];
        for (int i = 0; i < 5; ++i) { CK(cudaMalloc(&y[i], (size_t) pair * rows * 4)); CK(cudaMemset(y[i], 0, (size_t) pair * rows * 4)); }
        const double fl = 2.0 * pair * rows * cols / 1e6;
        double t[5];
#define RUN(K, i) t[i] = time_us([&]() { if (pass) K<3><<<dim3(rows / G3M, E), 256>>>(jobs, lists, x, cols, y[i], rows, rows, 1); \
                                       else K<6><<<dim3(rows / G3M, E), 256>>>(jobs, lists, x, cols, y[i], rows, rows, 0); }, 20)
#define RUNS(K, i) t[i] = time_us([&]() { if (pass) K<3, true><<<dim3(rows / G3M, E), 256>>>(jobs, lists, x, cols, y[i], rows, rows, 1); \
                                       else K<6, true><<<dim3(rows / G3M, E), 256>>>(jobs, lists, x, cols, y[i], rows, rows, 0); }, 20)
        RUN(k_iq3_gemm3, 0); RUNS(k_iq3_gemm3, 1); RUN(k_iq3_gemm4, 2); RUNS(k_iq3_gemm4, 3); RUNS(k_iq3_gemm5, 4);
#undef RUN
#undef RUNS
        const auto r0 = down(y[0], (size_t) pair * rows);
        std::printf("%s, %d pairs (%d .. %d a expert):\n", pass ? "down" : "gate", pair, nlo, nhi - 1);
        const char* nm[5] = {"gemm3", "gemm3 skip", "gemm4", "gemm4 skip", "gemm5"};
        for (int i = 0; i < 5; ++i)
            std::printf("  %-11s %7.0f us  %5.2f TFLOPS  rel vs gemm3 %.2e%s\n", nm[i], t[i], fl / t[i], rel(r0, down(y[i], (size_t) pair * rows)),
                        i == 4 ? (rel(down(y[2], (size_t) pair * rows), down(y[4], (size_t) pair * rows)) == 0.0 ? "  (= gemm4 bit for bit)" : "  (differs from gemm4)") : "");
        cudaFree(m); cudaFree(jobs); cudaFree(lists); cudaFree(x);
        for (int i = 0; i < 5; ++i) cudaFree(y[i]);
    }
}

int main() {
    CK(cudaSetDevice(0));
    if (getenv("SAMO_X")) {
        bench_iq3x();
        return 0;
    }
    if (getenv("SAMO_PF4")) {   // v2: the expert kernels at a chunk's token counts (512 experts, 2048 tokens x 16 picks)
        bench_pf4(512, 5, 125);
        bench_pf4(512, 30, 100);
        return 0;
    }
    if (getenv("SAMO_PF")) {
        bench_pf(256, 20, 60);
        bench_pf(512, 30, 100);
        return 0;
    }
    // ---- experts: 16 jobs, gate (768 x 1536, one input) and down (1536 x 768, input row j)
    for (int pass = 0; pass < 2; ++pass) {
        const int rows = pass ? LAT : FFE / 2, cols = pass ? FFE / 2 : LAT, nj = 16;
        const size_t mb = (size_t) rows * (cols / 256) * 98;
        uint8_t* m = up(iq3_bytes((size_t) rows * nj, cols));
        std::vector<EJob> jh(nj);
        for (int j = 0; j < nj; ++j) jh[j] = {m + (size_t) j * mb, pass ? j : 0, j};
        EJob* jobs = up(jh);
        std::vector<float> xh((size_t) nj * cols);
        for (auto& v : xh) v = urand();
        float* x = up(xh);
        float *y1, *y2;
        CK(cudaMalloc(&y1, (size_t) nj * rows * 4));
        CK(cudaMalloc(&y2, (size_t) nj * rows * 4));
        auto f1 = [&]() { k_iq3_rows<<<dim3(rows / IQ_BR, nj), 256>>>(jobs, x, cols, y1, rows, rows, cols); };
        auto f2 = [&]() { k_iq3_rows2<<<dim3(rows / IQ2_BR, nj), 256>>>(jobs, x, cols, y2, rows, rows, cols); };
        const double t1 = time_us(f1), t2 = time_us(f2);
        const double gb = (double) mb * nj / 1e3;
        std::printf("iq3 %s: v1 %.1f us (%.0f GB/s), v1g %.1f us (%.0f GB/s), rel %.2e\n", pass ? "down" : "gate", t1, gb / t1, t2, gb / t2,
                    rel(down(y1, (size_t) nj * rows), down(y2, (size_t) nj * rows)));
        auto f3 = [&]() { k_iq3_rows3<4><<<dim3(rows / 32, nj), 256>>>(jobs, x, cols, y2, rows, rows, cols); };
        auto f4 = [&]() { k_iq3_rows3<8><<<dim3(rows / 64, nj), 256>>>(jobs, x, cols, y2, rows, rows, cols); };
        auto f5 = [&]() { k_iq3_rows3<2><<<dim3(rows / 16, nj), 256>>>(jobs, x, cols, y2, rows, rows, cols); };
        const double t3 = time_us(f3);
        const double r3 = rel(down(y1, (size_t) nj * rows), down(y2, (size_t) nj * rows));
        const double t4 = time_us(f4), t5 = time_us(f5);
        std::printf("iq3_3 %s: R4 %.1f us (%.0f GB/s, rel %.2e), R8 %.1f us, R2 %.1f us\n", pass ? "down" : "gate", t3, gb / t3, r3, t4, t5);
        auto f6 = [&]() { k_iq3_rows4<4><<<dim3(rows / 32, nj), 256>>>(jobs, x, cols, y2, rows, rows, cols); };
        auto f7 = [&]() { k_iq3_rows4<8><<<dim3(rows / 64, nj), 256>>>(jobs, x, cols, y2, rows, rows, cols); };
        auto f8 = [&]() { k_iq3_rows4<2><<<dim3(rows / 16, nj), 256>>>(jobs, x, cols, y2, rows, rows, cols); };
        const double t6 = time_us(f6);
        const double r6 = rel(down(y1, (size_t) nj * rows), down(y2, (size_t) nj * rows));
        const double t7 = time_us(f7), t8 = time_us(f8);
        std::printf("iq3_4 %s: R4 %.1f us (rel %.2e), R8 %.1f us, R2 %.1f us\n", pass ? "down" : "gate", t6, r6, t7, t8);
        auto f9 = [&]() { k_iq3_rows5<4><<<dim3(rows / 32, nj), 256>>>(jobs, x, cols, y2, rows, rows, cols); };
        auto f10 = [&]() { k_iq3_rows5<2><<<dim3(rows / 16, nj), 256>>>(jobs, x, cols, y2, rows, rows, cols); };
        auto f11 = [&]() { k_iq3_rows5<8><<<dim3(rows / 64, nj), 256>>>(jobs, x, cols, y2, rows, rows, cols); };
        const double t9 = time_us(f9);
        const double r9 = rel(down(y1, (size_t) nj * rows), down(y2, (size_t) nj * rows));
        const double t10 = time_us(f10), t11 = time_us(f11);
        int* dummy;
        CK(cudaMalloc(&dummy, 4));
        const double tf = time_us([&]() { k_ld_floor<<<224, 256>>>((const int4*) m, mb * nj / 16, dummy); });
        std::printf("iq3_5 %s: R4 %.1f us (%.0f GB/s, rel %.2e), R2 %.1f us, R8 %.1f us; reads only %.1f us (%.0f GB/s)\n", pass ? "down" : "gate",
                    t9, gb / t9, r9, t10, t11, tf, gb / tf);
    }
    if (getenv("SAMO_IQ3")) return 0;
    // ---- attention
    struct Case { int H, window, p, list; const char* name; };
    const Case cs[] = {{32, 1, 252, 0, "okno 253"}, {32, 1, 3000, 0, "okno 512"}, {16, 0, 252, 0, "redka 253"}, {16, 0, 2047, 0, "redka 2048"},
                       {16, 0, 9000, 1, "redka izbor 2048"}};
    const int ctx = 16384;
    std::vector<__half> ch((size_t) ctx * KROW);
    for (auto& v : ch) v = __float2half(urand());
    __half* cache = up(ch);
    float *opart, *mlpart;
    CK(cudaMalloc(&opart, (size_t) MLA_PART * KVL * 4));
    CK(cudaMalloc(&mlpart, (size_t) MLA_PART * 2 * 4));
    for (const Case& c : cs) {
        for (int T : {1, 9}) {
            std::vector<float> qh((size_t) T * c.H * KROW), sk(c.H);
            for (auto& v : qh) v = urand() * 3.0f;
            for (auto& v : sk) v = urand();
            std::vector<int> ph(T), nh(T, -1), sh((size_t) T * MAXSEL);
            for (int t = 0; t < T; ++t) ph[t] = c.p + t;
            if (c.list)
                for (int t = 0; t < T; ++t) {
                    nh[t] = MAXSEL - 4 + t % 4;
                    for (int j = 0; j < nh[t]; ++j) sh[(size_t) t * MAXSEL + j] = (int) (rng() % (unsigned) (ph[t] + 1));
                }
            float *q = up(qh), *s = up(sk);
            int *p = up(ph), *ns = up(nh), *sl = up(sh);
            float *o1, *o2;
            CK(cudaMalloc(&o1, (size_t) T * c.H * KVL * 4));
            CK(cudaMalloc(&o2, (size_t) T * c.H * KVL * 4));
            const float scale = 1.0f / 16.0f;
            int nmax = c.list ? MAXSEL : c.window ? std::min(SWA, c.p + T) : c.p + T;
            auto f1 = [&]() {
                k_mla_attn<<<dim3(c.H, T), 256, (size_t) (KROW + MAXSEL) * 4>>>(q, cache, p, c.window, c.list ? sl : nullptr, c.list ? ns : nullptr,
                                                                               s, scale, o1, c.H);
            };
            auto f2 = [&]() { mla_dec(q, cache, p, c.window, c.list ? sl : nullptr, c.list ? ns : nullptr, s, scale, o2, c.H, T, nmax, opart, mlpart, 0); };
            const double t1 = time_us(f1, 50);
            std::printf("mla %s T=%d: v1 %.1f us", c.name, T, t1);
            for (int sub : {16, 32}) {
                g_msub = sub;
                const double t2 = time_us(f2, 50);
                std::printf(", v1g/%d %.1f us (rel %.2e)", sub, t2, rel(down(o1, (size_t) T * c.H * KVL), down(o2, (size_t) T * c.H * KVL)));
            }
            auto f3 = [&]() { mla_dec2(q, cache, p, c.window, c.list ? sl : nullptr, c.list ? ns : nullptr, s, scale, o2, c.H, T, nmax, opart, mlpart, 0); };
            for (int sub : {16, 32}) {
                g_msub = sub;
                const double t3 = time_us(f3, 50);
                std::printf(", dec2/%d %.1f us (rel %.2e)", sub, t3, rel(down(o1, (size_t) T * c.H * KVL), down(o2, (size_t) T * c.H * KVL)));
            }
            std::printf("\n");
            cudaFree(q); cudaFree(s); cudaFree(p); cudaFree(ns); cudaFree(sl); cudaFree(o1); cudaFree(o2);
        }
    }
    // ---- router: logits and picks
    {
        std::vector<float> wh((size_t) NEXP * N), xh(N), bh(NEXP);
        for (auto& v : wh) v = urand() * 0.05f;
        for (auto& v : xh) v = urand();
        for (auto& v : bh) v = urand() * 0.1f;
        Mat W;
        W.type = 0; W.rows = NEXP; W.cols = N; W.d = up(wh);
        float *x = up(xh), *b = up(bh), *l1, *l2, *w1, *w2;
        int *i1, *i2;
        CK(cudaMalloc(&l1, NEXP * 4)); CK(cudaMalloc(&l2, NEXP * 4));
        CK(cudaMalloc(&w1, TOPK * 4)); CK(cudaMalloc(&w2, TOPK * 4));
        CK(cudaMalloc(&i1, TOPK * 4)); CK(cudaMalloc(&i2, TOPK * 4));
        auto g1 = [&]() { k_gemv32<<<(NEXP + GW_BR - 1) / GW_BR, 256, 128 * 4>>>((const float*) W.d, NEXP, N, x, N, l1, NEXP, 1); };
        auto g2 = [&]() { gemv32_t1(W, x, l2, 0); };
        const double t1 = time_us(g1), t2 = time_us(g2);
        std::printf("router gemv: v1 %.1f us, v1g %.1f us (%.0f GB/s), rel %.2e\n", t1, t2, NEXP * N * 4 / 1e3 / t2,
                    rel(down(l1, NEXP), down(l2, NEXP)));
        auto r1 = [&]() { k_route<<<1, 32>>>(l1, b, 3.998f, i1, w1); };
        auto r2 = [&]() { k_route2<<<1, 32>>>(l1, b, 3.998f, i2, w2); };
        const double u1 = time_us(r1), u2 = time_us(r2);
        const auto a1 = down(i1, TOPK), a2 = down(i2, TOPK);
        const auto b1 = down(w1, TOPK), b2 = down(w2, TOPK);
        int same = 0;
        for (int j = 0; j < TOPK; ++j) same += a1[j] == a2[j] && b1[j] == b2[j];
        std::printf("k_route: v1 %.1f us, v1g %.1f us, same %d / %d\n", u1, u2, same, TOPK);
        auto r3 = [&]() { k_route3<<<1, NEXP>>>(l1, b, 3.998f, i2, w2); };
        const double u3 = time_us(r3);
        const auto c2 = down(i2, TOPK);
        const auto d2 = down(w2, TOPK);
        same = 0;
        for (int j = 0; j < TOPK; ++j) same += a1[j] == c2[j] && b1[j] == d2[j];
        std::printf("k_route3: %.1f us, same %d / %d\n", u3, same, TOPK);
        auto r4 = [&]() { k_route4<<<1, 256>>>(l1, b, 3.998f, i2, w2); };
        const double u4 = time_us(r4);
        const auto e2 = down(i2, TOPK);
        const auto f2 = down(w2, TOPK);
        same = 0;
        for (int j = 0; j < TOPK; ++j) same += a1[j] == e2[j] && b1[j] == f2[j];
        std::printf("k_route4: %.1f us, same %d / %d\n", u4, same, TOPK);
    }
    // ---- Q8S at T = 1: bandwidth by shape
    {
        struct S { int rows, cols; const char* name; };
        const S ss[] = {{1024, 3072, "q_a"}, {8192, 1024, "q_b"}, {576, 3072, "kv_a"}, {8192, 3072, "gate"}, {3072, 8192, "wo"},
                        {1536, 3072, "r_down"}, {3072, 1536, "r_up"}, {768, 3072, "sh_gate"}, {3072, 768, "sh_down"}};
        std::vector<float> xh(8192);
        for (auto& v : xh) v = urand();
        float* x = up(xh);
        float* y;
        CK(cudaMalloc(&y, 8192 * 4));
        for (const S& s : ss) {
            std::vector<int8_t> qh((size_t) s.rows * s.cols);
            for (auto& v : qh) v = (int8_t) rng();
            std::vector<__half> dh((size_t) s.rows * s.cols / 32);
            for (auto& v : dh) v = __float2half(0.01f);
            Mat W;
            W.type = 8; W.rows = s.rows; W.cols = s.cols; W.q = up(qh); W.d = up(dh);
            const double t = time_us([&]() { gemv(W, x, s.cols, y, s.rows, 1, 0); });
            std::printf("gemv8_t1 %s %dx%d: %.1f us, %.0f GB/s\n", s.name, s.rows, s.cols, t, (double) s.rows * s.cols * 1.0625 / 1e3 / t);
            cudaFree((void*) W.q); cudaFree((void*) W.d);
        }
    }
    return 0;
}
