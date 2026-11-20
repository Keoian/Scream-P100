// m40_pf_test.cu (2026-11-15): the prompt path's expert kernels on the M40 (sm_52): k_gemm4_job (p100_moe, production) against k_m40_pf
// (m40_pf_gemm.cuh, FNX_M40_PF), both against m40_rows_test.cu's CPU reference (ggml's dequantize_row_* formulas, sums in double), on
// made-up weights of every M40 type and jobs of 1 to 96 tokens as the prompt path makes them (an expert's tokens from anywhere in the
// chunk). Prints each kernel's worst relative error and its time for the same jobs. Idle cards only. Build: mikro/m40_test_gradnja.sh
// m40_pf_test (after m40_pf_gemm.cuh is in the tree, included by p100_moe.cu).
#define main m40_rows_main
#include "m40_rows_test.cu"
#undef main
#ifndef FNX_M40_PF_GEMM
#include "m40_pf_gemm.cuh"
#endif

static void run_pf(const Case& c, const Ref& ref, const std::vector<int>& sizes, int reps) {
    std::mt19937 g(c.type * 31 + 5);
    const size_t row_bytes = (size_t) c.K / c.block_values * c.block_bytes, stride = (size_t) c.N * row_bytes;
    const int E = 64, C = 4096;
    std::vector<uint8_t> w((size_t) E * stride);
    for (auto& v : w) v = (uint8_t) g();
    for (size_t r = 0; r + c.block_bytes <= w.size(); r += c.block_bytes) {
        const uint16_t d = f2h(0.01f + 0.02f * (float) (g() % 1000) / 1000.0f);
        std::memcpy(&w[r], &d, 2);
    }
    std::vector<uint16_t> xh((size_t) C * c.K);
    std::vector<float> x(xh.size());
    for (size_t i = 0; i < x.size(); ++i) xh[i] = f2h(((float) (g() % 2001) - 1000.0f) / 500.0f), x[i] = h2f_host(xh[i]);
    std::vector<BigJob> big;
    int pair = 0;
    long long flops = 0;
    for (int r = 0; r < reps; ++r)
        for (int n : sizes) {
            BigJob j{};
            j.w = (int) (big.size() * 7 + 3) % E;
            j.n = n;
            for (int t = 0; t < n; ++t) j.y[t] = (int) (g() % C), j.out[t] = pair++;
            big.push_back(j);
            flops += 2ll * n * c.N * c.K;
        }
    const int njobs = (int) big.size();
    cudaStream_t s;
    cudaStreamCreate(&s);
    uint8_t* dw;
    __half* dx;
    float *dy0, *dy1;
    BigJob* dj;
    int* dn;
    cudaMalloc(&dw, w.size());
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dx, xh.size() * 2);
    cudaMemcpy(dx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&dy0, (size_t) pair * c.N * 4);
    cudaMalloc(&dy1, (size_t) pair * c.N * 4);
    cudaMemset(dy0, 0, (size_t) pair * c.N * 4);
    cudaMemset(dy1, 0, (size_t) pair * c.N * 4);
    cudaMalloc(&dj, big.size() * sizeof(BigJob));
    cudaMemcpy(dj, big.data(), big.size() * sizeof(BigJob), cudaMemcpyHostToDevice);
    cudaMalloc(&dn, 4);
    cudaMemcpy(dn, &njobs, 4, cudaMemcpyHostToDevice);
    DW A;
    A.p = dw;
    A.type = c.type;
    A.n_in = c.K;
    A.n_out = c.N;
    A.row_bytes = row_bytes;
    auto old_k = [&] { p100_moe(A, stride, dj, dn, njobs, (const uint16_t*) dx, dy0, s); };
    auto new_k = [&] {
        if (!m40_pf_run(A, stride, dj, dn, njobs, (const uint16_t*) dx, dy1, s)) { std::printf("%s: m40_pf_run ne zna\n", c.name); std::exit(1); }
    };
    old_k();
    new_k();
    if (cudaStreamSynchronize(s) != cudaSuccess) { std::printf("%s: CUDA %s\n", c.name, cudaGetErrorString(cudaGetLastError())); std::exit(1); }
    std::vector<float> y0((size_t) pair * c.N), y1(y0.size());
    cudaMemcpy(y0.data(), dy0, y0.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(y1.data(), dy1, y1.size() * 4, cudaMemcpyDeviceToHost);
    // the reference on the first jobs of each size (all rows), against both
    double w0 = 0, w1 = 0, sc = 0;
    std::vector<float> wr((size_t) c.K);
    for (size_t ji = 0; ji < sizes.size(); ++ji) {
        const BigJob& j = big[ji];
        for (int r = 0; r < c.N; ++r) {
            ref.row(c.type, w.data() + (size_t) j.w * stride + (size_t) r * row_bytes, c.K, wr.data());
            for (int t = 0; t < j.n; ++t) {
                double acc = 0;
                for (int k = 0; k < c.K; ++k) acc += (double) wr[(size_t) k] * x[(size_t) j.y[t] * c.K + k];
                const size_t o = (size_t) j.out[t] * c.N + r;
                sc = std::max(sc, std::fabs(acc));
                w0 = std::max(w0, std::fabs(acc - (double) y0[o]));
                w1 = std::max(w1, std::fabs(acc - (double) y1[o]));
            }
        }
    }
    auto timed = [&](auto&& f) {
        f();
        cudaStreamSynchronize(s);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) f();
        cudaStreamSynchronize(s);
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 20;
    };
    const double u0 = timed(old_k), u1 = timed(new_k);
    const double r0 = w0 / std::max(sc, 1e-30), r1 = w1 / std::max(sc, 1e-30);
    std::printf("%-8s N %4d K %4d, %3d jobs: k_gemm4_job %8.1f us (%6.0f GFLOP/s, rel %.1e) | k_m40_pf %8.1f us (%6.0f GFLOP/s, rel %.1e) %s  x%.2f\n",
                c.name, c.N, c.K, njobs, u0, flops / u0 / 1e3, r0, u1, flops / u1 / 1e3, r1, r1 < 1e-5 ? "OK" : "NAPAKA", u0 / u1);
    cudaFree(dw), cudaFree(dx), cudaFree(dy0), cudaFree(dy1), cudaFree(dj), cudaFree(dn);
    cudaStreamDestroy(s);
}

int main(int argc, char** argv) {
    cudaSetDevice(argc > 1 ? atoi(argv[1]) : 0);
    Ref ref;
    ref.load();
    const Case cases[] = {
        {"IQ2_XXS", T_IQ2_XXS, 256, 66, 640, 2560}, {"IQ2_XS", T_IQ2_XS, 256, 74, 640, 2560}, {"IQ2_S", T_IQ2_S, 256, 82, 640, 2560},
        {"IQ3_XXS", T_IQ3_XXS, 256, 98, 640, 2560}, {"IQ3_S", T_IQ3_S, 256, 110, 640, 2560}, {"IQ4_NL", T_IQ4_NL, 32, 18, 2560, 320},
        {"Q2_0", T_Q2_0, 64, 18, 2560, 320}, {"Q4_0", T_Q4_0, 32, 18, 2560, 320},
    };
    // a long read's chunk: about 80 tokens an expert (96-token jobs, the rest), and the small jobs of rare experts
    const std::vector<int> mix = {96, 96, 96, 80, 72, 64, 57, 49, 48, 40, 33, 24, 17, 9, 5, 1};
    const std::vector<int> full = {96, 96, 96, 96};
    const std::vector<int> small = {12, 8, 6, 5, 4, 3, 2, 1};
    if (argc > 2) {   // one case, full jobs only (for nvprof): m40_pf_test CARD CASE
        run_pf(cases[atoi(argv[2])], ref, full, 32);
        return 0;
    }
    for (const Case& c : cases) {
        run_pf(c, ref, mix, 16);
        run_pf(c, ref, full, 32);
        run_pf(c, ref, small, 32);
    }
    return 0;
}
