// gemm_bench.cu (2026-11-21): the prompt path's GEMM shapes on one P100, 2048 tokens: cublasSgemm (f32, what the prompt path uses)
// against cublasGemmEx with f16 inputs and f32 sums (COMPUTE_32F), with f16 sums (COMPUTE_16F), and COMPUTE_32F_FAST_16F; TFLOPS
// and the error against f32 (relative RMS over the output). Weights ~N(0, 0.02), activations ~N(0, 1) like a normed row.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("CUDA %s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

struct Shape { const char* name; int m, n, k, batch; };

int main() {
    const Shape shapes[] = {
        {"gate 8192x3072", 8192, 2048, 3072, 1}, {"wo 3072x8192", 3072, 2048, 8192, 1}, {"q_b 8192x1024", 8192, 2048, 1024, 1},
        {"q_a 1024x3072", 1024, 2048, 3072, 1}, {"r_down 1536x3072", 1536, 2048, 3072, 1}, {"r_up 3072x1536", 3072, 2048, 1536, 1},
        {"sh_gate 768x3072", 768, 2048, 3072, 1}, {"window S 576x2048x576 x32", 576, 2048, 576, 32},
        {"window PV 512x2048x576 x32", 512, 2048, 576, 32},
    };
    cublasHandle_t cb;
    cublasCreate(&cb);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (const Shape& s : shapes) {
        const size_t na = (size_t) s.m * s.k * s.batch, nb = (size_t) s.k * s.n * s.batch, nc = (size_t) s.m * s.n * s.batch;
        std::vector<float> ha(na), hb(nb);
        for (auto& v : ha) v = 0.02f * nd(rng);
        for (auto& v : hb) v = nd(rng);
        std::vector<__half> ha16(na), hb16(nb);
        for (size_t i = 0; i < na; ++i) ha16[i] = __float2half(ha[i]);
        for (size_t i = 0; i < nb; ++i) hb16[i] = __float2half(hb[i]);
        float *a, *b, *c, *c2;
        __half *a16, *b16, *c16;
        CK(cudaMalloc(&a, na * 4)); CK(cudaMalloc(&b, nb * 4)); CK(cudaMalloc(&c, nc * 4)); CK(cudaMalloc(&c2, nc * 4));
        CK(cudaMalloc(&a16, na * 2)); CK(cudaMalloc(&b16, nb * 2)); CK(cudaMalloc(&c16, nc * 2));
        CK(cudaMemcpy(a, ha.data(), na * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(b, hb.data(), nb * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(a16, ha16.data(), na * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(b16, hb16.data(), nb * 2, cudaMemcpyHostToDevice));
        const float one = 1.0f, zero = 0.0f;
        const __half h1 = __float2half(1.0f), h0 = __float2half(0.0f);
        const long long sa = (long long) s.m * s.k, sb = (long long) s.k * s.n, sc = (long long) s.m * s.n;
        // y [n][m] = x [n][k] . W^T, W [m][k] row-major: op(A) = W^T as the engine calls it (CUBLAS_OP_T, CUBLAS_OP_N)
        auto run = [&](int mode) -> cublasStatus_t {
            if (mode == 0)
                return cublasSgemmStridedBatched(cb, CUBLAS_OP_T, CUBLAS_OP_N, s.m, s.n, s.k, &one, a, s.k, sa, b, s.k, sb, &zero, c, s.m, sc, s.batch);
            if (mode == 1)
                return cublasGemmStridedBatchedEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, s.m, s.n, s.k, &one, a16, CUDA_R_16F, s.k, sa, b16, CUDA_R_16F, s.k, sb,
                                                  &zero, c2, CUDA_R_32F, s.m, sc, s.batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            if (mode == 2)
                return cublasGemmStridedBatchedEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, s.m, s.n, s.k, &h1, a16, CUDA_R_16F, s.k, sa, b16, CUDA_R_16F, s.k, sb,
                                                  &h0, c16, CUDA_R_16F, s.m, sc, s.batch, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
            return cublasGemmStridedBatchedEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, s.m, s.n, s.k, &one, a, CUDA_R_32F, s.k, sa, b, CUDA_R_32F, s.k, sb,
                                              &zero, c2, CUDA_R_32F, s.m, sc, s.batch, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT);
        };
        std::vector<float> ref(nc);
        const char* names[] = {"f32 Sgemm", "f16 in, f32 sums", "f16 in, f16 sums", "f32 FAST_16F"};
        std::printf("%s (2 x %d x %d x %d x %d = %.1f GFLOP)\n", s.name, s.m, s.n, s.k, s.batch, 2.0 * s.m * s.n * s.k * s.batch / 1e9);
        for (int mode = 0; mode < 4; ++mode) {
            if (run(mode) != CUBLAS_STATUS_SUCCESS) { std::printf("  %-18s not supported\n", names[mode]); continue; }
            CK(cudaDeviceSynchronize());
            const int reps = 10;
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) run(mode);
            CK(cudaDeviceSynchronize());
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
            std::vector<float> got(nc);
            if (mode == 0) CK(cudaMemcpy(ref.data(), c, nc * 4, cudaMemcpyDeviceToHost));
            else if (mode == 2) {
                std::vector<__half> g16(nc);
                CK(cudaMemcpy(g16.data(), c16, nc * 2, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < nc; ++i) got[i] = __half2float(g16[i]);
            } else CK(cudaMemcpy(got.data(), c2, nc * 4, cudaMemcpyDeviceToHost));
            double e2 = 0, r2 = 0;
            if (mode) for (size_t i = 0; i < nc; ++i) { const double d = got[i] - ref[i]; e2 += d * d; r2 += (double) ref[i] * ref[i]; }
            std::printf("  %-18s %7.3f ms  %5.2f TFLOPS  rel.err %.2e\n", names[mode], sec * 1e3, 2.0 * s.m * s.n * s.k * s.batch / sec / 1e12,
                        mode ? std::sqrt(e2 / r2) : 0.0);
        }
        cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(c2); cudaFree(a16); cudaFree(b16); cudaFree(c16);
    }
    return 0;
}
