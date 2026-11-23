// cpu_exact_gpu_test.cu: FNX_CPU_EXACT against the card itself (a P100: the row kernels and HFMA2 are the ones the engine runs).
//   cpu_exact_gpu_test [DEV]
//   1. the card's __hfma2, __hmul2 and __float2half_rn against the CPU's oracle (src/cpu_experts.cpp), lane by lane, bit for bit;
//   2. random experts of every type the exact path covers (gate|up Q4_K/Q5_K, down IQ4_NL/Q5_1/Q8_0), 1 to 16 tokens: the card's row
//      kernels (p100_moe_rows_phase, phase 0) against the CPU's exact kernels; the SwiGLU of the handoff (k_cpu_swiglu, through
//      p100_cpu_swiglu_probe) against the card's own two (k_swiglu_after, swiglu_h16); then the card's down rows of the card's h
//      against the CPU's of the same h. Every value bit for bit.
// Built with the engine's objects (as split_fnx_run): cmake -DSTRATA_BUILD_TESTS=ON, target split_cpu_exact_gpu_test.
#include "cpu_experts.hpp"
#include "kernels.hpp"
#include "model.hpp"
#include "p100_over.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FNX_ROW_G
#define FNX_ROW_G 2   // as p100_moe_rows.cuh: the card's lanes a gate|up row
#endif

using namespace fnx;
namespace cx = fnx::cpux;

namespace {
std::mt19937_64 rng(20261123);
int fails = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
bool h_nan(uint16_t h) { return (h & 0x7c00) == 0x7c00 && (h & 1023); }
bool same16(uint16_t a, uint16_t b) { return a == b || (h_nan(a) && h_nan(b)); }
bool same32(float a, float b) { return fbits(a) == fbits(b) || (std::isnan(a) && std::isnan(b)); }
uint16_t rnd_half(int lo, int hi, bool sign = true) {
    uint16_t h;
    if (rng() % 8 == 0) {
        do h = (uint16_t) rng(); while ((h & 0x7c00) == 0x7c00);
    } else h = (uint16_t) (((lo + rng() % (hi - lo + 1)) << 10) | (rng() & 1023));
    if (sign && (rng() & 1)) h |= 0x8000;
    else if (!sign) h &= 0x7fff;
    return h;
}

__global__ void k_ops(const uint32_t* a, const uint32_t* b, const uint32_t* c, const float* f, uint32_t* fma, uint32_t* mul, uint16_t* cvt, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const half2 x = *reinterpret_cast<const half2*>(a + i), y = *reinterpret_cast<const half2*>(b + i), z = *reinterpret_cast<const half2*>(c + i);
    const half2 r = __hfma2(x, y, z), m = __hmul2(x, y);
    fma[i] = *reinterpret_cast<const uint32_t*>(&r);
    mul[i] = *reinterpret_cast<const uint32_t*>(&m);
    const __half h = __float2half_rn(f[i]);
    cvt[i] = *reinterpret_cast<const uint16_t*>(&h);
}

void test_ops(int n) {
    std::vector<uint32_t> a(n), b(n), c(n), fma(n), mul(n);
    std::vector<float> f(n);
    std::vector<uint16_t> cvt(n);
    for (int i = 0; i < n; ++i) {
        const int mode = i % 4;
        auto pick = [&](int lo, int hi) { return (uint32_t) rnd_half(lo, hi) | ((uint32_t) rnd_half(lo, hi) << 16); };
        a[i] = i < 32768 ? (uint32_t) (2 * i) | ((uint32_t) (2 * i + 1) << 16) : mode == 3 ? pick(1, 30) : pick(5, 13);   // every half once
        b[i] = mode == 3 ? pick(1, 30) : pick(8, 18);
        c[i] = mode == 3 ? pick(1, 30) : pick(1, 16);
        uint32_t u = (uint32_t) rng();
        if (i & 1) u = (u & 0x80000000u) | ((uint32_t) (100 + rng() % 46) << 23) | (u & 0x7fffffu);
        std::memcpy(&f[i], &u, 4);
    }
    uint32_t *da, *db, *dc, *dfma, *dmul;
    float* df;
    uint16_t* dcvt;
    const size_t B = (size_t) n * 4;
    ck(cudaMalloc(&da, B), "malloc"), ck(cudaMalloc(&db, B), "malloc"), ck(cudaMalloc(&dc, B), "malloc"), ck(cudaMalloc(&dfma, B), "malloc");
    ck(cudaMalloc(&dmul, B), "malloc"), ck(cudaMalloc(&df, B), "malloc"), ck(cudaMalloc(&dcvt, (size_t) n * 2), "malloc");
    ck(cudaMemcpy(da, a.data(), B, cudaMemcpyHostToDevice), "copy"), ck(cudaMemcpy(db, b.data(), B, cudaMemcpyHostToDevice), "copy");
    ck(cudaMemcpy(dc, c.data(), B, cudaMemcpyHostToDevice), "copy"), ck(cudaMemcpy(df, f.data(), B, cudaMemcpyHostToDevice), "copy");
    k_ops<<<(n + 255) / 256, 256>>>(da, db, dc, df, dfma, dmul, dcvt, n);
    ck(cudaDeviceSynchronize(), "k_ops");
    ck(cudaMemcpy(fma.data(), dfma, B, cudaMemcpyDeviceToHost), "copy"), ck(cudaMemcpy(mul.data(), dmul, B, cudaMemcpyDeviceToHost), "copy");
    ck(cudaMemcpy(cvt.data(), dcvt, (size_t) n * 2, cudaMemcpyDeviceToHost), "copy");
    long bf = 0, bm = 0, bc = 0;
    for (int i = 0; i < n; ++i) {
        for (int l = 0; l < 2; ++l) {
            const uint16_t x = (uint16_t) (a[i] >> (16 * l)), y = (uint16_t) (b[i] >> (16 * l)), z = (uint16_t) (c[i] >> (16 * l));
            if (!same16((uint16_t) (fma[i] >> (16 * l)), cx::f16_fma_ref(x, y, z))) {
                if (++bf <= 5) std::printf("  hfma2(%04x, %04x, %04x) = %04x, oracle %04x\n", x, y, z, (unsigned) (uint16_t) (fma[i] >> (16 * l)), cx::f16_fma_ref(x, y, z));
            }
            if (!same16((uint16_t) (mul[i] >> (16 * l)), cx::f16_fma_ref(x, y, 0x8000u))) ++bm;
        }
        if (!same16(cvt[i], cx::f16_from_f32_ref(f[i]))) ++bc;
    }
    if (bf || bm || bc) ++fails;
    std::printf("%s card ops: %d x 2 HFMA2 lanes (every half as a first operand), as many HMUL2, %d float -> half: %ld / %ld / %ld differ from the oracle\n",
                bf || bm || bc ? "FAIL" : "ok  ", n, n, bf, bm, bc);
    cudaFree(da), cudaFree(db), cudaFree(dc), cudaFree(dfma), cudaFree(dmul), cudaFree(df), cudaFree(dcvt);
}

void fill_row(uint32_t type, int n, uint8_t* p, int lo, int hi) {
    const size_t rb = cx::row_bytes(type, n);
    for (size_t i = 0; i < rb; ++i) p[i] = (uint8_t) rng();
    auto put = [&](size_t off, uint16_t v) { std::memcpy(p + off, &v, 2); };
    switch (type) {
        case cx::T_IQ4_NL: for (int b = 0; b < n / 32; ++b) put((size_t) b * 18, rnd_half(lo, hi, false)); break;
        case cx::T_Q5_1: for (int b = 0; b < n / 32; ++b) put((size_t) b * 24, rnd_half(lo, hi, false)), put((size_t) b * 24 + 2, (uint16_t) (rnd_half(lo, hi, false) | 0x8000)); break;
        case cx::T_Q8_0: for (int b = 0; b < n / 32; ++b) put((size_t) b * 34, rnd_half(lo, hi, false)); break;
        case cx::T_Q4_K: for (int b = 0; b < n / 256; ++b) put((size_t) b * 144, rnd_half(lo, hi, false)), put((size_t) b * 144 + 2, rnd_half(lo, hi, false)); break;
        case cx::T_Q5_K: for (int b = 0; b < n / 256; ++b) put((size_t) b * 176, rnd_half(lo, hi, false)), put((size_t) b * 176 + 2, rnd_half(lo, hi, false)); break;
        default: break;
    }
}
template <class T>
T* dev(const std::vector<T>& v) {
    T* p = nullptr;
    ck(cudaMalloc(&p, v.size() * sizeof(T)), "malloc");
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "copy");
    return p;
}
template <class T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "copy");
    return v;
}

void test_rows(cudaStream_t st) {
    const int ffs = FF / FNX_NCARD;   // the card's share of an expert's FF
    long n_gu = 0, b_gu = 0, n_h = 0, b_h = 0, n_dn = 0, b_dn = 0;
    cx::Pool pool(3, {});
    for (uint32_t gt : {cx::T_Q4_K, cx::T_Q5_K})
        for (uint32_t dt : {cx::T_IQ4_NL, cx::T_Q5_1, cx::T_Q8_0})
            for (int nt : {1, 2, 3, 5, 9, 16})
                for (int range = 0; range < 2; ++range) {
                    const int slo = range ? 1 : 3, shi = range ? 20 : 8;
                    const size_t gb = cx::row_bytes(gt, N), db = cx::row_bytes(dt, ffs);
                    std::vector<uint8_t> gu((size_t) 2 * ffs * gb), dn((size_t) N * db);
                    for (int j = 0; j < 2 * ffs; ++j) fill_row(gt, N, gu.data() + (size_t) j * gb, slo, shi);
                    for (int i = 0; i < N; ++i) fill_row(dt, ffs, dn.data() + (size_t) i * db, slo, shi);
                    std::vector<uint16_t> x16((size_t) nt * N);
                    for (auto& v : x16) v = range ? rnd_half(1, 22) : rnd_half(9, 17);
                    // the card: gate|up rows, the three SwiGLUs, down rows of its own h
                    Job job{};
                    job.w = 0, job.n = nt;
                    for (int k = 0; k < nt; ++k) job.y[k] = k, job.out[k] = k;
                    const std::vector<Job> jobs{job};
                    const std::vector<int> nj{1, 1}, list{1, 0};
                    Job* djobs = dev(jobs);
                    int* dnj = dev(nj);
                    int* dlist = dev(list);
                    uint8_t* dgu = dev(gu);
                    uint8_t* ddn = dev(dn);
                    uint16_t* dx = dev(x16);
                    float *dg, *dout;
                    uint16_t *dh1, *dh2, *dh3;
                    ck(cudaMalloc(&dg, (size_t) nt * 2 * ffs * 4), "malloc"), ck(cudaMalloc(&dout, (size_t) nt * N * 4), "malloc");
                    ck(cudaMalloc(&dh1, (size_t) nt * ffs * 2), "malloc"), ck(cudaMalloc(&dh2, (size_t) nt * ffs * 2), "malloc"), ck(cudaMalloc(&dh3, (size_t) nt * ffs * 2), "malloc");
                    DW A;
                    A.p = dgu, A.type = gt, A.n_in = N, A.n_out = 2 * ffs, A.row_bytes = gb;
                    p100_moe_rows_phase(A, gu.size(), djobs, dnj, 1, nt, dx, dg, 0, st);
                    p100_swiglu_after(dg, ffs, nt, dh1, dlist, st);
                    swiglu_h16(dg, ffs, nt, dh2, st);
                    ck(cudaStreamSynchronize(st), "rows gate|up");
                    p100_cpu_swiglu_probe(dg, nt, ffs, dh3, st);
                    DW D;
                    D.p = ddn, D.type = dt, D.n_in = ffs, D.n_out = N, D.row_bytes = db;
                    p100_moe_rows_phase(D, dn.size(), djobs, dnj, 1, nt, dh1, dout, 0, st);
                    ck(cudaStreamSynchronize(st), "rows down");
                    const std::vector<float> g_card = host(dg, (size_t) nt * 2 * ffs), o_card = host(dout, (size_t) nt * N);
                    const std::vector<uint16_t> h1 = host(dh1, (size_t) nt * ffs), h2 = host(dh2, (size_t) nt * ffs), h3 = host(dh3, (size_t) nt * ffs);
                    // the CPU: gate|up of the same x, down of the card's h
                    std::vector<float> x((size_t) nt * N), h((size_t) nt * ffs), g_cpu((size_t) nt * 2 * ffs), o_cpu((size_t) nt * N);
                    cx::f16_to_f32(x16.data(), x.data(), x.size());
                    cx::f16_to_f32(h1.data(), h.data(), h.size());
                    cx::Work w;
                    w.gu_type = gt, w.dn_type = dt, w.ffs = ffs, w.N = N, w.x = x.data(), w.x_stride = N, w.out_stride = N, w.exact = true, w.gu_lanes = FNX_ROW_G;
                    cx::Expert e;
                    e.gu = gu.data(), e.dn = dn.data(), e.n = nt, e.h = h.data(), e.out = o_cpu.data(), e.gu_out = g_cpu.data();
                    for (int k = 0; k < nt; ++k) e.rows[k] = k;
                    w.ex = &e, w.m = 1;
                    pool.run_gu(w);
                    pool.run_down(w);
                    for (size_t i = 0; i < g_cpu.size(); ++i, ++n_gu)
                        if (!same32(g_cpu[i], g_card[i]) && ++b_gu <= 5)
                            std::printf("  gate|up %s, %d tokens, value %zu: CPU %.9g, card %.9g\n", cx::type_name(gt), nt, i, g_cpu[i], g_card[i]);
                    for (size_t i = 0; i < h1.size(); ++i, ++n_h)
                        if ((!same16(h1[i], h2[i]) || !same16(h1[i], h3[i])) && ++b_h <= 5)
                            std::printf("  SwiGLU value %zu: k_swiglu_after %04x, swiglu_h16 %04x, k_cpu_swiglu %04x\n", i, h1[i], h2[i], h3[i]);
                    for (size_t i = 0; i < o_cpu.size(); ++i, ++n_dn)
                        if (!same32(o_cpu[i], o_card[i]) && ++b_dn <= 5)
                            std::printf("  down %s, %d tokens, value %zu: CPU %.9g, card %.9g\n", cx::type_name(dt), nt, i, o_cpu[i], o_card[i]);
                    cudaFree(djobs), cudaFree(dnj), cudaFree(dlist), cudaFree(dgu), cudaFree(ddn), cudaFree(dx), cudaFree(dg), cudaFree(dout);
                    cudaFree(dh1), cudaFree(dh2), cudaFree(dh3);
                }
    if (b_gu || b_h || b_dn) ++fails;
    std::printf("%s row kernels: gate|up %ld values, %ld not the card's; SwiGLU %ld values, %ld where the three kernels differ; down %ld values, %ld not the card's\n",
                b_gu || b_h || b_dn ? "FAIL" : "ok  ", n_gu, b_gu, n_h, b_h, n_dn, b_dn);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const int devn = argc > 1 ? std::atoi(argv[1]) : 0;
        ck(cudaSetDevice(devn), "device");
        cudaDeviceProp p;
        ck(cudaGetDeviceProperties(&p, devn), "properties");
        std::printf("card %d: %s, sm_%d%d; CPU AVX2: %s\n", devn, p.name, p.major, p.minor, cx::have_simd() ? "yes" : "no");
        cudaStream_t st;
        ck(cudaStreamCreate(&st), "stream");
        test_ops(1 << 23);
        test_rows(st);
    } catch (const std::exception& e) {
        std::printf("FAIL %s\n", e.what());
        return 2;
    }
    std::printf(fails ? "cpu_exact_gpu_test: FAILED\n" : "cpu_exact_gpu_test: all bit for bit\n");
    return fails ? 1 : 0;
}
