// cpu_exact_test.cpp: FNX_CPU_EXACT's arithmetic (src/cpu_experts.cpp), no GPU.
//   cpu_exact_test [N]        1. the oracle (binary16 on the integers) on known answers, its rounding against F16C on N million floats;
//                             2. the AVX2 fp16 FMA against the oracle on N million random triples, and on every first operand;
//                             3. the exact kernels (every gate|up and down type, 1-4 lanes, 1-3 tokens) against the card's steps done
//                                with the oracle, bit for bit, directly and through the pool
//   cpu_exact_test bench NTOK THREADS REPS
//                             one expert half (640 x 2560 gate|up, 2560 x 320 down), Q4_K / IQ4_NL: the fp32 path and the exact one
// The card's side of the claim (its HFMA2 and row kernels against these steps) needs a GPU: tools/cpu_exact_gpu_test.cu.
#include "cpu_experts.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <immintrin.h>

using namespace fnx::cpux;

namespace {
std::mt19937_64 rng(20261123);
int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float bitsf(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
bool h_nan(uint16_t h) { return (h & 0x7c00) == 0x7c00 && (h & 1023); }
bool same16(uint16_t a, uint16_t b) { return a == b || (h_nan(a) && h_nan(b)); }
bool same32(float a, float b) { return fbits(a) == fbits(b) || (std::isnan(a) && std::isnan(b)); }
__attribute__((target("f16c"))) uint16_t hw16(float f) { return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT); }

// a finite half: mostly in `lo..hi` binades (exponent fields), sometimes anything finite (subnormals, zeros, both signs)
uint16_t rnd_half(int lo, int hi, bool sign = true) {
    uint16_t h;
    if (rng() % 8 == 0) {
        do h = (uint16_t) rng(); while ((h & 0x7c00) == 0x7c00);
    } else h = (uint16_t) (((lo + rng() % (hi - lo + 1)) << 10) | (rng() & 1023));
    if (sign && (rng() & 1)) h |= 0x8000;
    else if (!sign) h &= 0x7fff;
    return h;
}

void test_oracle(long n) {
    // known answers
    const uint16_t one = 0x3c00, two = 0x4000;
    CHECK(f16_fma_ref(one, one, one) == two, "1*1+1");
    CHECK(f16_fma_ref(0x3e00, 0x3956, 0x0001) == 0x3c01, "1.5 * 0.6669921875 + 2^-24 = 1.0009765625 (0x3c01), not 1.0: got %04x",
          f16_fma_ref(0x3e00, 0x3956, 0x0001));
    CHECK(hw16(f16_to_f32(0x3e00) * f16_to_f32(0x3956) + f16_to_f32(0x0001)) == 0x3c00,
          "the same case through fp32 and one rounding to fp16 gives 1.0 (double rounding): the case the exact path is for");
    CHECK(f16_fma_ref(0x7bff, two, 0) == 0x7c00, "65504*2 -> inf");
    CHECK(f16_fma_ref(0x0001, one, 0x0001) == 0x0002, "subnormal sum");
    CHECK(f16_fma_ref(0x0000, one, 0x8000) == 0x0000, "+0 + -0 = +0");
    CHECK(f16_fma_ref(0x8000, one, 0x8000) == 0x8000, "-0 + -0 = -0");
    CHECK(f16_fma_ref(one, one, 0xbc00) == 0x0000, "1 - 1 = +0");
    CHECK(f16_fma_ref(0x3555, 0x3555, 0) == hw16(f16_to_f32(0x3555) * f16_to_f32(0x3555)), "a product rounded once");
    CHECK(h_nan(f16_fma_ref(0x7c00, 0, one)), "inf * 0");
    CHECK(h_nan(f16_fma_ref(0x7c00, one, 0xfc00)), "inf - inf");
    CHECK(f16_fma_ref(0x7c00, one, 0x7c00) == 0x7c00, "inf + inf");
    // the oracle's rounding (the fma's too) against the hardware's float -> half on random floats and on every half's neighbourhood
    long bad = 0;
    for (long i = 0; i < n; ++i) {
        uint32_t u = (uint32_t) rng();
        if (i & 1) u = (u & 0x80000000u) | ((uint32_t) (100 + rng() % 46) << 23) | (u & 0x7fffffu);   // the binades fp16 covers
        const float f = bitsf(u);
        if (!same16(f16_from_f32_ref(f), hw16(f))) ++bad;
    }
    for (uint32_t h = 0; h < 0x7c00; ++h) {
        const uint32_t c = fbits(f16_to_f32((uint16_t) h));
        for (int d = -4; d <= 4; ++d) {   // the half itself, the floats around it, and around the midpoints above it
            const float f1 = bitsf(c + d), f2 = bitsf(c + (1u << 12) + d);
            if (!same16(f16_from_f32_ref(f1), hw16(f1)) || !same16(f16_from_f32_ref(f2), hw16(f2))) ++bad;
        }
    }
    CHECK(bad == 0, "oracle rounding differs from F16C in %ld cases", bad);
    if (!bad) std::printf("oracle: known answers, and its rounding against F16C on %ld random floats and around every half\n", n);
}

void test_fma16(long n) {
    if (!have_simd()) {
        std::printf("fma16: no AVX2 here, skipped\n");
        return;
    }
    long bad = 0, naive = 0, cases = 0;
    alignas(32) float w[8], x[8], c[8], out[8];
    uint16_t hw[8], hx[8], hc[8];
    auto run = [&] {
        fma16_x8(w, x, c, out);
        for (int i = 0; i < 8; ++i) {
            const uint16_t ref = f16_fma_ref(hw[i], hx[i], hc[i]);
            if (!same16(hw16(out[i]), ref) || !same32(out[i], f16_to_f32(ref))) ++bad;
            if (!same16(hw16(w[i] * x[i] + c[i]), ref)) ++naive;   // fp32 then fp16: wrong where it double-rounds
            ++cases;
        }
    };
    for (long i = 0; i < n / 8; ++i) {
        const int mode = (int) (i % 3);   // weights x activations as the kernels see them; and any finite values
        for (int k = 0; k < 8; ++k) {
            hw[k] = mode == 2 ? rnd_half(1, 30) : rnd_half(5, 13);
            hx[k] = mode == 2 ? rnd_half(1, 30) : rnd_half(8, 18);
            hc[k] = mode == 0 ? rnd_half(1, 16) : rnd_half(1, 30);
            if (mode == 1 && (rng() & 1)) {   // aim the sum at a midpoint: c = a fp16 value near -w*x, so lots of exact cancellation too
                const float p = f16_to_f32(hw[k]) * f16_to_f32(hx[k]);
                hc[k] = hw16(-p);
                hc[k] = (uint16_t) (hc[k] + (int) (rng() % 5) - 2);
                if ((hc[k] & 0x7c00) == 0x7c00) hc[k] = 0;
            }
            w[k] = f16_to_f32(hw[k]), x[k] = f16_to_f32(hx[k]), c[k] = f16_to_f32(hc[k]);
        }
        run();
    }
    for (uint32_t a = 0; a < 65536; ++a) {   // every first operand (infinities and NaN included)
        for (int k = 0; k < 8; ++k) {
            hw[k] = (uint16_t) a;
            hx[k] = rnd_half(1, 30);
            hc[k] = rnd_half(1, 30);
            w[k] = f16_to_f32(hw[k]), x[k] = f16_to_f32(hx[k]), c[k] = f16_to_f32(hc[k]);
        }
        run();
    }
    CHECK(bad == 0, "fma16 differs from the oracle in %ld of %ld cases", bad, cases);
    if (!bad) std::printf("fma16: %ld cases bit for bit the oracle's; fp32-then-fp16 would have differed in %ld of them\n", cases, naive);
}

// random blocks: every byte random, the fp16 fields in the ranges given (exponent fields)
void fill_row(uint32_t type, int n, uint8_t* p, int lo, int hi) {
    const size_t rb = row_bytes(type, n);
    for (size_t i = 0; i < rb; ++i) p[i] = (uint8_t) rng();
    auto put = [&](size_t off, uint16_t v) { std::memcpy(p + off, &v, 2); };
    switch (type) {
        case T_IQ4_NL: for (int b = 0; b < n / 32; ++b) put((size_t) b * 18, rnd_half(lo, hi, false)); break;
        case T_Q5_1: for (int b = 0; b < n / 32; ++b) put((size_t) b * 24, rnd_half(lo, hi, false)), put((size_t) b * 24 + 2, (uint16_t) (rnd_half(lo, hi, false) | 0x8000)); break;
        case T_Q8_0: for (int b = 0; b < n / 32; ++b) put((size_t) b * 34, rnd_half(lo, hi, false)); break;
        case T_Q4_K: for (int b = 0; b < n / 256; ++b) put((size_t) b * 144, rnd_half(lo, hi, false)), put((size_t) b * 144 + 2, rnd_half(lo, hi, false)); break;
        case T_Q5_K: for (int b = 0; b < n / 256; ++b) put((size_t) b * 176, rnd_half(lo, hi, false)), put((size_t) b * 176 + 2, rnd_half(lo, hi, false)); break;
        default: break;
    }
}

void test_kernels(bool through_pool) {
    const int N = 2560, ffs = 320, R = 48;   // R rows of each kind (not a multiple of 16: the kernels' tails)
    long rows = 0, bad = 0;
    Pool pool(3, {});
    for (uint32_t gt : {T_Q4_K, T_Q5_K})
        for (uint32_t dt : {T_IQ4_NL, T_Q5_1, T_Q8_0})
            for (int lanes : {1, 2, 4})
                for (int nt : {1, 3})
                    for (int range = 0; range < 2; ++range) {   // 0: scales and values as in a model; 1: wide (ties, subnormals, overflow)
                        const int slo = range ? 1 : 3, shi = range ? 20 : 8;
                        std::vector<uint8_t> gu((size_t) 2 * ffs * row_bytes(gt, N)), dn((size_t) N * row_bytes(dt, ffs));
                        for (int j = 0; j < 2 * ffs; ++j) fill_row(gt, N, gu.data() + (size_t) j * row_bytes(gt, N), slo, shi);
                        for (int i = 0; i < N; ++i) fill_row(dt, ffs, dn.data() + (size_t) i * row_bytes(dt, ffs), slo, shi);
                        std::vector<uint16_t> x16((size_t) nt * N), h16((size_t) nt * ffs);
                        for (auto& v : x16) v = range ? rnd_half(1, 22) : rnd_half(9, 17);
                        for (auto& v : h16) v = range ? rnd_half(1, 22) : rnd_half(6, 16);
                        std::vector<float> x(x16.size()), h(h16.size()), gout((size_t) nt * 2 * ffs, -1.f), out((size_t) nt * N, -1.f);
                        f16_to_f32(x16.data(), x.data(), x.size());
                        f16_to_f32(h16.data(), h.data(), h.size());
                        Work w;
                        w.gu_type = gt, w.dn_type = dt, w.ffs = ffs, w.N = N, w.x = x.data(), w.x_stride = N, w.out_stride = N;
                        w.exact = true, w.gu_lanes = lanes;
                        Expert e;
                        e.gu = gu.data(), e.dn = dn.data(), e.n = nt, e.h = h.data(), e.out = out.data(), e.gu_out = gout.data();
                        for (int k = 0; k < nt; ++k) e.rows[k] = k;
                        w.ex = &e, w.m = 1;
                        if (through_pool) {
                            pool.run_gu(w);
                            pool.run_down(w);
                        } else {
                            gu_chunk(w, e, 0, R);
                            gu_chunk(w, e, ffs - R, ffs);
                            down_chunk(w, e, 0, R);
                            down_chunk(w, e, N - R, N);
                        }
                        auto gcheck = [&](int part, int j) {
                            for (int k = 0; k < nt; ++k) {
                                const float ref = gu_row_card_ref(gt, gu.data() + (size_t) (part * ffs + j) * row_bytes(gt, N), N, x16.data() + (size_t) k * N, lanes);
                                const float got = gout[(size_t) k * 2 * ffs + (size_t) part * ffs + j];
                                ++rows;
                                if (!same32(got, ref)) {
                                    if (++bad <= 5) std::printf("  gate|up %s lanes %d part %d row %d token %d: %.9g, the card's steps %.9g\n", type_name(gt), lanes, part, j, k, got, ref);
                                }
                            }
                        };
                        auto dcheck = [&](int i) {
                            for (int k = 0; k < nt; ++k) {
                                const float ref = dn_row_card_ref(dt, dn.data() + (size_t) i * row_bytes(dt, ffs), ffs, h16.data() + (size_t) k * ffs);
                                const float got = out[(size_t) k * N + i];
                                ++rows;
                                if (!same32(got, ref)) {
                                    if (++bad <= 5) std::printf("  down %s row %d token %d: %.9g, the card's steps %.9g\n", type_name(dt), i, k, got, ref);
                                }
                            }
                        };
                        for (int part = 0; part < 2; ++part)
                            for (int j = 0; j < R; ++j) gcheck(part, j), gcheck(part, ffs - R + j);
                        for (int i = 0; i < R; ++i) dcheck(i), dcheck(N - R + i);
                    }
    CHECK(bad == 0, "exact kernels%s: %ld of %ld row sums differ from the card's steps", through_pool ? " (pool)" : "", bad, rows);
    if (!bad) std::printf("exact kernels%s: %ld row sums bit for bit the card's steps (Q4_K/Q5_K gate|up with 1, 2, 4 lanes; IQ4_NL/Q5_1/Q8_0 down; 1 and 3 tokens)\n",
                through_pool ? ", through the pool" : "", rows);
}

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int bench(int nt, int threads, int reps) {
    const int N = 2560, ffs = 320;
    const uint32_t gt = T_Q4_K, dt = T_IQ4_NL;
    std::vector<uint8_t> gu((size_t) 2 * ffs * row_bytes(gt, N)), dn((size_t) N * row_bytes(dt, ffs));
    for (int j = 0; j < 2 * ffs; ++j) fill_row(gt, N, gu.data() + (size_t) j * row_bytes(gt, N), 3, 8);
    for (int i = 0; i < N; ++i) fill_row(dt, ffs, dn.data() + (size_t) i * row_bytes(dt, ffs), 3, 8);
    std::vector<uint16_t> x16((size_t) nt * N);
    for (auto& v : x16) v = rnd_half(9, 17);
    std::vector<uint16_t> h16((size_t) nt * ffs);
    for (auto& v : h16) v = rnd_half(6, 16);
    std::vector<float> x(x16.size()), h(h16.size()), gout((size_t) nt * 2 * ffs), out((size_t) nt * N);
    f16_to_f32(x16.data(), x.data(), x.size());
    f16_to_f32(h16.data(), h.data(), h.size());
    Pool pool(threads - 1, {});
    for (int exact = 0; exact < 2; ++exact) {
        Work w;
        w.gu_type = gt, w.dn_type = dt, w.ffs = ffs, w.N = N, w.x = x.data(), w.x_stride = N, w.out_stride = N, w.exact = exact != 0;
        Expert e;
        e.gu = gu.data(), e.dn = dn.data(), e.n = nt, e.h = h.data(), e.out = out.data(), e.gu_out = gout.data();
        for (int k = 0; k < nt; ++k) e.rows[k] = k;
        w.ex = &e, w.m = 1;
        double best = 1e30, bg = 1e30;
        for (int r = 0; r < reps; ++r) {
            const double t0 = now_ms();
            pool.run_gu(w);
            const double t1 = now_ms();
            pool.run_down(w);
            const double t2 = now_ms();
            best = std::min(best, t2 - t0), bg = std::min(bg, t1 - t0);
        }
        std::printf("%-6s one expert half, %d token(s), %d thread(s): %.1f us (gate|up %.1f, down %.1f)\n", exact ? "exact" : "fp32", nt, threads, best * 1e3,
                    bg * 1e3, (best - bg) * 1e3);
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "bench")
        return bench(argc > 2 ? std::atoi(argv[2]) : 1, argc > 3 ? std::atoi(argv[3]) : 1, argc > 4 ? std::atoi(argv[4]) : 20);
    const long n = (argc > 1 ? std::atol(argv[1]) : 8) * 1000000L;
    std::printf("AVX2+FMA+F16C: %s\n", have_simd() ? "yes" : "no (the exact kernels run the oracle's steps)");
    test_oracle(n);
    test_fma16(n);
    test_kernels(false);
    test_kernels(true);
    std::printf(fails ? "cpu_exact_test: %d FAILED\n" : "cpu_exact_test: all passed\n", fails);
    return fails ? 1 : 0;
}
