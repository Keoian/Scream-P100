// cpu_experts_test.cpp: the CPU expert path's decoders, products and pool (src/cpu_experts.cpp), no GPU.
//   cpu_experts_test                      every type: the AVX2 decoder against ggml's scalar steps on random blocks, the chunked products
//                                         against a double-precision sum, the pool against one thread
//   cpu_experts_test check FILE           a real expert's half (tools/cpu_experts_real.py wrote it with gguf-py's own dequantizer): the
//                                         outputs for its tokens against the file's
//   cpu_experts_test bench M NTOK THREADS NODE REPS [GU_TYPE DN_TYPE]
//                                         M experts' halves of random blocks on NUMA node NODE, NTOK tokens each: ms a run, GB/s read
#include "cpu_experts.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

using namespace fnx::cpux;

namespace {
std::mt19937_64 rng(20261122);
uint16_t sane_half() {   // a positive half between 2^-12 and 2^-7: a block's scale
    return (uint16_t) (((3 + rng() % 6) << 10) | (rng() & 1023));
}
// random blocks of a row of n values: every byte random, the f16 fields sane
void fill_row(uint32_t type, int n, uint8_t* p) {
    const size_t rb = row_bytes(type, n);
    for (size_t i = 0; i < rb; ++i) p[i] = (uint8_t) rng();
    auto put = [&](size_t off, uint16_t v) { std::memcpy(p + off, &v, 2); };
    switch (type) {
        case T_Q4_0: case T_IQ4_NL: for (int b = 0; b < n / 32; ++b) put((size_t) b * 18, sane_half()); break;
        case T_Q5_0: for (int b = 0; b < n / 32; ++b) put((size_t) b * 22, sane_half()); break;
        case T_Q5_1: for (int b = 0; b < n / 32; ++b) put((size_t) b * 24, sane_half()), put((size_t) b * 24 + 2, (uint16_t) (sane_half() | 0x8000)); break;
        case T_Q8_0: for (int b = 0; b < n / 32; ++b) put((size_t) b * 34, sane_half()); break;
        case T_Q4_K: for (int b = 0; b < n / 256; ++b) put((size_t) b * 144, sane_half()), put((size_t) b * 144 + 2, sane_half()); break;
        case T_Q5_K: for (int b = 0; b < n / 256; ++b) put((size_t) b * 176, sane_half()), put((size_t) b * 176 + 2, sane_half()); break;
        case T_Q6_K: for (int b = 0; b < n / 256; ++b) put((size_t) b * 210 + 208, sane_half()); break;
        default: break;
    }
}
double silu_d(double g) { return g / (1.0 + std::exp(-g)); }

// the expert's outputs in double precision from ggml's scalar decode
void naive(const Work& w, const Expert& e, std::vector<double>& out) {
    const size_t gb = row_bytes(w.gu_type, w.N), db = row_bytes(w.dn_type, w.ffs);
    std::vector<float> g(w.N), u(w.N), d(w.ffs);
    std::vector<double> h((size_t) e.n * w.ffs);
    for (int j = 0; j < w.ffs; ++j) {
        decode_row_ref(w.gu_type, e.gu + (size_t) j * gb, w.N, g.data());
        decode_row_ref(w.gu_type, e.gu + (size_t) (w.ffs + j) * gb, w.N, u.data());
        for (int k = 0; k < e.n; ++k) {
            const float* x = w.x + (size_t) e.rows[k] * w.x_stride;
            double sg = 0, su = 0;
            for (int i = 0; i < w.N; ++i) sg += (double) g[i] * x[i], su += (double) u[i] * x[i];
            h[(size_t) k * w.ffs + j] = silu_d(sg) * su;
        }
    }
    out.assign((size_t) e.n * w.N, 0.0);
    for (int i = 0; i < w.N; ++i) {
        decode_row_ref(w.dn_type, e.dn + (size_t) i * db, w.ffs, d.data());
        for (int k = 0; k < e.n; ++k) {
            double s = 0;
            for (int j = 0; j < w.ffs; ++j) s += (double) d[j] * h[(size_t) k * w.ffs + j];
            out[(size_t) k * w.N + i] = s;
        }
    }
}
// largest |a - b| over the largest |b| of the rows
double rel_err(const float* a, const double* b, size_t n) {
    double e = 0, m = 0;
    for (size_t i = 0; i < n; ++i) e = std::max(e, std::fabs((double) a[i] - b[i])), m = std::max(m, std::fabs(b[i]));
    return m > 0 ? e / m : e;
}

int selftest() {
    int bad = 0;
    std::printf("AVX2 + FMA + F16C: %s\n", have_simd() ? "yes" : "no (the scalar steps only)");
    // 1. decoders: AVX2 against ggml's scalar steps
    const uint32_t types[] = {T_Q4_0, T_Q5_0, T_Q5_1, T_Q8_0, T_IQ4_NL, T_Q4_K, T_Q5_K, T_Q6_K};
    for (uint32_t t : types) {
        const int n = 2560;
        std::vector<uint8_t> row(row_bytes(t, n));
        std::vector<float> a(n), b(n);
        double worst = 0;
        long diff = 0;
        for (int rep = 0; rep < 2000; ++rep) {
            fill_row(t, n, row.data());
            decode_row(t, row.data(), n, a.data());
            decode_row_ref(t, row.data(), n, b.data());
            for (int i = 0; i < n; ++i) {
                if (a[i] != b[i]) ++diff;
                const double e = std::fabs((double) a[i] - b[i]) / std::max(1e-30, std::fabs((double) b[i]));
                if (std::fabs(b[i]) > 1e-6) worst = std::max(worst, e);
                if (std::fabs((double) a[i] - b[i]) > 1e-6 * std::max(1.0, std::fabs((double) b[i]) * 1e3)) ++bad;
            }
        }
        std::printf("decode %-6s: 2000 rows of %d, values not bit-equal to the scalar steps %ld (an FMA rounds once where they round twice), worst relative difference %.2e\n",
                    type_name(t), n, diff, worst);
    }
    // 2. an expert's half, chunked and pooled, against the double-precision sum; the window's rows come as halves, as the engine's do
    struct Pair { uint32_t gu, dn; };
    const Pair pairs[] = {{T_Q4_K, T_IQ4_NL}, {T_Q5_K, T_IQ4_NL}, {T_Q4_K, T_Q5_1}, {T_Q5_K, T_Q8_0}, {T_Q6_K, T_Q8_0}, {T_Q4_K, T_Q4_0}, {T_Q5_K, T_Q5_0}};
    Pool pool(7, {});
    for (const Pair& pr : pairs) {
        const int N = 2560, ffs = 320, T = 16, m = 5;
        Work w;
        w.gu_type = pr.gu, w.dn_type = pr.dn, w.N = N, w.ffs = ffs, w.out_stride = N;
        std::vector<uint16_t> x16((size_t) T * N);
        for (uint16_t& v : x16) v = (uint16_t) ((rng() & 0x8000) | ((11 + rng() % 7) << 10) | (rng() & 1023));   // +-2^-4 .. 2^3
        std::vector<float> xf((size_t) T * N);
        Pool::Rows rows;
        rows.x16 = x16.data(), rows.T = T, rows.xf = xf.data();
        const size_t gb = row_bytes(pr.gu, N) * 2 * ffs, db = row_bytes(pr.dn, ffs) * N;
        std::vector<std::vector<uint8_t>> gu(m, std::vector<uint8_t>(gb)), dn(m, std::vector<uint8_t>(db));
        std::vector<Expert> ex(m);
        std::vector<std::vector<float>> h(m), out(m), out1(m);
        for (int e = 0; e < m; ++e) {
            for (int r = 0; r < 2 * ffs; ++r) fill_row(pr.gu, N, gu[e].data() + (size_t) r * row_bytes(pr.gu, N));
            for (int r = 0; r < N; ++r) fill_row(pr.dn, ffs, dn[e].data() + (size_t) r * row_bytes(pr.dn, ffs));
            ex[e].gu = gu[e].data(), ex[e].dn = dn[e].data();
            ex[e].n = 1 + (int) (rng() % (e == 0 ? 16 : 3));
            for (int k = 0; k < ex[e].n; ++k) ex[e].rows[k] = (int) (rng() % T);
            h[e].resize((size_t) ex[e].n * ffs), out[e].assign((size_t) ex[e].n * N, 0.f), out1[e].assign((size_t) ex[e].n * N, 0.f);
            ex[e].h = h[e].data(), ex[e].out = out[e].data();
        }
        w.ex = ex.data(), w.m = m;
        pool.run(w, &rows);
        double worst = 0;
        bool same = true;
        for (int e = 0; e < m; ++e) {
            std::vector<double> ref;
            naive(w, ex[e], ref);
            worst = std::max(worst, rel_err(out[e].data(), ref.data(), ref.size()));
            Expert one = ex[e];   // the same expert in one piece on this thread: the pool's chunks must give the same bits
            std::vector<float> h1((size_t) one.n * ffs);
            one.h = h1.data(), one.out = out1[e].data();
            gu_chunk(w, one, 0, ffs);
            down_chunk(w, one, 0, N);
            same = same && std::memcmp(out1[e].data(), out[e].data(), out[e].size() * 4) == 0;
        }
        const bool ok = worst < 5e-6 && same;
        if (!ok) ++bad;
        std::printf("expert %-5s gate|up, %-6s down: %d experts, largest error over the largest output %.2e against the double sum; pool == one thread: %s%s\n",
                    type_name(pr.gu), type_name(pr.dn), m, worst, same ? "yes" : "NO", ok ? "" : "  <-- FAIL");
    }
    std::printf(bad ? "FAIL (%d)\n" : "ALL OK\n", bad);
    return bad ? 1 : 0;
}

int check(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::perror(path); return 2; }
    int hd[5];   // gu_type, dn_type, N, ffs, n
    if (std::fread(hd, 4, 5, f) != 5) return 2;
    Work w;
    w.gu_type = (uint32_t) hd[0], w.dn_type = (uint32_t) hd[1], w.N = hd[2], w.ffs = hd[3];
    const int n = hd[4];
    const size_t gb = row_bytes(w.gu_type, w.N) * 2 * w.ffs, db = row_bytes(w.dn_type, w.ffs) * w.N;
    std::vector<uint8_t> gu(gb), dn(db);
    std::vector<float> x((size_t) n * w.N), out((size_t) n * w.N), h((size_t) n * w.ffs);
    std::vector<double> want((size_t) n * w.N);
    if (!gb || !db || std::fread(gu.data(), 1, gb, f) != gb || std::fread(dn.data(), 1, db, f) != db ||
        std::fread(x.data(), 4, x.size(), f) != x.size() || std::fread(want.data(), 8, want.size(), f) != want.size()) {
        std::fprintf(stderr, "%s: short or not a type the CPU path decodes\n", path);
        return 2;
    }
    std::fclose(f);
    w.x = x.data(), w.x_stride = w.N, w.out_stride = w.N;
    Expert e;
    e.gu = gu.data(), e.dn = dn.data(), e.n = n, e.h = h.data(), e.out = out.data();
    for (int k = 0; k < n; ++k) e.rows[k] = k;
    w.ex = &e, w.m = 1;
    Pool pool(3, {});
    pool.run(w);
    const double err = rel_err(out.data(), want.data(), want.size());
    std::printf("%s: %s gate|up, %s down, N %d, ffs %d, %d tokens: largest error over the largest output %.2e against gguf-py's dequantizer in double precision: %s\n",
                path, type_name(w.gu_type), type_name(w.dn_type), w.N, w.ffs, n, err, err < 5e-6 ? "OK" : "FAIL");
    return err < 5e-6 ? 0 : 1;
}

int bench(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "bench M NTOK THREADS NODE REPS [GU_TYPE DN_TYPE [EXPERTS_IN_MEMORY]]\n"); return 2; }
    const int m = std::atoi(argv[2]), ntok = std::atoi(argv[3]), threads = std::atoi(argv[4]), node = std::atoi(argv[5]), reps = std::atoi(argv[6]);
    Work w;
    w.gu_type = argc > 7 ? (uint32_t) std::atoi(argv[7]) : T_Q4_K, w.dn_type = argc > 8 ? (uint32_t) std::atoi(argv[8]) : T_IQ4_NL;
    w.N = 2560, w.ffs = 320;
    const int T = 16, POOL = argc > 9 ? std::atoi(argv[9]) : 512;   // experts in memory: far beyond L3 by default, each run takes the next m of them
    const size_t gb = row_bytes(w.gu_type, w.N) * 2 * w.ffs, db = row_bytes(w.dn_type, w.ffs) * w.N, per = gb + db;
    if (node >= 0) {   // this thread's pages on the node (MPOL_BIND), as FNX_NUMA_EXPERTS places a card's host copies
        const unsigned long mask = 1ul << node;
        if (syscall(SYS_set_mempolicy, 2, &mask, 8 * sizeof mask) != 0) std::perror("set_mempolicy");
    }
    uint8_t* mem = (uint8_t*) mmap(nullptr, per * POOL, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) { std::perror("mmap"); return 2; }
    {   // one random expert, copied POOL times with its scales kept sane
        std::vector<uint8_t> one(per);
        for (int r = 0; r < 2 * w.ffs; ++r) fill_row(w.gu_type, w.N, one.data() + (size_t) r * row_bytes(w.gu_type, w.N));
        for (int r = 0; r < w.N; ++r) fill_row(w.dn_type, w.ffs, one.data() + gb + (size_t) r * row_bytes(w.dn_type, w.ffs));
        for (int e = 0; e < POOL; ++e) std::memcpy(mem + (size_t) e * per, one.data(), per);
    }
    std::vector<float> x((size_t) T * w.N);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (float& v : x) v = nd(rng);
    w.x = x.data(), w.x_stride = w.N, w.out_stride = w.N;
    std::vector<uint16_t> x16((size_t) T * w.N);
    for (uint16_t& v : x16) v = (uint16_t) ((rng() & 0x8000) | ((11 + rng() % 7) << 10) | (rng() & 1023));
    Pool::Rows rows;   // the window's rows as the engine's host thread gets them: T halves' rows, converted by the pool each run
    rows.x16 = x16.data(), rows.T = std::min(T, std::max(1, m * ntok)), rows.xf = x.data();
    std::vector<Expert> ex(m);
    std::vector<float> h((size_t) m * ntok * w.ffs), out((size_t) m * ntok * w.N);
    const std::vector<int> cpus = node >= 0 ? node_cores(node) : std::vector<int>{};
    std::vector<int> wc(cpus.begin() + (cpus.size() > 2 ? 2 : 0), cpus.end());   // the node's first two cores left to the caller and the card's threads
    Pool pool(threads - 1, wc);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    double best = 1e9, sum = 0;
    int at = 0;
    for (int rep = 0; rep < reps; ++rep) {
        for (int e = 0; e < m; ++e, at = (at + 1) % POOL) {
            ex[e].gu = mem + (size_t) at * per, ex[e].dn = ex[e].gu + gb, ex[e].n = ntok;
            for (int k = 0; k < ntok; ++k) ex[e].rows[k] = (e + k) % rows.T;
            ex[e].h = h.data() + (size_t) e * ntok * w.ffs, ex[e].out = out.data() + (size_t) e * ntok * w.N;
        }
        w.ex = ex.data(), w.m = m;
        const auto t0 = std::chrono::steady_clock::now();
        pool.run(w, &rows);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (rep >= reps / 5) best = std::min(best, ms), sum += ms;
    }
    const double mean = sum / (reps - reps / 5), mb = (double) m * per / 1e6;
    std::printf("%s|%s, %d experts' halves (%.2f MB) x %d tokens, %d threads on node %d: mean %.3f ms (best %.3f), %.1f GB/s, %.1f us an expert\n",
                type_name(w.gu_type), type_name(w.dn_type), m, mb, ntok, threads, node, mean, best, mb / mean, mean * 1e3 / m);
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && !std::strcmp(argv[1], "check")) return check(argv[2]);
    if (argc >= 2 && !std::strcmp(argv[1], "bench")) return bench(argc, argv);
    return selftest();
}
