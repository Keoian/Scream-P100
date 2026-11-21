// bench_bw.cu (2026-11-21): host -> card copies of one expert half (1.35 MB) as the writing path makes them: pinned memory on
// each NUMA node, each card alone and both at once, one copy at a time and 16 queued
#include <cuda_runtime.h>
#include <pthread.h>
#include <sched.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("CUDA %s: %s\n", #x, cudaGetErrorString(e)); std::exit(1); } } while (0)
constexpr size_t HALF = 1354752, N = 12;

static void bind_node(int n) {
    cpu_set_t set;
    CPU_ZERO(&set);
    const int cpus[2][8] = {{0, 1, 2, 3, 8, 9, 10, 11}, {4, 5, 6, 7, 12, 13, 14, 15}};
    for (int c : cpus[n]) CPU_SET(c, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}
static uint8_t* host_on(int node) {
    uint8_t* p = nullptr;
    std::thread t([&]() {
        bind_node(node);
        CK(cudaHostAlloc((void**) &p, HALF * N, cudaHostAllocPortable));
        std::memset(p, 1, HALF * N);
    });
    t.join();
    return p;
}
// card c copies N halves from h, one at a time (sync each) or all queued; GB/s
static double run(int c, const uint8_t* h, bool queued) {
    CK(cudaSetDevice(c));
    static uint8_t* d[2] = {};
    if (!d[c]) CK(cudaMalloc(&d[c], HALF * N));
    cudaStream_t st;
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    CK(cudaMemcpyAsync(d[c], h, HALF, cudaMemcpyHostToDevice, st));
    CK(cudaStreamSynchronize(st));
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < N; ++i) {
        CK(cudaMemcpyAsync(d[c] + i * HALF, h + i * HALF, HALF, cudaMemcpyHostToDevice, st));
        if (!queued) CK(cudaStreamSynchronize(st));
    }
    CK(cudaStreamSynchronize(st));
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CK(cudaStreamDestroy(st));
    return HALF * N / s / 1e9;
}

int main() {
    uint8_t* h[2] = {host_on(0), host_on(1)};
    for (int q = 0; q < 2; ++q) {
        std::printf("%s:\n", q ? "16 naenkrat v vrsti (cele vrste)" : "ena kopija naenkrat (cakanje vsake)");
        for (int c = 0; c < 2; ++c)
            for (int n = 0; n < 2; ++n) std::printf("  card %d, RAM on node %d: %.2f GB/s\n", c, n, run(c, h[n], q));
        double r[2];
        std::thread t1([&]() { bind_node(1); r[1] = run(1, h[1], q); });
        bind_node(0);
        r[0] = run(0, h[0], q);
        t1.join();
        std::printf("  both at once (each from its own node): %.2f + %.2f GB/s\n", r[0], r[1]);
    }
    return 0;
}
