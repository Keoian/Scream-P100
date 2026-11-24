// cpu_experts.hpp: a decode window's routed experts that are not in VRAM, computed on the CPU from their host copies while the cards
// compute the ones that are (FNX_CPU_EXPERTS=1, p100_cpu.cuh). Plain C++ beside the CUDA: the rows' decoders (ggml's block layouts,
// their steps in fp32), the products, and the worker pool of one card's socket. tools/cpu_experts_test.cpp checks the decoders
// against ggml's own steps and times the pool.
//
// One card holds half of every expert (model.cpp, expert_rows2 / expert_cols): gate|up as [gate rows ff0..ff1][up rows ff0..ff1], a
// row N values in the file's blocks; down as N rows of the blocks of columns ff0..ff1. For a token's FFN input x (N values):
//   h[j] = silu(gate[j] . x) * (up[j] . x)   for the card's ffs rows,     out[i] = down[i] . h   for the N output rows,
// which is the card's part of the expert's output (the ring sums the cards' parts, weighted by the router).
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace fnx {
namespace cpux {

// ggml's type numbers (ggml.h), the ones a row decoder exists for
enum : uint32_t { T_Q4_0 = 2, T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_IQ4_NL = 20 };

bool have_simd();                            // AVX2, FMA and F16C on this CPU (without them the decoders are ggml's scalar steps)
bool type_ok(uint32_t type, int n);          // a row of n values of this type can be decoded
size_t row_bytes(uint32_t type, int n);      // its bytes (0: not a type_ok row)
const char* type_name(uint32_t type);
// a row's n values. decode_row: AVX2 where there is a kernel for the type (Q4_K, Q5_K, IQ4_NL), else decode_row_ref;
// decode_row_ref: ggml's dequantize_row_* steps, one value at a time (the test's oracle)
void decode_row(uint32_t type, const uint8_t* src, int n, float* dst);
void decode_row_ref(uint32_t type, const uint8_t* src, int n, float* dst);
float dot(const float* a, const float* b, int n);                    // n a multiple of 8
void f16_to_f32(const uint16_t* src, float* dst, size_t n);

// one card's half of an expert, and the tokens that picked it
struct Expert {
    const uint8_t* gu = nullptr;   // [2 * ffs] rows of N values
    const uint8_t* dn = nullptr;   // [N] rows of ffs values
    int n = 0;                     // its tokens
    int rows[16] = {};             // each one's row of x
    float* h = nullptr;            // scratch: [n][ffs]
    float* out = nullptr;          // [n][N], row k at out + k * out_stride
};
struct Work {
    uint32_t gu_type = 0, dn_type = 0;
    int ffs = 0, N = 0;
    const float* x = nullptr;      // the window's rows, row t at x + t * x_stride
    size_t x_stride = 0, out_stride = 0;
    Expert* ex = nullptr;
    int m = 0;
};
// gate|up rows [j0, j1) of the card's ffs: h[k][j] for the expert's tokens; then down rows [i0, i1): out[k][i]
void gu_chunk(const Work& w, const Expert& e, int j0, int j1);
void down_chunk(const Work& w, const Expert& e, int i0, int i1);

// gu_chunk and down_chunk multiply straight from the blocks where there is an AVX2 kernel for the type (gate|up: Q4_K, Q5_K; down:
// IQ4_NL, Q5_1, Q8_0), in fp32; other types decode a row, then take the dot products.
//
// The pool of one card: `threads` workers (the caller is one more), each pinned to one of `cpus` in turn (empty: not pinned).
// run() computes every expert of the work: all gate|up chunks, then all down chunks, the chunks taken one at a time by whoever
// is free. Workers spin while work comes, sleep 100 us between looks after 20 ms without any and block after a second.
class Pool {
public:
    Pool(int threads, const std::vector<int>& cpus);
    ~Pool();
    // rows (may be null): the window's T rows as IEEE halves, row t at x16 + t * N: converted into xf by the pool before the experts
    // (a row a chunk; four rows or fewer on the calling thread), and the work's x then points at them
    struct Rows {
        const uint16_t* x16 = nullptr;
        int T = 0;
        float* xf = nullptr;
    };
    // workers that came to a finished round while the next was being set up, since the last call (they are turned away)
    unsigned long long late();
    void run(Work& w, const Rows* rows = nullptr);
    int threads() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// the CPUs of one NUMA node's cores, one logical CPU a core (its first sibling), from sysfs; empty when it cannot be read
std::vector<int> node_cores(int node);

}  // namespace cpux
}  // namespace fnx
