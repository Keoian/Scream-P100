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
    float* h = nullptr;            // scratch: [n][ffs] (exact: the card's SwiGLU of gu_out, as floats)
    float* out = nullptr;          // [n][N], row k at out + k * out_stride
    float* gu_out = nullptr;       // exact only: [n][2 * ffs], the gate rows' sums then the up rows'
};
struct Work {
    uint32_t gu_type = 0, dn_type = 0;
    int ffs = 0, N = 0;
    const float* x = nullptr;      // the window's rows, row t at x + t * x_stride
    size_t x_stride = 0, out_stride = 0;
    Expert* ex = nullptr;
    int m = 0;
    bool exact = false;            // FNX_CPU_EXACT: the card's arithmetic (below); gate|up then writes gu_out, not h
    int gu_lanes = 2;              // exact: the card's lanes a gate|up row (FNX_ROW_G, p100_moe_rows.cuh); 1, 2, 4 or 8
};

// FNX_CPU_EXACT=1: the card's own arithmetic, so a row comes out bit for bit as the card's row kernels (p100_moe_rows.cuh) give it.
// The card rounds each weight to fp16 (Q4_K/Q5_K: fp16(q * fp16(d sc) + fp16(-dmin m)) in one rounding; IQ4_NL, Q8_0: fp16(v * d);
// Q5_1: fp16(q * d + m)), sums each 32-value run in two fp16 FMA chains (the even values, the odd ones; 16 steps each from +0),
// adds the run into fp32 as s + (lo + hi), and a gate|up row's `gu_lanes` parts (equal shares of its 256-value groups) at the end,
// as the warp's xor shuffles do. A product of two fp16 values is exact in fp32; each fp16 FMA is the fp32 sum rounded to odd and
// then to fp16, which is the correctly rounded result (24 bits >= 11 + 2). The SwiGLU between gate|up and down is not done here:
// the card's (__expf and the division under --use_fast_math) cannot be reproduced on a CPU, so the card computes it (p100_cpu.cuh).
bool exact_ok(uint32_t gu_type, uint32_t dn_type);   // gate|up Q4_K or Q5_K, down IQ4_NL, Q5_1 or Q8_0
// the oracle, on the integers (no floating point): IEEE binary16 fma and round-from-fp32, to nearest, ties to even
uint16_t f16_fma_ref(uint16_t a, uint16_t b, uint16_t c);
uint16_t f16_from_f32_ref(float f);
float f16_to_f32(uint16_t h);
// the card's sum of one row with the oracle's steps: gate|up (Q4_K/Q5_K, n values, `lanes` parts) times x; down times h (fp16 values)
float gu_row_card_ref(uint32_t type, const uint8_t* row, int n, const uint16_t* x, int lanes);
float dn_row_card_ref(uint32_t type, const uint8_t* row, int n, const uint16_t* h);
// the AVX2 fp16 FMA of the exact kernels on 8 values (w * x exact in fp32), in its rounding mode: for the test; false without AVX2
bool fma16z_x8(const float* w, const float* x, const float* c, float* out);
// runs of the exact kernels' chains that left fp16's range and were done again by the oracle, since the start
uint64_t exact_ref_runs();
// gate|up rows [j0, j1) of the card's ffs: h[k][j] for the expert's tokens; then down rows [i0, i1): out[k][i]
void gu_chunk(const Work& w, const Expert& e, int j0, int j1);
// FNX_CPU_EXACT only: rows [r0, r1) of the expert's 2 * ffs gate|up rows (the gate rows, then the up rows): the pool's chunks
void gu_rows_exact(const Work& w, const Expert& e, int r0, int r1);
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
    // since the last call: the chunks' own time added over the threads, the rounds' time (set-up to the last chunk done), the time
    // from a round's set-up to the first chunk a worker began (added over the rounds), the chunks and the rounds
    struct Stat {
        unsigned long long busy_ns = 0, wall_ns = 0, first_ns = 0, chunks = 0, rounds = 0;
    };
    Stat stats();
    void run(Work& w, const Rows* rows = nullptr);
    // the two halves of run() alone (FNX_CPU_EXACT: the card's SwiGLU comes between them)
    void run_gu(Work& w, const Rows* rows = nullptr);
    void run_down(Work& w);
    int threads() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// the CPUs of one NUMA node's cores, one logical CPU a core (its first sibling), from sysfs; empty when it cannot be read
std::vector<int> node_cores(int node);
// the other logical CPUs of cpu's core (its hyperthreads), from sysfs
std::vector<int> core_siblings(int cpu);
// FNX_CPU_WARM: the pool's waiting threads (and whoever calls warm256 while it waits) run 256-bit FMAs: a list after a pause of
// some milliseconds then starts at full speed
void keep_warm(bool on);
void warm256();

}  // namespace cpux
}  // namespace fnx
