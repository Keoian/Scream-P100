// p100_compat.h: what the split engine's kernels ask of a card and sm_60 does not have (force-included in every CUDA source).
#pragma once
#include <cuda_runtime.h>

// the four-way signed byte dot product, instruction from sm_61 on; here vmad with byte selectors: 4 instructions, the C form took about 12 and 4 more per sum.
// "+&r": the accumulator never shares a register with a, b (Strata PR #4274: with "+r" a forced alias gave wrong sums)
__device__ __forceinline__ int fnx_dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__)
    asm("vmad.s32.s32.s32 %0, %1.b0, %2.b0, %0;" : "+&r"(c) : "r"(a), "r"(b));
    asm("vmad.s32.s32.s32 %0, %1.b1, %2.b1, %0;" : "+&r"(c) : "r"(a), "r"(b));
    asm("vmad.s32.s32.s32 %0, %1.b2, %2.b2, %0;" : "+&r"(c) : "r"(a), "r"(b));
    asm("vmad.s32.s32.s32 %0, %1.b3, %2.b3, %0;" : "+&r"(c) : "r"(a), "r"(b));
    return c;
#else
    return c + (int) (signed char) a * (int) (signed char) b +
           (int) (signed char) (a >> 8) * (int) (signed char) (b >> 8) +
           (int) (signed char) (a >> 16) * (int) (signed char) (b >> 16) + (a >> 24) * (b >> 24);
#endif
}
#define __dp4a fnx_dp4a

// the wait loops' short sleep, sm_70 on; here they simply spin
__device__ __forceinline__ void fnx_nanosleep(unsigned) {}
#define __nanosleep fnx_nanosleep

// Maxwell (sm_52, ai-streznik-m40): no f16 arithmetic, and CUDA 12.4 declares __hfma2 from sm_53 on only (__hmul2, __hsub2 and
// __hadd2 have the header's own per-lane fallback). Here each lane in f32, rounded once to f16 as HFMA2 rounds. Slow: an f16 -> f32
// conversion is quarter rate on Maxwell, so the hot loops (the experts' rows and prompt products) have f32 paths of their own;
// this keeps the rest (FNX_DENSE_H, FNX_HC_F16, the sm < 70 prompt attention) building and correct
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
#include <cuda_fp16.h>
__device__ __forceinline__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c) {
    const float2 fa = __half22float2(a), fb = __half22float2(b), fc = __half22float2(c);
    return __floats2half2_rn(fmaf(fa.x, fb.x, fc.x), fmaf(fa.y, fb.y, fc.y));
}
#endif
