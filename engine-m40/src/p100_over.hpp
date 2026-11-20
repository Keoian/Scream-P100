// p100_over.hpp: a decode layer's missing experts copied on a stream of their own while the ones in VRAM are computed (p100_moe_rows.cuh).
#pragma once
#include "kernels.hpp"

#include <vector>

namespace fnx {
bool p100_overlap_on();
void p100_moe_rows_phase(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_cap, int T, const uint16_t* x, float* out, int phase, void* st);
void p100_jobs_split(Job* gu, Job* dn, const int* njobs, int layer, const int* list, void* st);
void p100_cache_copy(int layer, size_t gu_stride, size_t dn_stride, const ExpertCache& ec, const int* list, void* st);
void p100_swiglu_after(const float* gu, int ffs, int P, uint16_t* h16, const int* list, void* st);
void p100_tail_slots(const ExpertCache& ec, int p, int count, bool lend, void* st);
// a prompt chunk's attention with Strata's P100 kernels (p100_attn.cuh, int8 K/V)
bool p100_attn_on();
size_t p100_attn_scratch_floats();
void p100_attn_prompt(const float* qg, const uint16_t* kc, const uint16_t* vc, const KvPages& g, const int* pos0, const int* cells, const int* n_cells, int card, int C, int r0, int T,
                      float* part, float* scratch, void* st);
// FNX_P100_DMA=1: a layer's missing experts by the copy engine, a host thread of the card's (p100_dma.cuh)
bool p100_dma_on();
void* p100_dma_start(int dev);
void p100_dma_stop(void* h);
void p100_dma_publish(void* h, int layer, const ExpertCache& ec, const int* list, size_t gu_stride, size_t dn_stride, void* st);
void p100_dma_wait(void* h, void* st);
// FNX_PF_REFILL=1: the copier also fills the prompt path's slots after a read, between its lists: an entry an expert, done its pool's count
// once it has landed (Engine::pf_refill_start, pf_refill_tick)
struct DmaFill {
    const char* src_gu;
    char* dst_gu;
    const char* src_dn;
    char* dst_dn;
    size_t gu_bytes, dn_bytes;
    int pool, done;
};
void p100_dma_fill(void* h, std::vector<DmaFill> jobs, const int* landed);
void p100_dma_fill_stop(void* h);
void p100_dma_landed(void* h, int* out);
// FNX_PREFETCH=1: layer `layer`'s guessed experts (the first *want of each row's guess) not in a slot take one (p100_moe_rows.cuh), the
// copier copies them on a stream of its own (box2), and the layer's window waits for them before its first phase (p100_dma.cuh)
void p100_pf_plan(const int* guess, int T, int layer, const ExpertCache& ec, const int* want, int* list, void* st);
void p100_dma_publish2(void* h, int layer, const ExpertCache& ec, const int* list, size_t gu_stride, size_t dn_stride, void* st);
void p100_dma_wait2(void* h, void* st);
// and a window's (FNX_P100_ATTN_DEC=1): attn()'s partials, its scratch p100_attn_dec_floats(T) floats
bool p100_attn_dec_on();
size_t p100_attn_dec_floats(int T);
void p100_attn_window(const float* qg, const uint16_t* kc, const uint16_t* vc, const KvPages& g, const int* pos0, const int* cells, const int* n_cells, int card, int T,
                      float* part, float* scratch, void* st);
}  // namespace fnx
