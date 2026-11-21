// --vram-elastic (FNX_VRAM_ELASTIC=1): each card's arena allocation (the K/V pool's own pages, then the arena pool's
// slots) from CUDA's virtual memory API, in chunks. A run of lent slots can then go back to the driver, so another
// program on the card (Strata's picture reader) has that VRAM while it runs, and come back later at the same addresses:
// no table, kernel or graph sees a pointer change. Without the flag the arena is one cudaMalloc, as before.
// HIP: not built (the flag refuses at load).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fnx {

bool vram_elastic();   // FNX_VRAM_ELASTIC: 0 (default) or 1; sfnx's --vram-elastic sets it

struct ElasticRegion {
    int dev = -1;
    uint64_t base = 0;          // the reserved address range (0: none)
    size_t chunk = 0;           // bytes a chunk: a multiple of the card's allocation granularity
    std::vector<char> mapped;   // each chunk: 1 mapped, 0 given back to the driver
    // `bytes` on card d (the current device): the range reserved, every chunk mapped, zeroed
    void* alloc(int d, size_t bytes);
    // the chunks wholly inside [lo, hi) (byte offsets from base) unmapped: their memory goes back to the driver. The card
    // must be idle and nothing may touch the range until take(). -> bytes given back
    size_t give(size_t lo, size_t hi);
    // every chunk given back mapped again at its address and zeroed; retried for a few seconds while the driver still
    // counts the memory as another process's (a reader that is ending) -> bytes taken back; throws if it stays short
    size_t take();
    size_t given() const;   // bytes given back now
    void release();         // all of it, the address range too (the engine's end)
};

}  // namespace fnx
