#include "elastic.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace fnx {

bool vram_elastic() {
    static const bool on = [] {
        const char* v = std::getenv("FNX_VRAM_ELASTIC");
        if (!v || !*v || !std::strcmp(v, "0")) return false;
        if (std::strcmp(v, "1")) throw std::runtime_error(std::string("FNX_VRAM_ELASTIC: 0 or 1, not '") + v + "'");
        return true;
    }();
    return on;
}

}  // namespace fnx

#if FNX_HIP
namespace fnx {
static void no_elastic() { throw std::runtime_error("--vram-elastic (FNX_VRAM_ELASTIC=1) is not built on HIP"); }
void* ElasticRegion::alloc(int, size_t) { no_elastic(); return nullptr; }
size_t ElasticRegion::give(size_t, size_t) { no_elastic(); return 0; }
size_t ElasticRegion::take() { return 0; }
size_t ElasticRegion::given() const { return 0; }
void ElasticRegion::release() {}
}  // namespace fnx
#else
#include <cuda.h>
#include <cuda_runtime.h>

#include <chrono>
#include <thread>

namespace fnx {
namespace {

// 32 MiB a chunk by default (16 of the P100's 2 MiB; FNX_VRAM_CHUNK_MIB, sfnx's --vram-segment-mib, sets another, rounded up
// to the granularity): a few hundred mappings a card, and a run given back loses at most a chunk at each end to the slots
// beside it
size_t chunk_bytes() {
    const char* v = std::getenv("FNX_VRAM_CHUNK_MIB");
    if (!v || !*v) return 32u << 20;
    char* end = nullptr;
    const long mib = std::strtol(v, &end, 10);
    if (*end || mib < 1 || mib > 4096) throw std::runtime_error(std::string("FNX_VRAM_CHUNK_MIB: MiB 1..4096, not '") + v + "'");
    return (size_t) mib << 20;
}

void cu(CUresult r, const std::string& what) {
    if (r == CUDA_SUCCESS) return;
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    throw std::runtime_error("--vram-elastic: " + what + ": " + (s ? s : "CUDA error " + std::to_string((int) r)));
}

CUmemAllocationProp prop_of(int dev) {
    CUmemAllocationProp p{};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = dev;
    return p;
}

// chunks [lo, hi) mapped (fresh physical memory each), readable and writable by the card, zeroed
CUresult map_run(const ElasticRegion& r, size_t lo, size_t hi) {
    const CUmemAllocationProp p = prop_of(r.dev);
    for (size_t i = lo; i < hi; ++i) {
        CUmemGenericAllocationHandle h = 0;
        if (const CUresult e = cuMemCreate(&h, r.chunk, &p, 0); e != CUDA_SUCCESS) {
            for (size_t j = lo; j < i; ++j) cuMemUnmap((CUdeviceptr) (r.base + j * r.chunk), r.chunk);
            return e;
        }
        const CUresult e = cuMemMap((CUdeviceptr) (r.base + i * r.chunk), r.chunk, 0, h, 0);
        cuMemRelease(h);   // the mapping keeps the memory; unmapping it gives the memory back
        if (e != CUDA_SUCCESS) {
            for (size_t j = lo; j < i; ++j) cuMemUnmap((CUdeviceptr) (r.base + j * r.chunk), r.chunk);
            return e;
        }
    }
    CUmemAccessDesc a{};
    a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    a.location.id = r.dev;
    a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (const CUresult e = cuMemSetAccess((CUdeviceptr) (r.base + lo * r.chunk), (hi - lo) * r.chunk, &a, 1); e != CUDA_SUCCESS) {
        for (size_t j = lo; j < hi; ++j) cuMemUnmap((CUdeviceptr) (r.base + j * r.chunk), r.chunk);
        return e;
    }
    if (cudaMemset((void*) (r.base + lo * r.chunk), 0, (hi - lo) * r.chunk) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

}  // namespace

void* ElasticRegion::alloc(int d, size_t bytes) {
    dev = d;
    CUdevice cd = 0;
    cu(cuDeviceGet(&cd, d), "card " + std::to_string(d));
    int vmm = 0;
    cu(cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cd), "card " + std::to_string(d) + "'s attributes");
    if (!vmm) throw std::runtime_error("--vram-elastic: card " + std::to_string(d) + " has no virtual memory management (CUDA's VMM API)");
    const CUmemAllocationProp p = prop_of(d);
    size_t g = 0;
    cu(cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "the allocation granularity");
    const size_t want = chunk_bytes();
    chunk = (want + g - 1) / g * g;
    const size_t n = (bytes + chunk - 1) / chunk;
    CUdeviceptr va = 0;
    cu(cuMemAddressReserve(&va, n * chunk, 0, 0, 0), "reserving " + std::to_string((n * chunk) >> 20) + " MiB of addresses");
    base = (uint64_t) va;
    mapped.assign(n, 0);
    if (const CUresult e = map_run(*this, 0, n); e != CUDA_SUCCESS) {
        size_t fr = 0, tot = 0;
        cudaMemGetInfo(&fr, &tot);
        cuMemAddressFree(va, n * chunk);
        base = 0;
        cu(e, "mapping " + std::to_string((n * chunk) >> 20) + " MiB on card " + std::to_string(d) + " (" + std::to_string(fr >> 20) + " MiB free)");
    }
    mapped.assign(n, 1);
    return (void*) base;
}

size_t ElasticRegion::give(size_t lo, size_t hi) {
    if (!base || hi <= lo) return 0;
    const size_t first = (lo + chunk - 1) / chunk, last = std::min(hi / chunk, mapped.size());
    size_t n = 0;
    for (size_t i = first; i < last; ++i)
        if (mapped[i]) {
            cu(cuMemUnmap((CUdeviceptr) (base + i * chunk), chunk), "giving back a chunk");
            mapped[i] = 0;
            ++n;
        }
    return n * chunk;
}

size_t ElasticRegion::take() {
    if (!base) return 0;
    size_t n = 0;
    for (size_t i = 0; i < mapped.size();) {
        if (mapped[i]) { ++i; continue; }
        size_t j = i;
        while (j < mapped.size() && !mapped[j]) ++j;
        // the reader that had this memory may still be ending: its process's memory is the driver's again a moment later
        CUresult e = CUDA_SUCCESS;
        for (int tries = 0; tries < 50; ++tries) {
            e = map_run(*this, i, j);
            if (e != CUDA_ERROR_OUT_OF_MEMORY) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        cu(e, "taking back " + std::to_string(((j - i) * chunk) >> 20) + " MiB on card " + std::to_string(dev));
        for (size_t k = i; k < j; ++k) mapped[k] = 1;
        n += j - i;
        i = j;
    }
    return n * chunk;
}

size_t ElasticRegion::given() const {
    size_t n = 0;
    for (const char m : mapped) n += !m;
    return n * chunk;
}

void ElasticRegion::release() {
    if (!base) return;
    for (size_t i = 0; i < mapped.size(); ++i)
        if (mapped[i]) cuMemUnmap((CUdeviceptr) (base + i * chunk), chunk);
    cuMemAddressFree((CUdeviceptr) base, mapped.size() * chunk);
    base = 0;
    mapped.clear();
}

}  // namespace fnx
#endif  // FNX_HIP
