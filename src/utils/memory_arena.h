#pragma once

// CUDA memory arena
#include "cuda_utils.h"
#include "exception.h"
#include "utils.h"

class MemoryArena {
public:
    MemoryArena(size_t bytes) {
        CUDA_CHECK(cudaMalloc(&m_base, bytes));
        m_capacity = bytes;
    }

    ~MemoryArena() {
        if (m_base)
            cudaFree(m_base);
    }

    inline void Reset() { m_offset = 0; }

    template<typename T>
    void *Alloc(size_t count = 1) {
        if (count == 0)
            throw std::bad_alloc();

        size_t bytes = sizeof(T) * count;
        size_t alignment = std::max(alignof(T), 256);     

        size_t aligned_offset = Utils::AlignUp(bytes, alignment);
        if (aligned_offset + bytes > m_capacity)
            throw std::bad_alloc();
        T *ptr = reinterpret_cast<T*>(static_cast<char*>(m_base) + aligned_offset);
        m_offset = aligned_offset + bytes;
        return ptr;   
    }
private:
    void *m_base = nullptr;
    size_t m_capacity = 0;
    size_t m_offset = 0;
};
