#pragma once

#include "memory_arena.h"

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer(MemoryArena &arena)
        : m_arena(arena)
        , m_ptr(nullptr)
        , m_size(0)
        , m_capacity(0)
    {}

    void Update(const std::vector<T> &host) {
        size_t n = host.size();
        if (n > m_capacity) {
            size_t new_cap = std::max(n, m_capacity ? m_capacity * 2 : 1);
            m_ptr = m_arena.Alloc<T>(new_cap);
            m_capacity = new_cap;
        }

        if (n > 0) {
            CUDA_CHECK(cudaMemcpy(m_ptr, host.data(), n * sizeof(T), cudaMemcpyHostToDevice));
        }
        m_size = n;
    }

    inline T *Data() const { return m_ptr; }
    inline size_t Size() const { return m_size;  }
    inline size_t Capacity() const { return m_capacity; }
private:
    MemoryArena &m_arena;
    T *m_ptr;
    size_t m_size, m_capacity;
};
