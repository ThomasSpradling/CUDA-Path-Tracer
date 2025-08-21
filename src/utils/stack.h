#pragma once

template<typename T, size_t N = 64>
struct Stack {
    T data[N];
    uint32_t top = 0u;

    __device__ void Push(const T &item) {
        if (top < N)
            data[top++] = item;
    }

    __device__ T Pop() {
        if (top > 0)
            return data[--top];
        return {};
    }

    __device__ bool Empty() {
        return top == 0u;
    }
};
