#pragma once

#include <vector>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <string>
#include "exception.h"

#define CUDA_CHECK(expr) \
    if (cudaError_t err = (expr); err != cudaSuccess) { \
        PT_ERROR("Call '" + std::string(#expr) + "' returned " + std::string(cudaGetErrorString(err)) + "."); \
    }

template<typename T>
uint32_t CopyToDevice(T *&dst, const std::vector<T> &src) {
    size_t count = src.size();
    if (count == 0) {
        return 0;
    }

    if (dst == nullptr) {
        CUDA_CHECK(cudaMalloc((void **) &dst, count * sizeof(T)));
    }

    CUDA_CHECK(cudaMemcpy(dst, src.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    return count;
}
