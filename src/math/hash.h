#pragma once

namespace Math {

    __host__ __device__ __forceinline__ uint32_t PCG_Hash(uint32_t input) {
        uint32_t state = input * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }
};
