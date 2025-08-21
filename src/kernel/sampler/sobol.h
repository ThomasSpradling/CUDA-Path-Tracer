#pragma once

#include <glm/glm.hpp>

struct SobolSampler {
    uint32_t seed;
    uint32_t iteration;
    uint32_t dimension;
};

__device__ __forceinline__ void SeedSampler(const SobolSampler &sampler, glm::ivec2 pixel, int iteration) {
    // Hash pixel to get a seed
    // Set iteration and set dimension to 0
}

__device__ __forceinline__ float Sample1D(const SobolSampler &sampler) {
    // compute using generator matrices; need iteration and dimension
    
    // FastOwen scramble using seed and iteration
    // convert result from uint32_t to float

    // increment dimension
}
