#pragma once

#include <thrust/random.h>
#include <glm/glm.hpp>
#include "../../math/hash.h"

// Uses PCG as a psuedo-random generator
// See: https://en.wikipedia.org/wiki/Permuted_congruential_generator

struct IndependentSampler {
    uint64_t state;
};

__device__ __forceinline__ void SeedSampler(IndependentSampler &sampler, glm::ivec2 pixel, int iteration) {
    uint32_t key_xy = (uint32_t(pixel.x) & 0xFFFFu)
                    | ((uint32_t(pixel.y) & 0xFFFFu) << 16);
    
    // Mix with Knuth's golden ratio as in https://en.wikipedia.org/wiki/Tiny_Encryption_Algorithm
    uint32_t key = key_xy ^ (uint32_t(iteration) * 0x9E3779B9u);

    uint32_t hashed = Math::PCG_Hash(key);
    sampler.state = uint64_t(hashed) << 32 | uint64_t(hashed ^ 0xdeadbeefu);
}

__device__ __forceinline__ float Sample1D(IndependentSampler &sampler) {
    uint64_t old = sampler.state;
    sampler.state = old * 6364136223846793005ULL + 1;
    uint32_t x = uint32_t(((old >> 18) ^ old) >> 27);
    uint32_t r = uint32_t(old >> 59);
    uint32_t v = (x >> r) | (x << ((-int(r)) & 31));
    return v * (1.0f / 4294967296.0f);
}
