#pragma once

#include <thrust/random.h>

#include "sampler/independent.h"
#include "sampler/sobol.h"
#include <glm/glm.hpp>

enum class SamplerType {
    Independent,
    Sobol
};

struct Sampler {
    SamplerType type;

    union {
        IndependentSampler independent;
        SobolSampler sobol;
    };
};

__device__ __forceinline__ void SeedSampler(Sampler &sampler, glm::ivec2 pixel, int iteration) {
    switch (sampler.type) {
        case SamplerType::Independent:
            SeedSampler(sampler.independent, pixel, iteration);
            break;
        case SamplerType::Sobol:
            SeedSampler(sampler.sobol, pixel, iteration);
            break;
    }
}

__device__ __forceinline__ float Sample1D(Sampler &sampler) {
    switch (sampler.type) {
        case SamplerType::Independent:
            return Sample1D(sampler.independent);
        case SamplerType::Sobol:
            return Sample1D(sampler.sobol);
    }
}

__device__ __forceinline__ glm::vec2 Sample2D(Sampler &sampler) {
    return { Sample1D(sampler), Sample1D(sampler) };
}

__device__ __forceinline__ glm::vec3 Sample3D(Sampler &sampler) {
    return { Sample2D(sampler), Sample1D(sampler) };
}

__device__ __forceinline__ glm::vec4 Sample4D(Sampler &sampler) {
    return { Sample3D(sampler), Sample1D(sampler) };
}

// Discrete samplers

__device__ __forceinline__ uint32_t DiscreteSample1D(const DeviceDiscreteSampler1D &discrete_sampler, Sampler &sampler) {
    float x = Sample1D(sampler);
    float scaled = x * float(discrete_sampler.size);

    uint32_t i = static_cast<uint32_t>(scaled);
    if (scaled >= discrete_sampler.size) {
        i = discrete_sampler.size - 1;
        scaled = float(i);
    }

    float y = scaled - static_cast<float>(i);

    return (y < discrete_sampler.probability_table[i]) ? i : discrete_sampler.alias_table[i];
}
