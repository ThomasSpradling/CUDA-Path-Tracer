#pragma once

#include "common.h"

struct PerfectMirror {
    Texture<glm::vec3> albedo;

    // Texture<glm::vec3> albedo;

    __device__ BSDFSample Sample(const Intersection &intersection, const glm::vec3 &w_out, thrust::default_random_engine &rng) {        
        glm::vec3 w_in = Math::Reflect(w_out, intersection.normal);

        float cos_in = fabsf(glm::dot(intersection.normal, w_in));

        BSDFSample sample {};
        sample.pdf = 1.0f;
        sample.w_in = w_in;
        sample.type = BSDFSampleType::Specular | BSDFSampleType::Reflection;
        sample.bsdf = albedo.Get(intersection.uv) / cos_in;

        return sample;
    }
};
