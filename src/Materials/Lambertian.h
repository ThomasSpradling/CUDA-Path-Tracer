#pragma once

#include "common.h"

struct Lambertian {
    Texture<glm::vec3> albedo;

    __device__ BSDFSample Sample(
        const Intersection &intersection,
        const glm::vec3 &w_out,
        thrust::default_random_engine &rng)
    {
        glm::vec3 w_in = Math::SampleCosineHemisphere(intersection.normal, rng);

        BSDFSample sample {};
        sample.bsdf = albedo.Get(intersection.uv) * c_INV_PI;
        sample.pdf = PDF(intersection.normal, w_in);
        sample.type = BSDFSampleType::Diffuse | BSDFSampleType::Reflection;
        sample.w_in = w_in;

        return sample;
    }

    __device__ float PDF(const glm::vec3 &normal, const glm::vec3 &w_in) {
        return fmaxf(0.0f, glm::dot(normal, w_in)) * c_INV_PI;
    }
};
