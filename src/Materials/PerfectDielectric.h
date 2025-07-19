#pragma once

#include "common.h"

struct PerfectDielectric {
    Texture<glm::vec3> albedo;
    float ior;

    __device__ BSDFSample Sample(const Intersection &intersection, const glm::vec3 &w_out, thrust::default_random_engine &rng) {        
        BSDFSample sample;
        glm::vec3 normal = intersection.normal;
        float eta = 1.0f / ior;

        // Flip shading frame if on other side
        float cos_out = glm::dot(intersection.normal, w_out);
        if (cos_out < 0.0f) {
            normal = -normal;
            cos_out = -cos_out;
            eta = ior;
        }

        float reflectance = Fresnel(cos_out, eta);
        float transmittance = 1.0f - reflectance;

        glm::vec3 color = albedo.Get(intersection.uv);

        if (Math::Sample1D(rng) < reflectance) {
            glm::vec3 w_in = Math::Reflect(w_out, normal);

            // Reflected vector must be on opposite side.
            float cos_in = glm::dot(normal, glm::normalize(w_in));
            if (cos_in < 0.0f) {
                sample.type = BSDFSampleType::InvalidSample;
                return sample;
            }

            sample.type = BSDFSampleType::Specular | BSDFSampleType::Reflection;
            sample.w_in = w_in;
            sample.bsdf = color * reflectance / cos_in;
            sample.pdf = reflectance;
            return sample;
        } else {
            glm::vec3 w_in;
            if (!Math::Refract(w_out, normal, eta, w_in)) {
                sample.type = BSDFSampleType::InvalidSample;
                return sample;
            }

            // Transmitted vector must be on opposite side.
            float cos_in = glm::dot(w_in, normal);
            if (cos_in > 0.0f) {
                sample.type = BSDFSampleType::InvalidSample;
                return sample;
            }
            cos_in = fabsf(cos_in);

            sample.type = BSDFSampleType::Specular | BSDFSampleType::Transmission;
            sample.w_in = w_in;
            sample.bsdf = color * transmittance * eta * eta / cos_in;
            sample.pdf = transmittance;
            return sample;
        }
    }
};
