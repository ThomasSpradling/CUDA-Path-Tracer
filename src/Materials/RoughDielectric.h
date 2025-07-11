#pragma once

#include "common.h"

struct RoughDielectric {
    Texture<glm::vec3> albedo;

    Texture<float> roughness_map;
    float ior;

    __device__ BSDFSample Sample(const Intersection &intersection, const glm::vec3 &w_out, thrust::default_random_engine &rng) {        
        BSDFSample sample;

        float cos_o = glm::dot(intersection.normal, w_out);
        if (cos_o <= 0.0f) {
            sample.type = BSDFSampleType::InvalidSample;
            return sample;
        }
        cos_o = fabsf(cos_o);

        glm::vec3 color = albedo.Get(intersection.uv);
        float roughness = roughness_map.Get(intersection.uv);

        // clamp roughness
        float a = glm::clamp(roughness, 0.01f, 1.0f);

        // 1) sample h from GGX
        glm::vec3 h = Math::SampleGGX(intersection.normal, a, rng);
        float     D = TrowbridgeReitz(intersection.normal, h, a);

        // 2) Fresnel using *ior* (Refract will invert internally as needed)
        float F  = Fresnel(glm::dot(w_out, h), 1.0f / ior);
        float Tt = 1.0f - F;

        // 3) branch
        if (Math::Sample1D(rng) < F) {
            // reflection
            glm::vec3 w_in = Math::Reflect(w_out, h);
            float     cos_i = glm::dot(intersection.normal, w_in);
            if (cos_i <= 0.0f) {
                sample.type = BSDFSampleType::InvalidSample;
                return sample;
            }
            cos_i = fabsf(cos_i);

            sample.type = BSDFSampleType::Glossy | BSDFSampleType::Reflection;
            sample.w_in  = w_in;

            float G = SmithG(intersection.normal, w_in, w_out, a);
            float denom = 4.0f * cos_i * cos_o;
            sample.bsdf = glm::vec3(F) * (D * G / denom);
            sample.pdf  = F * (D * glm::dot(intersection.normal, h)
                             / (4.0f * fabsf(glm::dot(w_out, h))));
        }
        else {
            // refraction
            glm::vec3 w_in;
            float     eta = 1.0f / ior;  
            if (!Math::Refract(w_out, h, eta, w_in)) {
                // TIR: treat as invalid so direct bounce picks reflection next
                sample.type = BSDFSampleType::InvalidSample;
                return sample;
            }

            float cos_i = glm::dot(intersection.normal, w_in);
            if (cos_i >= 0.0f) {
                sample.type = BSDFSampleType::InvalidSample;
                return sample;
            }
            cos_i = fabsf(cos_i);

            sample.type = BSDFSampleType::Specular | BSDFSampleType::Transmission;
            sample.w_in  = w_in;

            float G      = SmithG(intersection.normal, w_in, w_out, a);
            float ho     = glm::dot(w_out, h);
            float hi     = glm::dot(w_in,  h);
            float denomJ = (eta * ho + hi);
            denomJ       = denomJ * denomJ;

            // BTDF (Walter2007) tinted by albedo
            sample.bsdf = color
                        * (Tt * eta * eta * D * G
                           * fabsf(ho * hi))
                        / (denomJ * cos_o * cos_i);

            sample.pdf  = Tt
                        * (D * glm::dot(intersection.normal, h)
                           * (eta * eta * fabsf(hi))
                           / denomJ);
        }

        return sample;
    }

};
