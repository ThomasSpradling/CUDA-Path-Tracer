#pragma once

#include "common.h"

struct MetallicRoughness {
    Texture<glm::vec3> albedo;

    Texture<float> metallic_map;
    Texture<float> roughness_map;

    __device__ BSDFSample Sample(const Intersection &intersection, const glm::vec3 &w_out, thrust::default_random_engine &rng) {        
        BSDFSample sample;

        glm::vec3 normal = intersection.normal;
        float cos_out = glm::dot(intersection.normal, w_out);
        if (cos_out < 0.0f) {
            normal = -normal;
            cos_out = -cos_out;
        }

        glm::vec3 color = albedo.Get(intersection.uv);
        float metallic = metallic_map.Get(intersection.uv);
        float roughness = roughness_map.Get(intersection.uv);

        float clamped_roughness = glm::clamp(roughness, 0.01f, 1.0f);
        glm::vec3 specular_k = glm::mix(glm::vec3(0.04f), color, metallic);
        glm::vec3 diffuse_k = color * (1.0f - metallic);

        float reflectance = FresnelSchlick(cos_out, specular_k.r);

        float l_s = (specular_k.r + specular_k.g + specular_k.b) * (1.0f / 3.0f);
        float l_d = (diffuse_k.r + diffuse_k.g + diffuse_k.b) * (1.0f / 3.0f) * (1.0f - reflectance);
        float sum = l_s + l_d;
        float spec_prob = (sum > 0.0f) ? (l_s / sum) : 0.0f;

        // glm::vec3 reflectance = FresnelSchlick(cos_out, base_reflectance);

        if (Math::Sample1D(rng) < spec_prob) {
            glm::vec3 half_vec = Math::SampleGGX(normal, clamped_roughness, rng);
            
            glm::vec3 w_in = Math::Reflect(w_out, half_vec);
            float cos_in = glm::dot(normal, w_in);
            if (cos_in < 0.0f) {
                sample.type = BSDFSampleType::InvalidSample;
                return sample;
            }

            sample.type = BSDFSampleType::Glossy | BSDFSampleType::Reflection;
            sample.w_in = w_in;
            
            float normal_density = TrowbridgeReitz(normal, half_vec, clamped_roughness);
            
            // BSDF
            float masking_func = SmithG(normal, w_in, w_out, clamped_roughness);
            float refl_in = FresnelSchlick(glm::dot(half_vec, w_out), specular_k.r);
            float denom = 4.0f * cos_in * cos_out;
            
            sample.bsdf = specular_k * refl_in * (normal_density * masking_func / denom);
            sample.pdf = spec_prob * (normal_density * glm::dot(normal, half_vec)
                / (4.0f * fabsf(glm::dot(w_out, half_vec))));
            return sample;
        } else {
            glm::vec3 w_in = Math::SampleCosineHemisphere(normal, rng);
            float cos_in = fabsf(glm::dot(normal, w_in));
            if (cos_in < 0.0f) {
                sample.type = BSDFSampleType::InvalidSample;
                return sample;
            }

            glm::vec3 half_vec = glm::normalize(w_in + w_out);
            float refl_in = FresnelSchlick(glm::dot(half_vec, w_in), specular_k.r);

            sample.bsdf = diffuse_k * (1.0f - reflectance) * (1.0f - refl_in) * c_INV_PI;
            sample.pdf  = (1.0f - spec_prob) * cos_in * c_INV_PI;
            sample.type = BSDFSampleType::Diffuse | BSDFSampleType::Reflection;
            sample.w_in = w_in;
            return sample;
        }
    }

};