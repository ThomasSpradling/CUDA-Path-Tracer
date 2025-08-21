#pragma once

#include "common.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>

__device__ __forceinline__ BSDF_Sample SampleMetallicRoughness(
    const MetallicRoughnessMaterial &material,
    const DeviceTexturePool &texture_pool,
    const Intersection &intersection,
    const glm::vec3 &w_out,
    Sampler &sampler
) {
    BSDF_Sample result;

    // Flip shading frame as needed
    glm::vec3 normal = intersection.normal;
    float cos_out = glm::dot(normal, w_out);
    if (cos_out < 0.0f) {
        normal = -normal;
        cos_out = -cos_out;
    }

    // Sample textures
    glm::vec3 base_color = SampleTexture<glm::vec3>(texture_pool, material.albedo_texture, intersection.uv);
    float metallic = SampleTexture<float>(texture_pool, material.metallic_texture, intersection.uv);
    float alpha = SampleTexture<float>(texture_pool, material.roughness_texture, intersection.uv);
    alpha = glm::clamp(alpha, 0.02f, 1.0f);
    alpha *= alpha;

    // Metallic roughness params
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), base_color, metallic);

    float w_diff = Math::Luminance(base_color) * (1.0f - metallic);
    float w_spec = Math::Luminance(F0);
    float w_sum = glm::max(w_spec + w_diff, Math::EPSILON);

    w_diff /= w_sum;
    w_spec /= w_sum;

    if (Sample1D(sampler) < w_spec) {
        // Sample half vector (micro-facet normal) and use this for reflection
        glm::vec3 half_vec = Math::SampleGGX(normal, alpha, Sample2D(sampler));
        glm::vec3 w_in = Math::Reflect(w_out, half_vec);
        float cos_in = glm::dot(normal, w_in);

        // Ensure result is on same hemisphere
        if (cos_in <= 0.0f) {
            result.type = BSDF_SampleType::Invalid;
            return result;
        }

        float D = TrowbridgeReitz(normal, half_vec, alpha);
        float G = SmithG(normal, w_out, w_in, alpha);

        // Compute PDF and BSDF
        glm::vec3 refl_in = FresnelSchlick(fmaxf(glm::dot(w_out, half_vec), 0.0f), F0);
        float pdf_specular = (D * fmaxf(glm::dot(normal, half_vec), 0.0f)) / (4.0f * fmaxf(fabsf(glm::dot(w_out, half_vec)), Math::EPSILON));

        result.type = BSDF_SampleType::Glossy | BSDF_SampleType::Reflection;
        result.bsdf = refl_in * (D * G) / fmaxf(4.0f * cos_in * cos_out, 1e-6f);
        result.pdf = w_spec * pdf_specular;
        result.w_in = w_in;
        
        return result;
    } else {
        glm::vec3 w_in = Math::SampleCosineHemisphere(normal, Sample2D(sampler));
        float cos_in = fmaxf(glm::dot(normal, w_in), 0.0f);
        
        // Ensure result is on same hemisphere
        if (cos_in <= 0.0f) {
            result.type = BSDF_SampleType::Invalid;
            return result;
        }

        glm::vec3 sum_dir = w_in + w_out;                                                     
        glm::vec3 half_vec = (glm::length2(sum_dir) >= Math::EPSILON)                         
            ? glm::normalize(sum_dir)                                                         
            : normal;                                                                         
        float cos_f = fabsf(glm::dot(w_out, half_vec));                                       
        glm::vec3 refl_out = FresnelSchlick(cos_f, F0);

        result.type = BSDF_SampleType::Diffuse | BSDF_SampleType::Reflection;
        result.w_in = w_in;
        result.bsdf = base_color * (1.0f - metallic) * (1.0f - refl_out) * Math::INV_PI;
        result.pdf = w_diff * cos_in * Math::INV_PI;
        return result;
    }
}

__device__ __forceinline__ glm::vec3 EvalMetallicRoughnessBSDF(
    const MetallicRoughnessMaterial &material,
    const DeviceTexturePool &texture_pool,
    const Intersection &intersection,
    const glm::vec3 &w_in,
    const glm::vec3 &w_out
) {
    // Ensure normal on correct side
    glm::vec3 normal = intersection.normal;
    float cos_out = glm::dot(normal, w_out);
    float cos_in = glm::dot(normal, w_in);
    if (glm::dot(normal, w_out) <= 0.0f || glm::dot(normal, w_in) <= 0.0f)
        return glm::vec3(0.0f);

    // Sample textures
    glm::vec3 base_color = SampleTexture<glm::vec3>(texture_pool, material.albedo_texture, intersection.uv);
    float metallic = SampleTexture<float>(texture_pool, material.metallic_texture, intersection.uv);
    float alpha = SampleTexture<float>(texture_pool, material.roughness_texture, intersection.uv);
    alpha = glm::clamp(alpha, 0.02f, 1.0f);
    alpha *= alpha;
    
    // Compute reflectance and metallic roughness params
    glm::vec3 half_vec = glm::normalize(w_in + w_out);
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), base_color, metallic);
    float D = TrowbridgeReitz(normal, half_vec, alpha);
    float G = SmithG(normal, w_out, w_in, alpha);
    
    // Specular component
    glm::vec3 refl_in = FresnelSchlick(glm::dot(w_in, half_vec), F0);
    glm::vec3 specular = refl_in * (D * G) / glm::max(4.0f * cos_in * cos_out, Math::EPSILON);

    // Diffuse component
    glm::vec3 refl_out = FresnelSchlick(glm::dot(w_out, half_vec), F0);
    glm::vec3 diffuse = base_color * (1.0f - metallic) * (1.0f - refl_out) * (1.0f - refl_in) * Math::INV_PI;
    return diffuse + specular;
}

__device__ __forceinline__ float MetallicRoughnessPDF(
    const MetallicRoughnessMaterial &material,
    const DeviceTexturePool &texture_pool,
    const Intersection &intersection,
    const glm::vec3 &w_in,
    const glm::vec3 &w_out
) {
    // Ensure normal on correct side
    glm::vec3 normal = intersection.normal;
    float cos_out = glm::dot(normal, w_out);
    float cos_in = glm::dot(normal, w_in);
    if (cos_out <= 0.0f || cos_in <= 0.0f)
        return 0.0f;

    // Sample textures
    glm::vec3 base_color = SampleTexture<glm::vec3>(texture_pool, material.albedo_texture, intersection.uv);
    float metallic = SampleTexture<float>(texture_pool, material.metallic_texture, intersection.uv);
    float alpha = SampleTexture<float>(texture_pool, material.roughness_texture, intersection.uv);
    alpha = glm::clamp(alpha, 0.02f, 1.0f);
    alpha *= alpha;

    // Compute reflectance and metallic roughness params
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), base_color, metallic);

    float w_diff = Math::Luminance(base_color) * (1.0f - metallic);
    float w_spec = Math::Luminance(F0);
    float w_sum = glm::max(w_spec + w_diff, Math::EPSILON);

    w_diff /= w_sum;
    w_spec /= w_sum;

    // Diffuse PDF
    float pdf_diffuse = cos_in * Math::INV_PI;

    // Specular PDF
    float pdf_specular = 0.0f;
    glm::vec3 half_vec = w_in + w_out;
    if (glm::length2(half_vec) >= Math::EPSILON) {
        half_vec = glm::normalize(half_vec);
        float n_dot_h = glm::max(glm::dot(normal, half_vec), 0.0f);
        float out_dot_h = glm::max(glm::dot(w_out, half_vec), 0.0f);
        if (n_dot_h > 0.0f && out_dot_h > 0.0f) {
            float D = TrowbridgeReitz(normal, half_vec, alpha);
            pdf_specular = (D * n_dot_h) / glm::max(4.0f * out_dot_h, Math::EPSILON);
        }
    }

    return w_diff * pdf_diffuse + w_spec * pdf_specular;
}

