#pragma once

#include "common.h"

__device__ __forceinline__ BSDF_Sample SampleDielectric(
    const DielectricMaterial &material,
    const DeviceTexturePool &texture_pool,
    const Intersection &intersection,
    const glm::vec3 &w_out,
    Sampler &sampler
) {
    BSDF_Sample result;
    glm::vec3 normal = intersection.normal;
    float eta = 1.0f / material.ior;

    // Flip shading frame if on other side
    float cos_out = glm::dot(intersection.normal, w_out);
    if (cos_out < 0.0f) {
        normal = -normal;
        cos_out = -cos_out;
        eta = material.ior;
    }

    float reflectance = Fresnel(cos_out, eta);
    float transmittance = 1.0f - reflectance;

    glm::vec3 color = SampleTexture<glm::vec3>(texture_pool, material.albedo_texture, intersection.uv);

    if (Sample1D(sampler) < reflectance) {
        glm::vec3 w_in = Math::Reflect(w_out, normal);

        // Reflected vector must be on opposite side.
        float cos_in = glm::dot(normal, glm::normalize(w_in));
        if (cos_in < 0.0f) {
            result.type = BSDF_SampleType::Invalid;
            return result;
        }

        result.type = BSDF_SampleType::Specular | BSDF_SampleType::Reflection;
        result.w_in = w_in;
        result.bsdf = color * reflectance / cos_in;
        result.pdf = reflectance;
        return result;
    } else {
        
        glm::vec3 w_in;
        if (!Math::Refract(w_out, normal, eta, w_in)) {
            result.type = BSDF_SampleType::Invalid;
            return result;
        }
        
        // Transmitted vector must be on opposite side.
        float cos_in = glm::dot(w_in, normal);
        if (cos_in > 0.0f) {
            result.type = BSDF_SampleType::Invalid;
            return result;
        }
        cos_in = fabsf(cos_in);
                
        result.type = BSDF_SampleType::Specular | BSDF_SampleType::Transmission;
        result.w_in = w_in;
        result.bsdf = color * transmittance * eta * eta / cos_in;
        result.pdf = transmittance;
        return result;
    }
}
