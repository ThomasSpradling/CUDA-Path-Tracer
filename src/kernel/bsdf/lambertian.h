#pragma once

#include "common.h"

__device__ __forceinline__ BSDF_Sample SampleLambertian(
    const LambertianMaterial &material,
    const DeviceTexturePool &texture_pool,
    const Intersection &intersection,
    const glm::vec3 &w_out,
    Sampler &sampler
) {
    glm::vec3 w_in = Math::SampleCosineHemisphere(intersection.normal, Sample2D(sampler));
    glm::vec3 albedo = SampleTexture<glm::vec3>(texture_pool, material.albedo_texture, intersection.uv);

    BSDF_Sample result {};
    result.bsdf = albedo * Math::INV_PI;
    result.pdf = fmaxf(0.0f, glm::dot(intersection.normal, w_in)) * Math::INV_PI;
    result.type = BSDF_SampleType::Diffuse;
    result.w_in = w_in;

    return result;
}

__device__ __forceinline__ glm::vec3 EvalLambertianBSDF(
    const LambertianMaterial &material,
    const DeviceTexturePool &texture_pool,
    const Intersection &intersection,
    const glm::vec3 &w_in,
    const glm::vec3 &w_out
) {
    return SampleTexture<glm::vec3>(texture_pool, material.albedo_texture, intersection.uv) * Math::INV_PI;
}

__device__ __forceinline__ float LambertianPDF(
    const LambertianMaterial &material,
    const DeviceTexturePool &texture_pool,
    const Intersection &intersection,
    const glm::vec3 &w_in,
    const glm::vec3 &w_out
) {
    return fmaxf(0.0f, glm::dot(intersection.normal, w_in)) * Math::INV_PI;
}
