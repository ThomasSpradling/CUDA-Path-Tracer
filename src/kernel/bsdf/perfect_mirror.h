#pragma once

#include "common.h"

__device__ __forceinline__ BSDF_Sample SampleMirror(
    const MirrorMaterial &material,
    const DeviceTexturePool &texture_pool,
    const Intersection &intersection,
    const glm::vec3 &w_out,
    Sampler &sampler
) {
    glm::vec3 w_in = Math::Reflect(w_out, intersection.normal);

    float cos_in = fabsf(glm::dot(intersection.normal, w_in));

    BSDF_Sample result {};
    result.pdf = 1.0f;
    result.w_in = w_in;
    result.type = BSDF_SampleType::Specular | BSDF_SampleType::Reflection;
    result.bsdf = SampleTexture<glm::vec3>(texture_pool, material.albedo_texture, intersection.uv) / cos_in;

    return result;
}
