#pragma once

#include "bsdf/perfect_dielectric.h"
#include "bsdf/perfect_mirror.h"
#include "intersection.h"
#include "bsdf/lambertian.h"
#include "bsdf/metallic_roughness.h"

__device__ __forceinline__ BSDF_Sample SampleMaterial(
    const Material &material,
    const DeviceTexturePool &texture_pool,
    const Intersection &intersection,
    const glm::vec3 &w_out,
    Sampler &sampler
) {
    switch (material.type) {
        case Material::Lambertian:
            return SampleLambertian(material.lambertian, texture_pool, intersection, w_out, sampler);
        case Material::Dielectric:
            return SampleDielectric(material.dielectric, texture_pool, intersection, w_out, sampler);
        case Material::Mirror:
            return SampleMirror(material.mirror, texture_pool, intersection, w_out, sampler);
        case Material::MetallicRoughness:
            return SampleMetallicRoughness(material.metallic_roughness, texture_pool, intersection, w_out, sampler);
        default:
            return {};
    }
}

__device__ __forceinline__ glm::vec3 EvalMaterialBSDF(
    const Material &material,
    const DeviceTexturePool &texture_pool,
    const Intersection &intersection,
    const glm::vec3 &w_in,
    const glm::vec3 &w_out
) {
    switch (material.type) {
        case Material::Lambertian:
            return EvalLambertianBSDF(material.lambertian, texture_pool, intersection, w_in, w_out);
        case Material::MetallicRoughness:
            return EvalMetallicRoughnessBSDF(material.metallic_roughness, texture_pool, intersection, w_in, w_out);
        case Material::Mirror:
        case Material::Dielectric:
        default:
            return {};
    }
}

__device__ __forceinline__ float MaterialPDF(
    const Material &material,
    const DeviceTexturePool &texture_pool,
    const Intersection &intersection,
    const glm::vec3 &w_in,
    const glm::vec3 &w_out
) {
    switch (material.type) {
        case Material::Lambertian:
            return LambertianPDF(material.lambertian, texture_pool, intersection, w_in, w_out);
        case Material::MetallicRoughness:
            return MetallicRoughnessPDF(material.metallic_roughness, texture_pool, intersection, w_in, w_out);
        case Material::Mirror:
        case Material::Dielectric:
        default:
            return 0.0f;
    }
}
