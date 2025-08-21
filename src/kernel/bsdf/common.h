#pragma once

#include <glm/glm.hpp>
#include "../../math/math.h"
#include "../texture.h"
#include "../intersection.h"
#include "../../math/sampling.h"
#include "../sampler.h"
#include "../../math/color.h"
#include "../../math/geometry.h"

enum BSDF_SampleType : uint8_t {
    Invalid         = 0,

    Diffuse         = 1 << 0,
    Glossy          = 1 << 1,
    Specular        = 1 << 2,

    Reflection      = 1 << 4,
    Transmission    = 1 << 5,
};

struct BSDF_Sample {
    uint8_t type = BSDF_SampleType::Invalid;
    float pdf;
    glm::vec3 bsdf;
    glm::vec3 w_in;
};

/**
 * @brief The GGX micro-normal density function describing density
 * of surface normals per unit solid angle.
 *
 * For more info on this any below see
 *      Walter et. al, "Microfacet Models for Refraction through Rough Surfaces" (2007)
 */
__device__ inline float TrowbridgeReitz(const glm::vec3 &normal, const glm::vec3 &micro_normal, float alpha) {
    float cos_in = glm::dot(normal, micro_normal);
    if (cos_in <= 0.0f)
        return 0.0f;

    float a2 = alpha * alpha;

    float denom_term = fmaf((a2 - 1.0f), cos_in * cos_in, 1.0f);
    float denom = denom_term * denom_term + 1e-16f;

    float result = (a2 * Math::INV_PI) / denom;
    return isfinite(result) ? result : 0.0f;
}

/**
 * @brief Smith G1 masking term. Its the distribution of normals that are
 * actually visible according to the Towbridge Reitz distribution.
 */
__device__ inline float SmithG1(const glm::vec3 &normal, const glm::vec3 &vec, float roughness) {
    float cos_in = glm::dot(normal, vec);
    if (cos_in <= 0.0f)
        return 0.0f;

    float a2 = roughness * roughness;
    float denom = cos_in + sqrtf(a2 + (1.0f - a2) * cos_in * cos_in);
    return 2 * cos_in / denom;
}

__device__ inline float SmithG(const glm::vec3 &normal, const glm::vec3 &w_in, const glm::vec3 &w_out, float roughness) {
    return SmithG1(normal, w_in, roughness) * SmithG1(normal, w_out, roughness);
}

__device__ inline glm::vec3 HalfVectorReflection(const glm::vec3 &w_in, const glm::vec3 &w_out) {
    return glm::normalize(w_in + w_out);
}

__device__ inline glm::vec3 HalfVectorRefraction(const glm::vec3 &w_in, const glm::vec3 &w_out, float eta) {
    return -glm::normalize(eta * w_in + w_out);
}

/**
 * @brief Fraction of reflected unpolarized light
 */
__device__ inline float Fresnel(float cos_in, float eta) {
    float sin2_in = 1 - cos_in * cos_in;
    float sin2_t = sin2_in * eta * eta;
    if (sin2_t >= 1.0f) {
        // Total internal reflection
        return 1.0f;
    }
    float cos_t = fmaxf(0.0f, sqrtf(1 - sin2_t));

    float parallel = (cos_in - eta * cos_t) / (cos_in + eta * cos_t);
    float perp = (eta * cos_in - cos_t) / (eta * cos_in + cos_t);
    return (parallel * parallel + perp * perp) / 2.0f;
}

/**
 * @brief Schlick approximation of frensel equations
 */
__device__ inline float FresnelSchlick(float cos_in, float f0) {
    return f0 + (1.0f - f0) * Math::Pow5(1.0f - cos_in);
}
__device__ inline glm::vec3 FresnelSchlick(float cos_in, glm::vec3 f0) {
    return f0 + (1.0f - f0) * Math::Pow5(1.0f - cos_in);
}

