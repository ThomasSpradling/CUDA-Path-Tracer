#pragma once

#include "MathUtils.h"
#include "glm/fwd.hpp"
#include "utils.h"
#include <glm/glm.hpp>

enum BSDFSampleType : uint8_t {
    InvalidSample   = 0,

    Diffuse         = 1 << 0,
    Glossy          = 1 << 1,
    Specular        = 1 << 2,

    Reflection      = 1 << 4,
    Transmission    = 1 << 5,
};

struct BSDFSample {
    glm::vec3 w_in {};
    glm::vec3 bsdf {};
    float pdf = 0;
    uint8_t type = InvalidSample;
};

__device__ inline float Fresnel(float cos_in, float eta) {
    if (cos_in < 0.0f) {
        eta = 1.0f / eta;
        cos_in = -cos_in;
    }

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

struct Material {
    enum Type {
        Diffuse,
        Dielectric,
        Mirror,

        Light,
    };

    Type type = Type::Diffuse;
    glm::vec3 base_color = glm::vec3(.9f);
    float metallic = 0.f;
    float roughness = 1.f;
    float ior = 1.5f;

    // int baseColorMapId = NullTextureId;
    // int metallicMapId = NullTextureId;
    // int roughnessMapId = NullTextureId;
    // int normalMapId = NullTextureId;

    /**
     * @brief Computes the bidirectional scattering distribution function (BSDF)
     * for this material.
     */
    __device__ glm::vec3 BSDF(const glm::vec3 &normal, const glm::vec3 &w_in, const glm::vec3 &w_out) {
        if (type == Type::Diffuse) {
            return DiffuseBSDF();
        }

        if (type == Type::Dielectric) {
            return DielectricBSDF(normal, w_in, w_out);
        }
    }

    /**
     * @brief Computes the probability density over the rendering equation.
     */
    __device__ float PDF(const glm::vec3 &normal, const glm::vec3 &w_in, const glm::vec3 &w_out) {
        if (type == Type::Diffuse) {
            return DiffusePDF(normal, w_in);
        }

        if (type == Type::Dielectric) {
            return 1.0f;
        }
    }

    /**
     * @brief Samples the rendering equation assuming uniform random variable `uniform`
     */
    __device__ BSDFSample Sample(const glm::vec3 &normal, const glm::vec3 &w_out, const glm::vec3 &uniform) {
        if (type == Type::Diffuse) {
            return DiffuseSample(normal, w_out, uniform);
        }

        if (type == Type::Dielectric) {
            return DielectricSample(normal, w_out, uniform);
        }

        if (type == Type::Mirror) {
            return MirrorSample(normal, w_out, uniform);
        }

        // if (type == Type::Light) {
        //     return LightSample(normal, w_out, uniform);
        // }
    }

    //// Diffuse Material ////

    __device__ glm::vec3 DiffuseBSDF() {
        return base_color * c_INV_PI;
    }

    __device__ float DiffusePDF(const glm::vec3 &normal, const glm::vec3 &w_in) {
        return glm::max(0.0f, glm::dot(normal, w_in)) * c_INV_PI;
    }

    __device__ BSDFSample DiffuseSample(const glm::vec3 &normal, const glm::vec3 &w_out, const glm::vec3 &uniform) {
        // glm::vec3 w_in = Math::SampleUniformHemisphere(normal, uniform);
        glm::vec3 w_in = Math::SampleCosineHemisphere(normal, uniform);

        BSDFSample sample {};
        sample.bsdf = DiffuseBSDF();
        sample.pdf = DiffusePDF(normal, w_in);
        sample.type = BSDFSampleType::Diffuse | BSDFSampleType::Reflection;
        sample.w_in = w_in;

        return sample;
    }

    //// Light Implementations ////

    __device__ BSDFSample LightSample(const glm::vec3 &normal, const glm::vec3 &w_out, const glm::vec3 &uniform) {
        BSDFSample sample {};
        sample.w_in = glm::vec3(0.0f);
        sample.bsdf = base_color;
        sample.pdf = 1.0f;
        return sample;
    }

    //// Specular Mirror Implementations ////
    __device__ BSDFSample MirrorSample(const glm::vec3 &normal, const glm::vec3 &w_out, const glm::vec3 &uniform) {
        glm::vec3 w_in = glm::reflect(-w_out, normal);

        float cos_in = fabsf(glm::dot(normal, w_in));

        BSDFSample sample {};
        sample.pdf = 1.0f;
        sample.w_in = w_in;
        sample.type = BSDFSampleType::Specular | BSDFSampleType::Reflection;
        sample.bsdf = base_color / cos_in;

        return sample;
    }

    //// Dielectric Implementations ////

    __device__ glm::vec3 DielectricBSDF(const glm::vec3 &normal, const glm::vec3 &w_in, const glm::vec3 &w_out) {
        // Ignore for now
    }

    __device__ float DielectricPDF(const glm::vec3 &normal, const glm::vec3 &w_in) {
        return 1.0f;
    }

    __device__ BSDFSample DielectricSample(
        const glm::vec3 &normal,
        const glm::vec3 &w_out,
        const glm::vec3 &uniform
    ) {
        BSDFSample sample;
        float eta = 1.0f / ior;
        float cos_out = glm::dot(normal, w_out);

        float reflectance = Fresnel(cos_out, eta);
        float transmittance = 1.0f - reflectance;
        float total_prob = reflectance + transmittance;

        if (uniform.x < reflectance / total_prob) {
            glm::vec3 w_in = Math::Reflect(w_out, normal);
            float cos_in = fabsf(glm::dot(normal, w_in));

            sample.type = Specular | Reflection;
            sample.w_in = w_in;
            sample.bsdf = base_color * reflectance / cos_in;
            sample.pdf = reflectance / total_prob;
            return sample;
        }
        else {
            glm::vec3 w_in;
            if (!Math::Refract(w_out, normal, eta, w_in)) {
                sample.type = InvalidSample;
                return sample;
            }

            float cos_in = fabsf(glm::dot(normal, w_in));

            sample.type = Specular | Transmission;
            sample.w_in = w_in;
            sample.bsdf = base_color * transmittance * eta * eta / cos_in;
            sample.pdf = transmittance / total_prob;
            return sample;
        }
    }
};
