#pragma once

#include "Intersections.h"
#include <thrust/random.h>
#include <glm/glm.hpp>

#include "Materials/Lambertian.h"
#include "Materials/MetallicRoughness.h"
#include "Materials/PerfectDielectric.h"
#include "Materials/PerfectMirror.h"
#include "Materials/RoughDielectric.h"
#include "exception.h"

struct Light {
    glm::vec3 base_color;
    float emittance;
    int light_id = -1;
};

struct Material {
    enum class Type : uint8_t {
        Lambertian,
        Light,

        Mirror,
        Dielectric,
        Metallic_Roughness,
        RoughDielectric
    };

    Type type;

    Material() {};
    ~Material() {
        switch (type) {
            case Type::Lambertian:
                mat.lambert.~Lambertian();
                break;
            case Type::Light:
                break;
            case Type::Mirror:
                mat.mirror.~PerfectMirror();
                break;
            case Type::Dielectric:
                mat.dielectric.~PerfectDielectric();
                break;
            case Type::Metallic_Roughness:
                mat.lambert.~Lambertian();
                break;
            case Type::RoughDielectric:
                mat.rough_dielectric.~RoughDielectric();
                break;
        }
    }

    union U {
        Lambertian lambert;
        Light light;
        PerfectMirror mirror;
        PerfectDielectric dielectric;
        MetallicRoughness metallic_roughness;
        RoughDielectric rough_dielectric;

        __host__ __device__ U() {}
        __host__ __device__ ~U() {}
    } mat;

    __device__ BSDFSample Sample(const Intersection &intersection, const glm::vec3 &w_out, thrust::default_random_engine &rng) {
        if (type == Type::Light) {
            return {};
        }

        if (type == Type::Lambertian) {
            return mat.lambert.Sample(intersection, w_out, rng);
        }

        if (type == Type::Mirror) {
            return mat.mirror.Sample(intersection, w_out, rng);
        }

        if (type == Type::Dielectric) {
            return mat.dielectric.Sample(intersection, w_out, rng);
        }

        if (type == Type::Metallic_Roughness) {
            return mat.metallic_roughness.Sample(intersection, w_out, rng);
        }

        if (type == Type::RoughDielectric) {
            return mat.rough_dielectric.Sample(intersection, w_out, rng);
        }
    }

    __device__ glm::vec3 BSDF(const Intersection &intersection, const glm::vec3 &w_out, const glm::vec3 &w_in) {
        if (type == Type::Light) {
            return {};
        }

        if (type == Type::Lambertian) {
            return mat.lambert.BSDF(intersection);
        }

        if (type == Type::Mirror) {
            return {};
        }

        if (type == Type::Dielectric) {
            return {};
        }

        if (type == Type::Metallic_Roughness) {
            return mat.metallic_roughness.BSDF(intersection, w_out, w_in);
        }

        if (type == Type::RoughDielectric) {
            return {};
        }
    }

    __device__ float PDF(const Intersection &intersection, const glm::vec3 &w_out, const glm::vec3 &w_in) {
        if (type == Type::Light) {
            return {};
        }

        if (type == Type::Lambertian) {
            return mat.lambert.PDF(intersection.normal, w_in);
        }

        if (type == Type::Mirror) {
            return {};
        }

        if (type == Type::Dielectric) {
            return {};
        }

        if (type == Type::Metallic_Roughness) {
            return mat.metallic_roughness.PDF(intersection, w_out, w_in);
        }

        if (type == Type::RoughDielectric) {
            return {};
        }
    }
};
