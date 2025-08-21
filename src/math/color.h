#pragma once

#include <glm/glm.hpp>

namespace Math {

    __host__ __device__ __forceinline__ float Luminance(const glm::vec3 &color) {
        return color.x * 0.212671f + color.y * 0.715160f + color.z * 0.072169f;
    }

    __host__ __device__ __forceinline__ glm::vec3 NormalizeRGB(int r, int g, int b) {
        const float inv255 = 1.0f / 255.0f;
        return glm::vec3(r, g, b) * inv255;
    }

    // Compute XYZ by fitting to matching functions to CIE 1931 standard. An approximation
    // used here is found at
    // Wyman et al. "Simple Analytic Approximations to the CIE XYZ Color Matching Functions" (2013)
    __host__ __device__ __forceinline__ glm::vec3 WavelengthToXYZ(float wavelength) {
        glm::vec3 result;

        // X(lambda)
        {
            float t1 = (wavelength - 442.0f) * ((wavelength < 442.0f) ? 0.0624f : 0.0374f);
            float t2 = (wavelength - 599.8f) * ((wavelength < 599.8f) ? 0.0264f : 0.0323f);
            float t3 = (wavelength - 501.1f) * ((wavelength < 501.1f) ? 0.0490f : 0.0382f);
            result.x = 0.362f * expf(-0.5f*t1*t1)
                + 1.056f * expf(-0.5f*t2*t2)
                - 0.065f * expf(-0.5f*t3*t3);
        }

        // Y(lambda)
        {
            float t1 = (wavelength - 568.8f) * ((wavelength < 568.8f) ? 0.0213f : 0.0247f);
            float t2 = (wavelength - 530.9f) * ((wavelength < 530.9f) ? 0.0613f : 0.0322f);
            result.y = 0.821f * expf(-0.5f * t1 * t1)
                + 0.286f * expf(-0.5f * t2 * t2);
        }

        // Z(lambda)
        {
            float t1 = (wavelength - 437.0f) * ((wavelength < 437.0f) ? 0.0845f : 0.0278f);
            float t2 = (wavelength - 459.0f) * ((wavelength < 459.0f) ? 0.0385f : 0.0725f);
            result.z = 1.217f * expf(-0.5f*t1*t1)
                + 0.681f * expf(-0.5f*t2*t2);
        }
        return result;
    }

    __host__ __device__ __forceinline__ glm::vec3 XYZ_To_LinearRGB(const glm::vec3 &xyz) {
        return {
             3.240479f * xyz.x - 1.537150f * xyz.y - 0.498535f * xyz.z,
            -0.969256f * xyz.x + 1.875991f * xyz.y + 0.041556f * xyz.z,
             0.055648f * xyz.x - 0.204043f * xyz.y + 1.057311f * xyz.z
        };
    }

    __host__ __device__ __forceinline__ float _InvGamma_sRGB(float c) {
        c = glm::clamp(c, 0.0f, 1.0f);
        return (c <= 0.04045f)
                ? c / 12.92f
                : powf((c + 0.055f) / 1.055f, 2.4f);
    }

    __host__ __device__ __forceinline__ glm::vec3 sRGB_To_LinearRGB(const glm::vec3 &srgb) {
        return glm::vec3(
            _InvGamma_sRGB(srgb.r),
            _InvGamma_sRGB(srgb.g),
            _InvGamma_sRGB(srgb.b)
        );
    }

    __host__ __device__ __forceinline__ float _Gamma_sRGB(float c) {
        c = glm::clamp(c, 0.0f, 1.0f);
        if (c <= 0.0031308f) 
            return 12.92f * c;
        else
            return 1.055f * powf(c, 1.0f/2.4f) - 0.055f;
    }

    __host__ __device__ __forceinline__ glm::vec3 LinearRGB_To_sRGB(const glm::vec3 &rgb) {
        return glm::vec3(
            _Gamma_sRGB(rgb.r),
            _Gamma_sRGB(rgb.g),
            _Gamma_sRGB(rgb.b)
        );
    }

}
