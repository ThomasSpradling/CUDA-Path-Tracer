#pragma once

#include <algorithm>
#include <glm/glm.hpp>

#include "cl_happy.h"

enum class ColorSpace {
    RGB,
    sRGB,
    XYZ
};

using Spectrum = glm::vec3;

__host__ __device__ float SRGBChannelToLinear(float c);
__host__ __device__ glm::vec3 SRGBToLinear(const glm::vec3 &srgb);

// __host__ __device__ __forceinline__ glm::vec3 NormalizeRGB(const glm::ivec3 &rgb) {
//     return glm::vec3(rgb.r / 255.0f, rgb.g / 255.0f, rgb.b / 255.0f);
// }
