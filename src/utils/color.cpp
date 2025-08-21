#include "color.h"

__host__ __device__ float SRGBChannelToLinear(float c) {
    c = glm::clamp(c, 0.0f, 1.0f);
    if (c <= 0.04045f) {
        return c / 12.92f;
    } else {
        return powf((c + 0.055f) / 1.055f, 2.4f);
    }
}

__host__ __device__ glm::vec3 SRGBToLinear(const glm::vec3 &srgb) {
    return glm::vec3(
        SRGBChannelToLinear(srgb.r),
        SRGBChannelToLinear(srgb.g),
        SRGBChannelToLinear(srgb.b)
    );
}
