#pragma once

#include "device_types.h"
#include <glm/glm.hpp>

template<typename T>
__device__ __forceinline__ T SampleConstantTexture(ConstantTexture<T> &texture, float u, float v) {
    return texture.value;
}

template<typename T>
__device__ __forceinline__ T SampleCheckerboardTexture(CheckerboardTexture<T> &texture, float u, float v) {
    float s = u * texture.scale.x + texture.offset.x;
    float t = v * texture.scale.y + texture.offset.y;

    int i = static_cast<int>(floorf(s));
    int j = static_cast<int>(floorf(t));
    return ((i + j) & 1) ? texture.color1 : texture.color0;
}

template<typename T>
__device__ __forceinline__ T SampleImageTexture(ImageTexture<T> &texture, cudaTextureObject_t image, float u, float v) {
#ifdef __CUDACC__
    float s = u * texture.scale.x + texture.offset.x;
    float t = v * texture.scale.y + texture.offset.y;

    if constexpr (std::is_same_v<T, glm::vec3>) {
        float4 color = tex2D<float4>(image, s, t);
        return glm::vec3(color.x, color.y, color.z);
    } else {
        float color;
        tex2D<float>(&color, image, s, t);
        return color;
    }
#endif

    // Should never reach
    return T(0.0f);
}

template<typename T>
__device__ T SampleTexture(const DeviceTexturePool &texture_pool, int texture_id, float u, float v) {
    if (texture_id < 0)
        return T(0.0f);

    Texture<T> texture;

    // Grab the texture based on template
    if constexpr (std::is_same_v<T, float>) {
        if (texture_id >= texture_pool.texture1_count) {
            return T(0.0f);
        }
        texture = texture_pool.textures1[texture_id];
    } else {
        if (texture_id >= texture_pool.texture3_count) {
            return T(0.0f);
        }
        texture = texture_pool.textures3[texture_id];
    }
    
    switch (texture.type) {
        case TextureType::Constant:
            return SampleConstantTexture(texture.constant, u, v);
        case TextureType::Checkerboard:
            return SampleCheckerboardTexture(texture.checkerboard, u, v);
        case TextureType::Image: {
            cudaTextureObject_t image;
            int image_id = texture.image.image_id;
            if (image_id == -1) {
                return T(0.0f);
            }

            // Grab the image based on template
            if constexpr (std::is_same_v<T, float>) {
                if (image_id >= texture_pool.image1_count) {
                    return T(0.0f);
                }
                image = texture_pool.images1[texture.image.image_id];
            } else {
                if (image_id >= texture_pool.image3_count) {
                    return T(0.0f);
                }
                image = texture_pool.images3[texture.image.image_id];
            }
            return SampleImageTexture(texture.image, image, u, v);
        }
    }
};

template<typename T>
__device__ T SampleTexture(const DeviceTexturePool &texture_pool, int texture_id, glm::vec2 uv) {
    return SampleTexture<T>(texture_pool, texture_id, uv.x, uv.y);
}
