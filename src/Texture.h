#pragma once

#include "Image.h"
#include "exception.h"
#include "image.h"
#include <iostream>
#include <string>
#include <texture_indirect_functions.h>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>

template <typename T>
struct TextureRef {
    uint32_t texture_id;
};

template <typename T>
struct ImageTexture {
    void Destroy () { 
        if (md_array)
            cudaDestroyTextureObject(md_texture);
        if (md_texture)
            cudaFreeArray(md_array);
    }

    void SetScale(const glm::vec2 &scale) { m_scale = scale; }
    void SetOffset(const glm::vec2 &offset) { m_offset = offset; }

    void LoadTexture(const std::string &filename) {
        Image<T> image;
        if constexpr (std::is_same_v<T, float>) {
            Load1DImageFromFile(image, filename);
        } else if constexpr (std::is_same_v<T, glm::vec3>) {
            Load3DImageFromFile(image, filename);
        }

        m_width = image.Width();
        m_height = image.Height();

        T *data = image.Data();

        cudaChannelFormatDesc channel_desc {};
        if constexpr (std::is_same_v<T, float>) {
            channel_desc = cudaCreateChannelDesc<float>();
        } else if constexpr (std::is_same_v<T, glm::vec3>) {
            channel_desc = cudaCreateChannelDesc<float4>();
        } else {
            static_assert(false, "Invalid T for Texture<T>.");
        }

        CUDA_CHECK(cudaMallocArray(&md_array, &channel_desc, m_width, m_height));

        if constexpr (std::is_same_v<T, float>) {
            CUDA_CHECK(cudaMemcpy2DToArray(
                md_array, 0, 0,
                data,
                m_width * sizeof(float),
                m_width * sizeof(float),
                m_height,
                cudaMemcpyHostToDevice
            ));
        } else {
            std::vector<float4> host4(m_width * m_height);
            glm::vec3 *hrgb = image.Data();
            for (int i = 0; i < m_width*m_height; ++i) {
                auto &c = hrgb[i];
                host4[i] = make_float4(c.r, c.g, c.b, 0.0f);
            }
            CUDA_CHECK(cudaMemcpy2DToArray(
                md_array, 0, 0,
                host4.data(),
                m_width * sizeof(float4),
                m_width * sizeof(float4),
                m_height,
                cudaMemcpyHostToDevice
            ));
        }
    
        cudaResourceDesc resource_desc {};
        resource_desc.resType = cudaResourceTypeArray;
        resource_desc.res.array.array = md_array;

        cudaTextureDesc texture_desc {};
        texture_desc.addressMode[0] = cudaAddressModeWrap;
        texture_desc.addressMode[1] = cudaAddressModeWrap;
        texture_desc.filterMode = cudaFilterModeLinear;
        texture_desc.readMode = cudaReadModeElementType;
        texture_desc.normalizedCoords = 1;

        CUDA_CHECK(cudaCreateTextureObject(&md_texture, &resource_desc, &texture_desc, nullptr));
    }

    __device__
    T Get(float u, float v) const {
#ifdef __CUDACC__
        float s = u * m_scale.x + m_offset.x;
        float t = v * m_scale.y + m_offset.y;

        if constexpr (std::is_same_v<T, glm::vec3>) {
            float4 color = tex2D<float4>(md_texture, s, t);
            return glm::vec3(color.x, color.y, color.z);
        } else {
            float color;
            tex2D(&color, md_texture, s, t);
            return color;
        }
#endif
    }

    int m_width;
    int m_height;
    
    glm::vec2 m_scale;
    glm::vec2 m_offset;

    cudaArray_t md_array;
    cudaTextureObject_t md_texture;
};

template <typename T>
struct ConstantTexture {
    T value;

    __device__
    T Get(float u, float v) const {
        return value;
    }
};

template <typename T>
struct CheckerboardTexture {
    T color0, color1;
    glm::vec2 scale;
    glm::vec2 offset;

    __device__
    T Get(float u, float v) const {
        float s = u * scale.x + offset.x;
        float t = v * scale.y + offset.y;

        int i = static_cast<int>(floorf(s));
        int j = static_cast<int>(floorf(t));
        return ((i + j) & 1) ? color1 : color0;
    }
};

enum class TextureType : uint8_t {
    ImageTexture,
    ConstantTexture,
    CheckerboardTexture
};

template<typename T>
struct Texture {
    TextureType type;

    Texture() {};
    ~Texture() {
        switch (type) {
            case TextureType::ImageTexture:
                tex.image_texture.Destroy();
                break;
            case TextureType::ConstantTexture:
                tex.constant_texture.~ConstantTexture<T>();
                break;
            case TextureType::CheckerboardTexture:
                tex.checker_texture.~CheckerboardTexture<T>();
                break;
        }
    }

     __host__
    Texture(const T &value)
      : type(TextureType::ConstantTexture)
    {
        tex.constant_texture.value = value;
    }

    __host__
    void operator=(const T &value) {
        type = TextureType::ConstantTexture;
        tex.constant_texture.value = value;
    }

    // void EmplaceConstant(const glm::vec3 &value);

    union U {
        ImageTexture<T> image_texture;
        ConstantTexture<T> constant_texture;
        CheckerboardTexture<T> checker_texture;
    } tex;

    __device__
    T Get(float u, float v) const {
        switch (type) {
        case TextureType::ImageTexture:
            return tex.image_texture.Get(u, v);
        case TextureType::ConstantTexture:
            return tex.constant_texture.Get(u, v);
        case TextureType::CheckerboardTexture:
            return tex.checker_texture.Get(u, v);
        }
    }

    __device__
    T Get(const glm::vec2 &uv) const {
        return Get(uv.x, uv.y);
    }
};
