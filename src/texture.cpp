#include "texture.h"
#include "kernel/device_scene.h"
#include "utils/cuda_utils.h"
#include "utils/exception.h"
#include <channel_descriptor.h>
#include <vector_functions.h>

template<typename T>
std::pair<cudaTextureObject_t, cudaArray_t> _UploadImage(Image<T> &image) {
    cudaChannelFormatDesc channel_desc {};
    if constexpr (std::is_same_v<T, float>) {
        channel_desc = cudaCreateChannelDesc<float>();
    } else if constexpr (std::is_same_v<T, glm::vec3>) {
        channel_desc = cudaCreateChannelDesc<float4>();
    }

    const uint32_t width = image.Width();
    const uint32_t height = image.Height();

    if (width == 0 || height == 0) {
        std::cerr << "Cannot upload invalid image!";
        return {};
    }

    cudaArray_t d_array;
    cudaTextureObject_t d_texture;

    CUDA_CHECK(cudaMallocArray(&d_array, &channel_desc, image.Width(), image.Height()));

    if constexpr (std::is_same_v<T, float>) {
        CUDA_CHECK(cudaMemcpy2DToArray(
            d_array, 0, 0,
            image.Data(),
            width * sizeof(float),
            width * sizeof(float),
            height,
            cudaMemcpyHostToDevice
        ));
    } else {
        std::vector<float4> host4(width * height);
        glm::vec3 *hrgb = image.Data();
        for (int i = 0; i < width*height; ++i) {
            auto &c = hrgb[i];
            host4[i] = make_float4(c.r, c.g, c.b, 1.0f);
        }
        CUDA_CHECK(cudaMemcpy2DToArray(
            d_array, 0, 0,
            host4.data(),
            width * sizeof(float4),
            width * sizeof(float4),
            height,
            cudaMemcpyHostToDevice
        ));
    }

    cudaResourceDesc resource_desc {};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = d_array;

    cudaTextureDesc texture_desc {};
    texture_desc.addressMode[0] = cudaAddressModeWrap;
    texture_desc.addressMode[1] = cudaAddressModeWrap;
    texture_desc.filterMode = cudaFilterModeLinear;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = 1;

    if (image.GetColorSpace() == ColorSpace::sRGB) {
        texture_desc.sRGB = 1;
    }

    CUDA_CHECK(cudaCreateTextureObject(&d_texture, &resource_desc, &texture_desc, nullptr));

    return std::make_pair(d_texture, d_array);
}

void TexturePool::UpdateDevice(DeviceScene &scene) {
    if (!dirty) return;
    
    scene.texture_pool.textures1 = nullptr;
    scene.texture_pool.textures3 = nullptr;
    scene.texture_pool.images1 = nullptr;
    scene.texture_pool.images3 = nullptr;

    image_arrays.clear();
    image_arrays.reserve(images1.size() + images3.size());

    scene.texture_pool.texture1_count = CopyToDevice(scene.texture_pool.textures1, textures1);
    scene.texture_pool.texture3_count = CopyToDevice(scene.texture_pool.textures3, textures3);

    image_textures1.resize(images1.size());
    image_textures3.resize(images3.size());

    for (int i = 0; i < images1.size(); ++i) {
        cudaArray_t arr1;
        std::tie(image_textures1[i], arr1) = _UploadImage(images1[i]);
        image_arrays.push_back(arr1);
    }

    for (int i = 0; i < images3.size(); ++i) {
        cudaArray_t arr3;
        std::tie(image_textures3[i], arr3) = _UploadImage(images3[i]);
        image_arrays.push_back(arr3);
    }

    scene.texture_pool.image1_count = CopyToDevice(scene.texture_pool.images1, image_textures1);
    scene.texture_pool.image3_count = CopyToDevice(scene.texture_pool.images3, image_textures3);

    dirty = false;
}

void TexturePool::FreeDevice(DeviceScene &scene) {
    for (auto &tex : image_textures1)
        cudaDestroyTextureObject(tex);
    for (auto &tex : image_textures3)
        cudaDestroyTextureObject(tex);

    cudaFree(scene.texture_pool.images1);
    cudaFree(scene.texture_pool.images3);

    for (auto &arr : image_arrays)
        cudaFreeArray(arr);
    image_arrays.clear();

    cudaFree(scene.texture_pool.textures1);
    cudaFree(scene.texture_pool.textures3);

    image_textures1.clear();
    image_textures3.clear();
}
