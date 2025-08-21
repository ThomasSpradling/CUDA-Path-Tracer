#pragma once

#include "image.h"
#include "kernel/device_types.h"

class DeviceScene;

struct TexturePool {
    std::vector<Texture<float>> textures1;
    std::vector<Texture<glm::vec3>> textures3;

    std::vector<Image<float>> images1;
    std::vector<Image<glm::vec3>> images3;

    std::vector<cudaArray_t> image_arrays;
    std::vector<cudaTextureObject_t> image_textures1;
    std::vector<cudaTextureObject_t> image_textures3;

    bool dirty = false;
    
    void UpdateDevice(DeviceScene &scene);
    void FreeDevice(DeviceScene &scene);
};
