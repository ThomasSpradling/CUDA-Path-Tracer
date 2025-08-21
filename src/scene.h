#pragma once

#include "image.h"
#include "kernel/device_types.h"
#include "kernel/device_scene.h"
#include "discrete_sampler.h"
#include "camera.h"
#include "bvh.h"
#include "kernel/integrators/integrator.h"
#include "texture.h"
#include "mesh/gltf_model.h"
#include "utils/device_buffer.h"
#include "utils/utils.h"

#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

class Scene {
public:
    Scene(const std::string &filename);
    ~Scene();

    inline glm::ivec2 GetResolution() const { return m_camera->m_film.m_resolution; }
    inline void SetResolution(glm::ivec2 resolution) { m_camera->m_film.m_resolution = resolution; }

    bool UpdateDevice();
    const DeviceScene &GetDeviceScene() const { return m_device_scene; };

    int MaxIterations() const { return m_max_iterations; }

    Camera &GetCamera() { return *m_camera; }
    const Camera &GetCamera() const { return *m_camera; }

    const std::vector<glm::vec3> &Positions() const { return m_positions; }
    const std::vector<uint32_t> &Indices() const { return m_indices; }
private:
    bool m_dirty = false;

    IntegratorType m_integrator;
    int m_max_depth;
    int m_max_iterations;

    SamplerType m_sampler;
    
    // Vertex data
    std::vector<glm::vec3> m_positions;
    std::vector<glm::vec3> m_normals;
    std::vector<glm::vec2> m_texcoords;    
    std::vector<uint32_t> m_indices;

    std::vector<GeometryInstance> m_geometries;

    std::vector<DiscreteSampler1D> m_triangle_samplers;

    // BVH Data
    std::unique_ptr<TLAS> m_bvh;

    std::vector<Material> m_materials;
    std::unique_ptr<TexturePool> m_texture_pool {};

    std::vector<AreaLight> m_lights;
    DiscreteSampler1D m_light_sampler;
    float m_total_light_power = 0.0f;

    std::unique_ptr<Camera> m_camera;
    DeviceScene m_device_scene;

    fs::path m_scene_dir;
    uint32_t m_current_blas_index = 0;
private:
    template<typename T>
    int ParseTexture(const json &parsed_texture, int channel = 0, bool is_albedo_texure = false) {
        int texture_id;
        if constexpr (std::is_same_v<T, float>) {
            texture_id = m_texture_pool->textures1.size();
        } else if constexpr (std::is_same_v<T, glm::vec3>) {
            texture_id = m_texture_pool->textures3.size();
        }

        if (parsed_texture.contains("PATH")) {
            const auto &path = parsed_texture["PATH"];
            if (!path.is_string()) {
                std::cerr << std::format("Scene Warning: Invalid path: '%s'!", path.get<std::string>()) << std::endl;

                // The 0-th texture will always just be a constant white texture
                return 0;
            }

            if (auto resolved = Utils::ResolvePath(path, m_scene_dir)) {
                Texture<T> texture;

                texture.type = TextureType::Image;

                if constexpr (std::is_same_v<T, float>) {
                    texture.image.image_id = m_texture_pool->images1.size();

                    m_texture_pool->images1.emplace_back();

                    Image1 &image = m_texture_pool->images1.back();
                    LoadImageFromFile(m_texture_pool->images1.back(), *resolved, channel);

                    texture.image.width = image.Width();
                    texture.image.height = image.Height();
                } else if constexpr (std::is_same_v<T, glm::vec3>) {
                    texture.image.image_id = m_texture_pool->images3.size();

                    m_texture_pool->images3.emplace_back();

                    Image3 &image = m_texture_pool->images3.back();
                    LoadImageFromFile(image, *resolved);

                    if (is_albedo_texure) {
                        image.SetColorSpace(ColorSpace::sRGB);
                    }

                    texture.image.width = image.Width();
                    texture.image.height = image.Height();
                }

                texture.image.offset = glm::vec2(0.0f);
                texture.image.scale = glm::vec2(1.0f);

                if constexpr (std::is_same_v<T, float>) {
                    m_texture_pool->textures1.push_back(texture);
                } else if constexpr (std::is_same_v<T, glm::vec3>) {
                    m_texture_pool->textures3.push_back(texture);
                }
                return texture_id;
            }
        }

        if (channel > 0) {
            std::cerr << "Nontrivial channels are incompatible with non-image textures!" << std::endl;
        }

        if constexpr (std::is_same_v<T, float>) {
            if (parsed_texture.contains("VALUE")) {
                const auto &c = parsed_texture["VALUE"];
                if (!c.is_number()) {
                    std::cerr << "Scene Warning: Expected value to be float!" << std::endl;
                    return 0;
                }

                Texture<T> texture;

                texture.type = TextureType::Constant;
                texture.constant.value = static_cast<float>(c);
                
                m_texture_pool->textures1.push_back(texture);

                return texture_id;
            }
        } else if constexpr (std::is_same_v<T, glm::vec3>) {
            if (parsed_texture.contains("RGB")) {
                const auto &c = parsed_texture["RGB"];
                if (!c.is_array() || c.size() != 3) {
                    std::cerr << "Scene Warning: Expected RGB color to have 3 components!" << std::endl;
                    return 0;
                }

                Texture<T> texture;

                texture.type = TextureType::Constant;
                texture.constant.value = Utils::ParseVector(c, glm::vec3(0.0f));
                
                m_texture_pool->textures3.push_back(texture);

                return texture_id;
            }
        }

        std::cerr << "Invalid texture.\n";
        return 0;
    }

    template<typename T>
    uint32_t LoadTextureOrZero(const json &mat, const std::string &key, int channel = 0, bool is_albedo_texure = false) {
        return mat.contains(key)
            ? ParseTexture<T>(mat[key], channel, is_albedo_texure)
            : 0;
    }

    void ComputeLightData();

    void ParseIntegrator(const json &data);
    void ParseCamera(const json &data);
    void InitDefaults();
    bool ParseMaterial(const json &material);
    std::unordered_map<std::string, int> ParseMaterials(const json &data);
    
    std::vector<GeometryInstance> ParseObject(const json &parsed_object, const std::unordered_map<std::string, int> &material_ids);
    void ParseObjects(const json &data, const std::unordered_map<std::string, int> &material_ids);
    void LoadScene(const std::string &filename);
};
