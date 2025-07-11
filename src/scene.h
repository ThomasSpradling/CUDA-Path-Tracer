#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "Image.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include "geometry.h"

#include <cuda_runtime.h>
#include "GLTFModel.h"
#include "json.hpp"
#include "material.h"

// struct Material {
//     glm::vec3 color;
//     struct {
//         float exponent;
//         glm::vec3 color;
//     } specular;

//     bool has_reflective = false;
//     bool has_refractive = false;
//     float index_of_refraction;
//     float emittance;
// };

struct Camera {
    struct Default {
        glm::vec3 front { 0.0f, 0.0f, -1.0f };
        glm::vec3 look_at {};
        glm::vec3 position { 0.0f, 0.0f, -1.0f };
    } default_params;

    glm::ivec2 resolution { 0, 0 };
    glm::vec3 position {};
    glm::vec3 look_at { 0.0f, 0.0f, -1.0f };

    glm::vec3 front { 0.0f, 0.0f, -1.0f };
    glm::vec3 up { 0.0f, 1.0f, 0.0f };
    glm::vec3 right { 1.0f, 0.0f, 0.0f };

    float fovy;
    glm::vec2 pixel_length;
};

struct RenderState {
    Camera camera {};
    uint32_t iterations;
    int trace_depth;
    Image3 image;
    std::string output_name;
};

struct PathSegment {
    Ray ray;
    glm::vec3 throughput;
    glm::vec3 color_accum;
    int pixel_index;
    int remaining_bounces;

    bool hit = false;
};

class Scene {
public:
    Scene(const std::string &filename);
    ~Scene() = default;

    RenderState &State() { return m_state; }
    const RenderState &State() const { return m_state; }

    std::vector<Geometry> &Geometries() { return m_geometry; }
    const std::vector<Geometry> &Geometries() const { return m_geometry; }

    std::vector<Material> &Materials() { return m_materials; }
    const std::vector<Material> &Materials() const { return m_materials; }

    int VertexCount() const { return m_vertices.size(); }
    const std::vector<MeshVertex> &Vertices() const { return m_vertices; }
    
    int IndexCount() const { return m_indices.size(); }
    const std::vector<uint32_t> &Indices() const { return m_indices; }
private:
    std::vector<Geometry> m_geometry;
    std::vector<Material> m_materials;
    RenderState m_state;

    std::vector<TriangleMesh> m_mesh_buffer;
    std::vector<MeshVertex> m_vertices;
    std::vector<uint32_t> m_indices;

    fs::path m_scene_dir;
private:
    void LoadFromJSON(const std::string &filename);
    Texture<glm::vec3> LoadTexture3D(const nlohmann::json &parsed_texture);

    template<typename T>
    void LoadTexture(Texture<T> &texture, const nlohmann::json &parsed_texture) {
        if (parsed_texture.contains("PATH")) {
            const auto &path = parsed_texture["PATH"];
            if (!path.is_string()) {
                std::cerr << "Scene Warning: Invalid path: '" << path << "'!" << std::endl;
            }

            if (auto resolved = ResolvePath(path, m_scene_dir)) {
                texture.type = TextureType::ImageTexture;
                texture.tex.image_texture.LoadTexture(resolved->string());
                texture.tex.image_texture.m_offset = glm::vec2(0.0f);
                texture.tex.image_texture.m_scale = glm::vec2(1.0f);
                return;
            }
        }

        if constexpr (std::is_same_v<T, glm::vec3>) {
            if (parsed_texture.contains("RGB")) {
                const auto &c = parsed_texture["RGB"];
                if (!c.is_array() || c.size() != 3) {
                    std::cerr << "Scene Warning: Expected RGB color to have 3 components!" << std::endl;
                    texture = glm::vec3(0.0f);
                }
    
                texture = glm::vec3(c[0], c[1], c[2]);
                return;
            }
        } else if constexpr (std::is_same_v<T, float>) {
            if (parsed_texture.contains("VALUE")) {
                const auto &c = parsed_texture["VALUE"];
                if (!c.is_number()) {
                    std::cerr << "Scene Warning: Expected value to be float!" << std::endl;
                    texture = 0.0f;
                }
    
                texture = c;
                return;
            }
        }

        std::cerr << "Invalid texture: \n";

        texture = T(0.0f);
        return;
    }
};
