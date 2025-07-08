#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "Image.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <cuda_runtime.h>
#include "GLTFModel.h"
#include "material.h"

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;

    __host__ __device__ inline glm::vec3 operator()(float time) const {
        return origin + (time - 1e-6f) * glm::normalize(direction);
    }

    __host__ __device__ static Ray Make(const glm::vec3 &origin, const glm::vec3 &direction) {
        return Ray{origin, direction};
    }

    __host__ __device__ static Ray MakeOffseted(const glm::vec3 &origin, const glm::vec3 &direction) {
        return Ray{origin + direction * 1e-3f, direction};
    }
};

struct AABB {
    glm::vec3 min{FLT_MAX};
    glm::vec3 max{-FLT_MAX};

    __host__ __device__ inline void AddPoint(const glm::vec3 &point) {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }
    __host__ __device__ inline void Union(const AABB &other) {
        min = glm::min(min, other.min);
        max = glm::max(max, other.max);
    }

    __host__ __device__ inline bool IsEmpty() const {
        return (min.x > max.x) || (min.y > max.y) || (min.z > max.z);
    }

    __host__ __device__ inline glm::vec3 Extent() const {
        return max - min;
    }

    __host__ __device__ inline float SurfaceArea() const {
        glm::vec3 d = max - min;
        d = glm::max(d, glm::vec3(0.0f));
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }

    __host__ __device__ inline float Volume() const {
        glm::vec3 d = max - min;
        d = glm::max(d, glm::vec3(0.0f));
        return d.x * d.y * d.z;
    }
};

enum class GeometryType {
    Sphere,
    Cube,
    GLTF_Primitive,
};

struct TriangleMesh {
    uint32_t first_index;
    uint32_t index_count;
    uint32_t first_vertex;
    uint32_t vertex_count;
};

struct Geometry {
    enum GeometryType type;
    int material_id;

    glm::vec3 translation {};
    glm::vec3 rotation {};
    glm::vec3 scale { 1.0f, 1.0f, 1.0f };

    glm::mat4 transform { 1.0f };
    glm::mat4 inv_transform { 1.0f };
    glm::mat4 inv_transpose { 1.0f };

    TriangleMesh mesh;

    int first_bvh_node = -1;
    int bvh_node_count = -1;

    int first_tri_index = -1;
    int tri_index_count = -1;
};

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
    Image image;
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

struct ShadableIntersection {
    float t;
    glm::vec3 pos;
    glm::vec3 normal;
    int material_id;
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
private:
    void LoadFromJSON(const std::string &filename);
};
