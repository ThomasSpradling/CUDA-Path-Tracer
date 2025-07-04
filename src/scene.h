#pragma once

#include <string>
#include <vector>
#include "Image.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <cuda_runtime.h>

enum class GeometryType {
    Sphere,
    Cube,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;

    __host__ __device__ inline glm::vec3 operator()(float time) const {
        return origin + (time - .0001f) * glm::normalize(direction);
    }
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
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;

    float has_reflective;
    float has_refractive;
    float index_of_refraction;
    float emittance;
};

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
    glm::vec3 color;
    int pixel_index;
    int remaining_bounces;
};

struct ShadableIntersection {
    float t;
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
private:
    std::vector<Geometry> m_geometry;
    std::vector<Material> m_materials;
    RenderState m_state;
private:
    void LoadFromJSON(const std::string &filename);
};
