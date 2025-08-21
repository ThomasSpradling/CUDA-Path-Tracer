#pragma once

#include <glm/glm.hpp>
#include <texture_types.h>
#include "../utils/color.h"
#include "../math/aabb.h"

// -- Integrator ------------------

enum class DebugVisualizeMode {
    None,
    GeometricNormals,
    Albedo,
    UV,
    Depth,
    Metallic,
    Roughness
};

enum class BVH_VisualizeMode {
    BoundsHit,
    TLAS,
    TrianglesHit,
};

struct IntegratorState {
    bool dirty = false;

    bool visualize_bvh = false;
    BVH_VisualizeMode bvh_visualize_type = BVH_VisualizeMode::BoundsHit;
    int bvh_visualize_depth = 500;
    
    DebugVisualizeMode debug_mode = DebugVisualizeMode::None;
};

// -- Materials ------------------

struct EmissiveMaterial {
    int color_texture;
    float emittance;
};

struct LambertianMaterial {
    int albedo_texture;
};

struct MirrorMaterial {
    int albedo_texture;
};

struct DielectricMaterial {
    int albedo_texture;
    float ior;
};

struct MetallicRoughnessMaterial {
    int albedo_texture;
    int metallic_texture;
    int roughness_texture;
};

struct Material {
    enum Type : uint8_t {
        Lambertian,
        Dielectric,
        Mirror,
        MetallicRoughness,
        Emissive,

        First = 0,
        Last = Emissive
    };
    Type type;

    union {
        EmissiveMaterial emissive;
        DielectricMaterial dielectric;
        LambertianMaterial lambertian;
        MirrorMaterial mirror;
        MetallicRoughnessMaterial metallic_roughness;
    };
};

// -- Camera ------------------

struct DeviceFilm {
    uint32_t width, height;
    ColorSpace color_space;
    Spectrum *image;
};

enum ReconstructionFilterType {
    Box,
    Gaussian,
};

enum CameraType {
    Pinhole,
    ThinLens
};

struct DeviceCamera {
    glm::ivec2 resolution;
    glm::vec3 position;

    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;

    float near, far;

    float focal_length;

    float fovy;
    glm::vec2 pixel_length;

    glm::vec3 *film;
};

// -- Samplers ------------------

struct DeviceDiscreteSampler1D {
    uint32_t *alias_table = nullptr;
    float *probability_table = nullptr;

    uint32_t size = 0;
};

// -- Acceleration Structures ------------------

// If not leaf, then first_primitive is index of left child BVHNode
struct BVHNode {
    Math::AABB bounds;
    uint32_t first_primitive, primitive_count;

    __host__ __device__ bool IsLeaf() const { return primitive_count > 0; }
    
    __host__ __device__ uint32_t &LeftChild() { return first_primitive; }
    __host__ __device__ uint32_t LeftChild() const { return first_primitive; }
    
    __host__ __device__ uint32_t RightChild() const { return LeftChild() + 1; }

    __host__ __device__ uint32_t &FirstPrimitive() { return first_primitive; }
};

struct TLASNode {
    Math::AABB bounds;
    uint32_t instance;
    uint32_t left_right;

    __host__ __device__ bool IsLeaf() const { return left_right == 0u; }
    
    __host__ __device__ uint16_t LeftChild() const { return left_right & 0xFFFFu; }
    __host__ __device__ uint16_t RightChild() const { return left_right >> 16; }
};

struct DeviceBLAS {
    uint32_t first_node;
    uint32_t node_count;

    uint32_t first_index;
    uint32_t index_count;
};

struct DeviceTLAS {
    TLASNode *tlas_nodes;
    uint32_t tlas_node_count;

    DeviceBLAS *blases;
    uint32_t blas_count;

    BVHNode *blas_nodes; // the set of all BLAS nodes merged into one buffer
    uint32_t blas_node_count;

    uint32_t *tri_indices; // all tri indices, as used for BLASes
    uint32_t tri_index_count;
};

// -- Geometry ------------------

struct TriangleMesh {
    uint32_t first_index;
    uint32_t index_count;
    uint32_t vertex_count;
};

// As of now, all supported geometry are
// just triangle meshes
struct GeometryInstance {
    int material_id = -1;
    
    TriangleMesh triangle_mesh;

    glm::mat4 transform; // object-space -> world-space
    glm::mat4 inv_transform; // world-space -> object-space
    glm::mat4 inv_transpose; // transforms covectors from object-space -> world-space

    int blas_index = -1;
    
    // Relevant only to area lights
    float total_area = 0.0f;
    int triangle_sampler_index = -1;
    int area_light_id = -1;
};

// -- Texture ------------------

template<typename T>
struct ConstantTexture {
    T value;
};

template<typename T>
struct ImageTexture {
    int width, height;
    glm::vec2 scale, offset;

    int image_id;
};

template <typename T>
struct CheckerboardTexture {
    T color0, color1;
    glm::vec2 scale, offset;
};

enum class TextureType : uint8_t {
    Constant,
    Image,
    Checkerboard,
};

template<typename T>
struct Texture {
    TextureType type;

    union {
        ConstantTexture<T> constant;
        ImageTexture<T> image;
        CheckerboardTexture<T> checkerboard;
    };
};

struct DeviceTexturePool {
    Texture<float> *textures1;
    uint32_t texture1_count;
    
    Texture<glm::vec3> *textures3;
    uint32_t texture3_count;
    
    cudaTextureObject_t *images1;
    uint32_t image1_count;

    cudaTextureObject_t *images3;
    uint32_t image3_count;
};

// -- Light ------------------

struct AreaLight {
    float power {};
    int geometry_instance_index = -1;
};
