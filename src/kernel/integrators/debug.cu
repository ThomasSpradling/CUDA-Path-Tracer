#include "debug.h"
#include <cuda_device_runtime_api.h>
#include <vector_functions.h>

#include <thrust/random.h>

constexpr int block_size_1d = 128;

DebugIntegrator::DebugIntegrator() {

}

DebugIntegrator::~DebugIntegrator() {
    
}

void DebugIntegrator::Init(glm::ivec2 resolution) {
    const uint32_t num_pixels = resolution.x * resolution.y;
    
    CUDA_CHECK(cudaMalloc(&m_image, num_pixels * sizeof(glm::vec3)));
    CUDA_CHECK(cudaMalloc(&m_dev_scene, sizeof(DeviceScene)));
}

void DebugIntegrator::Destroy() {
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaFree(m_image));
}

void DebugIntegrator::Resize(glm::ivec2 resolution) {
    CUDA_CHECK(cudaDeviceSynchronize());

    Destroy();
    Init(resolution);
}

void DebugIntegrator::Reset(glm::ivec2 resolution) {
    const int pixel_count = resolution.x * resolution.y;
    CUDA_CHECK(cudaMemset(m_image, 0, pixel_count * sizeof(glm::vec3)));
}

__device__ glm::vec3 HeatMapColor(int depth, int max_depth) {
    float t = static_cast<float>(depth) / max_depth;
    float lambda = glm::mix(Math::VISIBLE_SPECTRUM_BEGIN, Math::VISIBLE_SPECTRUM_END, t);
    
    glm::vec3 xyz = Math::WavelengthToXYZ(lambda);
    glm::vec3 rgb = Math::XYZ_To_LinearRGB(xyz);
    return Math::LinearRGB_To_sRGB(rgb);
}

template<DebugVisualizeMode Mode>
__global__ void DebugView(DeviceScene scene, glm::ivec2 resolution, glm::vec3 *image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= resolution.x || y >= resolution.y) return;

    uint32_t index = x + resolution.x * y;

    glm::vec4 sample = glm::vec4(0.5f, 0.5f, 0.0f, 0.0f);
    Math::Ray ray = SpawnCameraRay(scene.camera, x, y, sample);

    const glm::vec3 background_color = glm::vec3(0.4f);

    Intersection intersection;
    bool hit = IntersectScene<IntersectQueryMode::CLOSEST>(scene, ray, intersection);
    if (!hit) {
        image[index] = background_color;
        return;
    }

    Material &material = scene.materials[intersection.material_id];
 
    glm::vec3 accum;
    if constexpr (Mode == DebugVisualizeMode::GeometricNormals) {
        accum = intersection.normal;
    } else if constexpr (Mode == DebugVisualizeMode::Albedo) {
        // Grabs albedo texture if the material supports one
        if (material.type == Material::Lambertian) {
            accum = SampleTexture<glm::vec3>(scene.texture_pool, material.lambertian.albedo_texture, intersection.uv);
        } else if (material.type == Material::MetallicRoughness)  {
            accum = SampleTexture<glm::vec3>(scene.texture_pool, material.metallic_roughness.albedo_texture, intersection.uv);
        } else {
            accum = glm::vec3(0.0f);
        }
    } else if constexpr (Mode == DebugVisualizeMode::Metallic) {
        // Grabs the metallic component of metallic roughness material
        if (material.type == Material::MetallicRoughness)  {
            float metallic = SampleTexture<float>(scene.texture_pool, material.metallic_roughness.metallic_texture, intersection.uv);
            accum = glm::vec3(metallic);
        } else {
            accum = glm::vec3(0.0f);
        }
    } else if constexpr (Mode == DebugVisualizeMode::Roughness) {
        // Grabs the roughness component of metallic roughness material
        if (material.type == Material::MetallicRoughness)  {
            float roughness = SampleTexture<float>(scene.texture_pool, material.metallic_roughness.roughness_texture, intersection.uv);
            accum = glm::vec3(roughness);
        } else {
            accum = glm::vec3(0.0f);
        }
    } else if constexpr (Mode == DebugVisualizeMode::UV) {
        accum = glm::vec3(intersection.uv, 0.0f);
    } else if constexpr (Mode == DebugVisualizeMode::Depth) {
        float near_clip = scene.camera.near;
        float far_clip = scene.camera.far;
        float t = intersection.time;

        if (t < near_clip || t > far_clip) {
            accum = background_color;
        } else {
            float depth = (t - near_clip) / (far_clip - near_clip);
            depth = glm::clamp(depth, 0.0f, 1.0f);
            accum  = glm::vec3(depth);
        }
    }
    image[index] = accum;
}

template<BVH_VisualizeMode Mode>
__global__ void VisualizeBVH(DeviceScene scene, glm::ivec2 resolution, glm::vec3 *image, int bvh_visualize_depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= resolution.x || y >= resolution.y) return;

    uint32_t index = x + resolution.x * y;
    
    glm::vec4 sample = glm::vec4(0.5f, 0.5f, 0.0f, 0.0f);
    Math::Ray ray = SpawnCameraRay(scene.camera, x, y, sample);
 
    glm::vec3 accum;

    // BVH Visualizations
    if constexpr (Mode == BVH_VisualizeMode::BoundsHit) {
        int depth = 0;
        auto callback = [&depth](int, bool is_blas) {
            depth++;
        };

        Intersection intersection;
        IntersectScene<IntersectQueryMode::CLOSEST>(scene, ray, intersection, FLT_MAX, callback);

        glm::vec3 color = HeatMapColor(depth, bvh_visualize_depth);
        accum = color;
    } else if constexpr (Mode == BVH_VisualizeMode::TrianglesHit) {
        int tri_count = 0;
        auto callback = [&tri_count](int prim_count, bool is_blas) {
            if (prim_count > 0 && is_blas) {
                tri_count += prim_count;
            }
        };

        Intersection intersection;
        IntersectScene<IntersectQueryMode::CLOSEST>(scene, ray, intersection, FLT_MAX, callback);

        const float MAX_TRIS = 50.0f;
        float t = logf(tri_count + 1.0f) / logf(MAX_TRIS + 1.0f);
        t = glm::clamp(t, 0.0f, 1.0f);

        glm::vec3 green(0.0f,0.3f,0.0f);
        glm::vec3 yellow(1.0f,1.0f,0.0f);
        accum = glm::mix(green, yellow, t);
    } else if constexpr (Mode == BVH_VisualizeMode::TLAS) {
        int depth = 0;
        auto callback = [&depth](int, bool is_blas) {
            if (!is_blas)
                depth++;
        };

        Intersection intersection;
        IntersectScene<IntersectQueryMode::CLOSEST>(scene, ray, intersection, FLT_MAX, callback);
        const int MAX_DEPTH = 300;

        glm::vec3 color = HeatMapColor(depth, bvh_visualize_depth);
        accum = color;
    }

    image[index] = accum;
}

__global__ void SendImageToBuffer(uchar4 *dest, glm::ivec2 resolution, const glm::vec3 *image) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + y * resolution.x;

        glm::vec3 pixel = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pixel.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pixel.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pixel.z * 255.0), 0, 255);

        dest[index] = make_uchar4(color.x, color.y, color.z, 255);
    }
}

void DebugIntegrator::Render(glm::ivec2 resolution, const DeviceScene &scene, uchar4 *data, int iteration, const IntegratorState &state) {
    dim3 block_size(8, 8);
    dim3 block_count(
        (resolution.x + block_size.x - 1) / block_size.x,
        (resolution.y + block_size.y - 1) / block_size.y
    );

    if (state.visualize_bvh) {
        if (state.bvh_visualize_type == BVH_VisualizeMode::BoundsHit) {
            VisualizeBVH<BVH_VisualizeMode::BoundsHit><<<block_count, block_size>>>(scene, resolution, m_image, state.bvh_visualize_depth);
        } else if (state.bvh_visualize_type == BVH_VisualizeMode::TLAS) {
            VisualizeBVH<BVH_VisualizeMode::TLAS><<<block_count, block_size>>>(scene, resolution, m_image, state.bvh_visualize_depth);
        } else if (state.bvh_visualize_type == BVH_VisualizeMode::TrianglesHit) {
            VisualizeBVH<BVH_VisualizeMode::TrianglesHit><<<block_count, block_size>>>(scene, resolution, m_image, state.bvh_visualize_depth);
        }
        SendImageToBuffer<<<block_count, block_size>>>(data, resolution, m_image);
    }

    switch (state.debug_mode) {
        case DebugVisualizeMode::Albedo:
            DebugView<DebugVisualizeMode::Albedo><<<block_count, block_size>>>(scene, resolution, m_image);
            break;
        case DebugVisualizeMode::GeometricNormals:
            DebugView<DebugVisualizeMode::GeometricNormals><<<block_count, block_size>>>(scene, resolution, m_image);
            break;
        case DebugVisualizeMode::Metallic:
            DebugView<DebugVisualizeMode::Metallic><<<block_count, block_size>>>(scene, resolution, m_image);
            break;
        case DebugVisualizeMode::Roughness:
            DebugView<DebugVisualizeMode::Roughness><<<block_count, block_size>>>(scene, resolution, m_image);
            break;
        case DebugVisualizeMode::UV:
            DebugView<DebugVisualizeMode::UV><<<block_count, block_size>>>(scene, resolution, m_image);
            break;
        case DebugVisualizeMode::Depth:
            DebugView<DebugVisualizeMode::Depth><<<block_count, block_size>>>(scene, resolution, m_image);
            break;
        default:
            return;
    }
    SendImageToBuffer<<<block_count, block_size>>>(data, resolution, m_image);
}
