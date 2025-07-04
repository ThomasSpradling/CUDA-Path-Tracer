#include "PathTracer.h"

#include <glm/glm.hpp>

#include "exception.h"
#include "scene.h"
#include <cuda_runtime_api.h>
#include "Intersections.h"
#include <cuda.h>

__global__ void GenerateRayFromCamera(Camera cam, int iteration, int trace_depth, PathSegment *path_segments) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + y * cam.resolution.x;
        PathSegment &segment = path_segments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        glm::vec3 direction = cam.front;
        direction -= cam.right * cam.pixel_length.x * ((float)x - (float)cam.resolution.x * 0.5f);
        direction -= cam.up * cam.pixel_length.y * ((float)y - (float)cam.resolution.y * 0.5f);

        segment.ray.direction = glm::normalize(direction);

        segment.pixel_index = index;
        segment.remaining_bounces = trace_depth;
    }
}

__global__ void ComputeIntersections(
    int depth,
    int num_paths,
    PathSegment *path_segments,
    Geometry *geometries,
    int geoms_size,
    ShadableIntersection *intersections
) {
    int path_index = threadIdx.x + blockIdx.x * blockDim.x;

    if (path_index < num_paths) {
        PathSegment path_segment = path_segments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        for (int i = 0; i < geoms_size; i++)
        {
            Geometry &geom = geometries[i];

            if (geom.type == GeometryType::Cube) {
                t = BoxIntersectionTest(geom, path_segment.ray, tmp_intersect, tmp_normal, outside);
            } else if (geom.type == GeometryType::Sphere) {
                t = SphereIntersectionTest(geom, path_segment.ray, tmp_intersect, tmp_normal, outside);
            }

            if (t > 0.0f && t_min > t) {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1) {
            intersections[path_index].t = -1.0f;
            intersections[path_index].material_id = -1;
            intersections[path_index].normal = glm::vec3(0.0f);
        }  else {
            intersections[path_index].t = t_min;
            intersections[path_index].material_id = geometries[hit_geom_index].material_id;
            intersections[path_index].normal = normal;
        }
    }
}

__global__ void ShadeFakeMaterial(
    int iteration,
    int num_paths,
    ShadableIntersection* shadable_intersections,
    PathSegment* path_segments,
    Material* materials
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < num_paths) {
        ShadableIntersection intersection = shadable_intersections[index];

        if (intersection.t > 0.0f && intersection.material_id >= 0) {
            Material material = materials[intersection.material_id];
            glm::vec3 material_color = material.color;

            if (material.emittance > 0.0f) {
                path_segments[index].color *= material_color * material.emittance;
            } else {
                float light_term = glm::dot(intersection.normal, glm::vec3(0.0f, 1.0f, 0.0f));
                path_segments[index].color *= (material_color * light_term) * 0.3f + ((1.0f - intersection.t * 0.02f) * material_color) * 0.7f;
            }
        } else {
            path_segments[index].color = glm::vec3(0.0f);
        }
    }
}

__global__ void FinalGather(int path_count, glm::vec3 *image, PathSegment *iteration_paths) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < path_count) {
        PathSegment iteration_path = iteration_paths[index];
        image[iteration_path.pixel_index] += iteration_path.color;
    }
}

__global__ void SendImageToVulkanBuffer(uchar4 *dest, glm::ivec2 resolution, int iteration, glm::vec3 *image) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + y * resolution.x;
        glm::vec3 pixel = image[index];

        int denom = glm::max(1, iteration);

        glm::ivec3 color;
        color.x = glm::clamp((int)(pixel.x / denom * 255.0), 0, 255);
        color.y = glm::clamp((int)(pixel.y / denom * 255.0), 0, 255);
        color.z = glm::clamp((int)(pixel.z / denom * 255.0), 0, 255);

        dest[index] = make_uchar4(color.x, color.y, color.z, 255);
    }
}

PathTracer::PathTracer(const VkExtent2D &extent, Scene &scene)
    : m_extent(extent)
    , m_scene(scene)
{
    Init();
}

PathTracer::~PathTracer() {
    Destroy();
}

void PathTracer::Init() {
    const Camera &cam = m_scene.State().camera;
    const int pixel_count = cam.resolution.x * cam.resolution.y;

    CUDA_CHECK(cudaMalloc(&md_image, pixel_count * sizeof(glm::vec3)));
    CUDA_CHECK(cudaMemset(md_image, 0, pixel_count * sizeof(glm::vec3)));

    CUDA_CHECK(cudaMalloc(&md_paths, pixel_count * sizeof(PathSegment)));

    CUDA_CHECK(cudaMalloc(&md_geometries, m_scene.Geometries().size() * sizeof(Geometry)));
    CUDA_CHECK(cudaMemcpy(md_geometries, m_scene.Geometries().data(), m_scene.Geometries().size() * sizeof(Geometry), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&md_materials, m_scene.Materials().size() * sizeof(Material)));
    CUDA_CHECK(cudaMemcpy(md_materials, m_scene.Materials().data(), m_scene.Materials().size() * sizeof(Material), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&md_intersections, pixel_count * sizeof(ShadableIntersection)));
    CUDA_CHECK(cudaMemset(md_intersections, 0, pixel_count * sizeof(ShadableIntersection)));
}

void PathTracer::Destroy() {
    CUDA_CHECK(cudaFree(md_image));
    CUDA_CHECK(cudaFree(md_paths));
    CUDA_CHECK(cudaFree(md_geometries));
    CUDA_CHECK(cudaFree(md_materials));
    CUDA_CHECK(cudaFree(md_intersections));
}

void PathTracer::PathTrace(uchar4 *data, int frame, int iteration) {
    const int trace_depth = m_scene.State().trace_depth;
    const Camera& cam = m_scene.State().camera;
    const int pixel_count = cam.resolution.x * cam.resolution.y;

    dim3 block_size(8, 8);
    dim3 block_count(
        (cam.resolution.x + block_size.x - 1) / block_size.x,
        (cam.resolution.y + block_size.y - 1) / block_size.y
    );

    const int block_size_1d = 128;

    GenerateRayFromCamera<<<block_count, block_size>>>(cam, iteration, trace_depth, md_paths);

    int depth = 0;
    PathSegment *d_path_end = md_paths + pixel_count;
    int num_paths = d_path_end - md_paths;

    bool iteration_complete = false;
    while (!iteration_complete) {
        cudaMemset(md_intersections, 0, pixel_count * sizeof(ShadableIntersection));

        dim3 num_blocks_segments = (num_paths + block_size_1d - 1) / block_size_1d;
        ComputeIntersections<<<num_blocks_segments, block_size_1d>>>(depth, num_paths, md_paths, md_geometries, m_scene.Geometries().size(), md_intersections);
        depth++;

        ShadeFakeMaterial<<<num_blocks_segments, block_size_1d>>>(iteration, num_paths, md_intersections, md_paths, md_materials);
        iteration_complete = true;
    }

    dim3 num_blocks_pixels = (pixel_count + block_size_1d - 1) / block_size_1d;
    FinalGather<<<num_blocks_pixels, block_size_1d>>>(num_paths, md_image, md_paths);

    SendImageToVulkanBuffer<<<block_count, block_size>>>(data, cam.resolution, iteration, md_image);

    cudaMemcpy(m_scene.State().image.Data(), md_image, pixel_count * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaGetLastError());
}
