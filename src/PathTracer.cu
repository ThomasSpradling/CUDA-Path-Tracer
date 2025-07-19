#include <thrust/detail/sort.inl>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "MathUtils.h"
#include "PathTracer.h"
#include "material.h"
#include "utils.h"


#define GLM_ENABLE_EXPERIMENTAL 
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "exception.h"
#include "scene.h"
#include <cuda_runtime_api.h>
#include "Intersections.h"
#include <cuda.h>

__host__ __device__
thrust::default_random_engine MakeSeededRandomEngine(int iter, int index, int depth) {
    int h = Math::JenkinsHash((1 << 31) | (depth << 22) | iter) ^ Math::JenkinsHash(index);
    return thrust::default_random_engine(h);
}

__global__ void GenerateRayFromCamera(Camera cam, int iteration, int trace_depth, PathSegment *path_segments) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + y * cam.resolution.x;
        PathSegment &segment = path_segments[index];

        segment.ray.origin = cam.position;
        segment.throughput = glm::vec3(1.0f);

        thrust::default_random_engine rng = MakeSeededRandomEngine(iteration, index ^ 12940, trace_depth);
        glm::vec2 jitter = Math::Sample2D(rng);
        glm::vec2 pixel_sample = glm::vec2(x, y);
        pixel_sample += jitter;

        glm::vec3 direction = cam.front;
        direction -= cam.right * cam.pixel_length.x * (pixel_sample.x - (float)cam.resolution.x * 0.5f);
        direction -= cam.up * cam.pixel_length.y * (pixel_sample.y - (float)cam.resolution.y * 0.5f);

        segment.ray.direction = glm::normalize(direction);

        segment.pixel_index = index;
        segment.remaining_bounces = trace_depth;
        segment.color_accum = glm::vec3(0.0f);
        segment.last_pdf = -1.0f;
        segment.last_pos = glm::vec3(0.0f);
        segment.hit = true;
    }
}
__global__ void ComputeIntersections(
    int depth,
    int num_paths,
    PathSegment *path_segments,
    SceneView scene,
#if SORT_BY_MATERIAL
    int *sort_keys,
#endif
    Intersection *intersections
) {
    int path_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (path_index >= num_paths) return;

    Ray ray = path_segments[path_index].ray;
    Intersection intersection{};
    float t = scene.ClosestHit(ray, intersection);

    if (t < 0.0f) {
        intersections[path_index].t = -1.0f;
        path_segments[path_index].hit = false;
        path_segments[path_index].remaining_bounces = 0;
#if SORT_BY_MATERIAL
        sort_keys[path_index] = -1;
#endif
    } else {
        intersections[path_index] = intersection;
        path_segments[path_index].hit = true;
#if SORT_BY_MATERIAL
        sort_keys[path_index] = intersection.material_id;
#endif
    }
}

__host__ __device__
float PdfLightForDir(const SceneView &scene,
                     const Intersection &light_isect,
                     float dist,
                     const glm::vec3 &w_in) {
    const AreaLight &L = scene.lights[0];
    float p_sel = L.power / scene.total_light_power;  

    float p_area = 1.0f / L.area;  

    float cosL = glm::dot(light_isect.normal, -w_in);
    if (cosL <= 0.0f) return 0.0f;

    float dist2 = dist * dist;

    return p_sel * p_area * (dist2 / cosL);
}

__global__ void ShadeMaterial(
    int iteration,
    int num_paths,
    Intersection* shadable_intersections,
    PathSegment* path_segments,
    SceneView scene
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    PathSegment &path_segment = path_segments[index];

    if (index >= num_paths)
        return;
    
    Intersection intersection = shadable_intersections[index];
    if (intersection.t <= 1e-6) {
        glm::vec3 radiance = path_segment.throughput * glm::vec3(0.0f);
        path_segment.color_accum += radiance;
        path_segment.remaining_bounces = 0;
        return;
    }

#if DEBUG_UV
    path_segment.color_accum = glm::vec3(intersection.uv, 0.0f);
    path_segment.remaining_bounces = 0;
#elif DEBUG_NORMAL
    path_segment.color_accum = glm::vec3(intersection.normal);
    path_segment.remaining_bounces = 0;
#elif DEBUG_DEPTH
    const float near_plane = 0.1f;
    const float far_plane  = 20.0f;
    float d = (intersection.t - near_plane) / (far_plane - near_plane);
    d = glm::clamp(d, 0.0f, 1.0f);

    float intensity = 1.0f - d;
    path_segment.color_accum = glm::vec3(intensity);
    path_segment.remaining_bounces = 0;
#else
    Material &material = scene.materials[intersection.material_id];
    thrust::default_random_engine rng = MakeSeededRandomEngine(iteration, index, path_segment.remaining_bounces);
    
    if (material.type == Material::Type::Light) {
        glm::vec3 radiance = material.mat.light.base_color * material.mat.light.emittance;
        
        float weight_bsdf = 1.0f;
#if USE_MIS
        if (path_segment.last_pdf > 0.0f) {
            // If the last hit was actually from a BSDF scatter. This is direct lighting contribution
            // from BSDF sampling
            AreaLight &light = scene.lights[material.mat.light.light_id];
            float dist2 = glm::distance2(path_segment.last_pos, intersection.pos);
            glm::vec3 w_in = glm::normalize(path_segment.last_pos - intersection.pos);
            float cos_light = glm::dot(intersection.normal, w_in);

            if (cos_light > 0.0f) {
                float light_pdf = (light.power / scene.total_light_power) * (1.0f / light.area) * dist2 / cos_light;
                weight_bsdf = Math::PowerHeuristic(path_segment.last_pdf, light_pdf);
            }
        }
#endif
        path_segment.color_accum += radiance * weight_bsdf * path_segment.throughput;
        path_segment.remaining_bounces = 0;
        return;
    } else {
        glm::vec3 w_out = -glm::normalize(path_segment.ray.direction);

#if USE_MIS
        // Direct contribution from area sampling
        if (!(material.type == Material::Type::Mirror || material.type == Material::Type::Dielectric)) {
            LightSample light_sample = scene.SampleLight(intersection, rng);

            glm::vec3 w_in = glm::normalize(light_sample.position - intersection.pos);
            float dist = glm::distance(light_sample.position, intersection.pos);
            
            Intersection hit;
            Ray shadow_ray = Ray::MakeOffseted(intersection.pos, w_in);
            float t = scene.AnyHit(shadow_ray, hit, dist - 1e-4);
            if (t < 0.0f) {
                // If not occluded

                float cos_surf = glm::dot(intersection.normal, w_in);
                float cos_light = glm::dot(light_sample.normal, -w_in);
                if (cos_surf > 0.0f && cos_light > 0.0f) {
                    glm::vec3 bsdf = material.BSDF(intersection, w_out, w_in);

                    float light_pdf = light_sample.pdf * dist * dist / cos_light;
                    float light_weight = Math::PowerHeuristic(light_pdf, material.PDF(intersection, w_out, w_in));
                    path_segment.color_accum += path_segment.throughput * light_weight * bsdf * cos_surf * light_sample.radiance / light_pdf;
                }
            }
        }
#endif
            
        // Indirect contribution
        BSDFSample sample = material.Sample(intersection, w_out, rng);
        float cos_surf = fabsf(glm::dot(intersection.normal, sample.w_in));
        path_segment.throughput *= sample.bsdf * cos_surf / sample.pdf;

        if (material.type == Material::Type::Mirror || material.type == Material::Type::Dielectric) {
            path_segment.last_pdf = -1.0f;
        } else {
            path_segment.last_pdf = sample.pdf;
        }
        path_segment.last_pos = intersection.pos;

        path_segment.ray = Ray::MakeOffseted(intersection.pos, sample.w_in);
        path_segment.remaining_bounces--;
    }


            
        //     float distance = glm::length(light_sample.position - intersection.pos);
        
        // BSDFSample sample = material.Sample(intersection, w_out, rng);
        // float cos_theta = fabsf(glm::dot(intersection.normal, sample.w_in));                
        // if (sample.type == InvalidSample || sample.pdf < 1e-6f) {
        //     path_segment.remaining_bounces = 0;
        // } else {
        //     Ray shadow_ray = Ray::MakeOffseted(intersection.pos, sample.w_in);

        //     Intersection light_hit;
        //     float t_hit = scene.ClosestHit(shadow_ray, light_hit);
        //     if (t_hit > 0.0f && scene.materials[light_hit.material_id].type == Material::Type::Light) {
        //         glm::vec3 radiance = scene.materials[light_hit.material_id].mat.light.base_color
        //         * scene.materials[light_hit.material_id].mat.light.emittance;
                
        //         float light_pdf = PdfLightForDir(scene, light_hit, t_hit, sample.w_in);
                
        //         float bsdf_weight;
        //         if (material.type == Material::Type::Mirror ||
        //             material.type == Material::Type::Dielectric) {
        //             bsdf_weight = 1.0f;
        //         } else {
        //             bsdf_weight = Math::PowerHeuristic(sample.pdf, light_pdf);
        //         }
                
        //         path_segment.color_accum += path_segment.throughput * sample.bsdf * radiance * bsdf_weight * cos_theta / sample.pdf;
        //         path_segment.remaining_bounces = 0;
        //         return;
        //     }

        //     path_segment.throughput *= sample.bsdf * cos_theta / sample.pdf;
        //     path_segment.ray = Ray::MakeOffseted(intersection.pos, sample.w_in);
        //     path_segment.remaining_bounces--;
        // }
#endif
}

__global__ void FinalGather(int path_count, glm::vec3 *image, PathSegment *iteration_paths) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= path_count)
        return;

    PathSegment &iteration_path = iteration_paths[index];

    if (iteration_path.pixel_index >= 0 && iteration_path.remaining_bounces <= 0) {
        glm::vec3 &c = iteration_path.throughput;
        if (isnan(c.r) || isnan(c.g) || isnan(c.b))
            return;

        glm::vec3 color = iteration_path.color_accum;
        float max_component = glm::compMax(iteration_path.color_accum);
        constexpr float max_component_value = 10.0f;
        if (max_component > max_component_value) {
            color *= max_component_value / max_component;
        }
        image[iteration_path.pixel_index] += iteration_path.color_accum;
    }
}

__global__ void SendImageToVulkanBuffer(uchar4 *dest, glm::ivec2 resolution, int iteration, glm::vec3 *image) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + y * resolution.x;
        glm::vec3 pixel = image[index];

        int denom = iteration + 1;

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

void PathTracer::Reset() {
    const int pixelCount =
        m_scene.State().camera.resolution.x *
        m_scene.State().camera.resolution.y;

    cudaMemset(md_image, 0, pixelCount * sizeof(glm::vec3));
    cudaMemset(md_paths, 0, pixelCount * sizeof(PathSegment));
    cudaMemset(md_terminated, 0, pixelCount * sizeof(PathSegment));
    cudaMemset(md_intersections, 0, pixelCount * sizeof(Intersection));
}

void PathTracer::Init() {
    const Camera &cam = m_scene.State().camera;
    const int pixel_count = cam.resolution.x * cam.resolution.y;

    CUDA_CHECK(cudaMalloc(&md_image, pixel_count * sizeof(glm::vec3)));
    CUDA_CHECK(cudaMemset(md_image, 0, pixel_count * sizeof(glm::vec3)));

    CUDA_CHECK(cudaMalloc(&md_paths, pixel_count * sizeof(PathSegment)));
    m_paths_thrust = thrust::device_ptr<PathSegment>(md_paths);

    CUDA_CHECK(cudaMalloc(&md_terminated, pixel_count * sizeof(PathSegment)));
    m_terminated_thrust = thrust::device_ptr<PathSegment>(md_terminated);

    CUDA_CHECK(cudaMalloc(&md_intersections, pixel_count * sizeof(Intersection)));
    CUDA_CHECK(cudaMemset(md_intersections, 0, pixel_count * sizeof(Intersection)));
    m_intersection_thrust = thrust::device_ptr<Intersection>(md_intersections);


#if SORT_BY_MATERIAL
    CUDA_CHECK(cudaMalloc(&md_keys, pixel_count * sizeof(uint32_t)));
    m_keys_thrust = thrust::device_ptr<int>(md_keys);
#endif
}

void PathTracer::Destroy() {
    CUDA_CHECK(cudaFree(md_image));
    CUDA_CHECK(cudaFree(md_paths));
    CUDA_CHECK(cudaFree(md_terminated));
    CUDA_CHECK(cudaFree(md_intersections));

#if SORT_BY_MATERIAL
    CUDA_CHECK(cudaFree(md_keys));
#endif
}

struct CompactTerminatedPaths {
    __host__ __device__
        bool operator()(const PathSegment& p) const {
        return p.remaining_bounces > 0;
    }
};

struct RemoveInvalidPaths {
    __host__ __device__
        bool operator()(const PathSegment& p) const {
        return (!p.hit) || (p.pixel_index < 0) || (p.remaining_bounces <= 0);
    }
};

void PathTracer::PathTrace(uchar4* data, int frame, int iteration) {
    const int trace_depth = m_scene.State().trace_depth;
    const Camera& cam = m_scene.State().camera;
    const int pixel_count = cam.resolution.x * cam.resolution.y;

    dim3 block_size(8, 8);
    dim3 block_count(
        (cam.resolution.x + block_size.x - 1) / block_size.x,
        (cam.resolution.y + block_size.y - 1) / block_size.y
    );

    const int block_size_1d = 128;

    using zip_it = thrust::zip_iterator<thrust::tuple<PathSegment*, Intersection*>>;

    GenerateRayFromCamera<<<block_count, block_size>>>(cam, iteration, trace_depth, md_paths);

    thrust::device_ptr<PathSegment> paths_begin = thrust::device_pointer_cast(md_paths);

    thrust::device_ptr<PathSegment> terminated_begin_base = thrust::device_pointer_cast(md_terminated);
    thrust::device_ptr<PathSegment> terminated_begin = thrust::device_pointer_cast(md_terminated);

    int num_paths = pixel_count;
    for (int depth = 0; depth < trace_depth; ++depth) {
        cudaMemset(md_intersections, 0, pixel_count * sizeof(Intersection));

        dim3 num_blocks_segments = (num_paths + block_size_1d - 1) / block_size_1d;
        ComputeIntersections<<<num_blocks_segments, block_size_1d>>>(
            depth, num_paths, md_paths,
            m_scene.View(),
#if SORT_BY_MATERIAL
            md_keys,
#endif
            md_intersections);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

#if SORT_BY_MATERIAL
        auto zipped = thrust::make_zip_iterator(thrust::make_tuple(m_intersection_thrust, m_paths_thrust));
        thrust::sort_by_key(m_keys_thrust, m_keys_thrust + num_paths, zipped);
        thrust::sort_by_key(m_keys_thrust, m_keys_thrust + num_paths, m_paths_thrust);
#endif

        ShadeMaterial<<<num_blocks_segments, block_size_1d>>>(iteration, num_paths, md_intersections, md_paths, m_scene.View());
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        terminated_begin = thrust::remove_copy_if(
            thrust::device,
            paths_begin,
            paths_begin + num_paths,
            terminated_begin,
            CompactTerminatedPaths()
        );

        auto end = thrust::remove_if(
            thrust::device,
            paths_begin,
            paths_begin + num_paths,
            RemoveInvalidPaths()
        );

        num_paths = static_cast<std::size_t>(end - paths_begin);

        if (num_paths == 0) {
            break;
        }
    }

    int num_terminated = terminated_begin - terminated_begin_base;
    if (num_terminated > 0) {
        dim3 num_blocks_pixels = (num_terminated + block_size_1d - 1) / block_size_1d;
        FinalGather<<<num_blocks_pixels, block_size_1d>>>(num_terminated, md_image, md_terminated);
    }
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    SendImageToVulkanBuffer<<<block_count, block_size>>>(data, cam.resolution, iteration, md_image);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(m_scene.State().image.Data(), md_image, pixel_count * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaGetLastError());
}