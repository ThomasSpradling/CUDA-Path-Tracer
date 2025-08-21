#include "path_Trace.h"
#include "../../math/tonemap.h"
#include "../light.h"
#include <vector>
#include <vector_functions.h>

constexpr int block_size_1d = 64;

PathIntegrator::PathIntegrator() {
}

PathIntegrator::~PathIntegrator() {
    
}

void PathIntegrator::Init(glm::ivec2 resolution) {
    const uint32_t num_pixels = resolution.x * resolution.y;
    
    CUDA_CHECK(cudaMalloc(&m_image, num_pixels * sizeof(glm::vec3)));
    CUDA_CHECK(cudaMalloc(&m_dev_scene, sizeof(DeviceScene)));

    CUDA_CHECK(cudaMalloc(&m_extension_queue, num_pixels * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&m_emissive_hit_queue, num_pixels * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&m_terminated_queue, num_pixels * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&m_shadow_ray_queue, num_pixels * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&m_lambertian_queue, num_pixels * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&m_mirror_queue, num_pixels * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&m_metallic_roughness_queue, num_pixels * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&m_dielectric_queue, num_pixels * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void **) &m_queue_counters, sizeof(QueueCounters)));
    
    m_path_segements.Alloc(num_pixels);
    m_num_paths = num_pixels;
}

void PathIntegrator::Destroy() {
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaFree(m_image));
    cudaFree(m_extension_queue);
    cudaFree(m_emissive_hit_queue);
    cudaFree(m_terminated_queue);
    cudaFree(m_shadow_ray_queue);
    cudaFree(m_lambertian_queue);
    cudaFree(m_mirror_queue);
    cudaFree(m_metallic_roughness_queue);
    cudaFree(m_dielectric_queue);

    m_path_segements.Free();

    m_extension_queue = nullptr;
    m_terminated_queue = nullptr;
    m_emissive_hit_queue = nullptr;
    m_shadow_ray_queue = nullptr;

    m_lambertian_queue = nullptr;
    m_mirror_queue = nullptr;
    m_metallic_roughness_queue = nullptr;
    m_dielectric_queue = nullptr;
}

void PathIntegrator::Resize(glm::ivec2 resolution) {
    CUDA_CHECK(cudaDeviceSynchronize());

    Destroy();
    Init(resolution);
}

void PathIntegrator::Reset(glm::ivec2 resolution) {
    const int pixel_count = resolution.x * resolution.y;
    CUDA_CHECK(cudaMemset(m_image, 0, pixel_count * sizeof(glm::vec3)));
}

__global__ void GenerateRaysKernel(
    DeviceScene scene,
    int iteration,
    glm::ivec2 resolution,
    PathSegments path_segments,
    uint32_t *extension_queue,
    QueueCounters *counter
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= resolution.x || y >= resolution.y) return;

    uint32_t index = x + resolution.x * y;
    path_segments.pixel_index[index] = index;

    Sampler sampler {};
    sampler.type = SamplerType::Independent;
    SeedSampler(sampler, glm::ivec2(x, y), iteration);

    path_segments.sampler[index] = sampler;
    
    glm::vec4 sample = Sample4D(path_segments.sampler[index]);
    Math::Ray ray = SpawnCameraRay(scene.camera, x, y, sample);

    path_segments.ray[index] = ray;
    path_segments.remaining_bounces[index] = scene.max_depth;
    path_segments.throughput[index] = Spectrum(1.0f);
    path_segments.accum[index] = Spectrum(0.0f);
    path_segments.last_pdf[index] = -1.0f;

    uint32_t idx = atomicAdd(&counter->extension_count, 1u);
    extension_queue[idx] = index;
}

__global__ void IntersectClosestKernel(
    DeviceScene scene,
    PathSegments path_segments,
    uint32_t num_paths,
    QueueCounters *queue_counters,
    uint32_t *extension_queue,
    uint32_t *emission_queue,
    uint32_t *lambertian_queue,
    uint32_t *mirror_queue,
    uint32_t *metallic_roughness_queue,
    uint32_t *dielectric_queue
) {
    int queue_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (queue_index >= queue_counters->extension_count) return;

    uint32_t index = extension_queue[queue_index];
    if (index >= num_paths) return;
    
    Intersection &intersection = path_segments.intersection[index];
    bool hit = IntersectScene<IntersectQueryMode::CLOSEST>(scene, path_segments.ray[index], intersection);

    path_segments.intersection[index] = intersection;
    if (hit) {
        Material &material = scene.materials[intersection.material_id];

        if (material.type == Material::Type::Emissive) {
            path_segments.emissive_type[index] = EmissiveType::Area;

            uint32_t idx = atomicAdd(&queue_counters->emissive_hit_count, 1u);
            emission_queue[idx] = index;
            return;
        }

        uint32_t* queue;
        uint32_t* counter;

        switch (material.type) {
        case Material::Type::Lambertian: {
            uint32_t idx = atomicAdd(&queue_counters->lambertian_count, 1u);
            lambertian_queue[idx] = index;
        } break;

        case Material::Type::Mirror: {
            uint32_t idx = atomicAdd(&queue_counters->mirror_count, 1u);
            mirror_queue[idx] = index;
        } break;

        case Material::Type::MetallicRoughness: {
            uint32_t idx = atomicAdd(&queue_counters->metallic_roughness_count, 1u);
            metallic_roughness_queue[idx] = index;
        } break;

        case Material::Type::Dielectric: {
            uint32_t idx = atomicAdd(&queue_counters->dielectric_count, 1u);
            dielectric_queue[idx] = index;
        } break;

        default:
            return;
        }
    }
    else {
        // Environment map

        path_segments.emissive_type[index] = EmissiveType::Infinite;

        uint32_t idx = atomicAdd(&queue_counters->emissive_hit_count, 1u);
        emission_queue[idx] = index;
    }
}

template<Material::Type M>
__global__ void MaterialShadeKernel(
    DeviceScene scene,
    PathSegments path_segments,
    uint32_t num_paths,
    uint32_t *material_queue,
    uint32_t *extension_queue,
    uint32_t *shadow_ray_queue,
    uint32_t *terminated_queue,
    QueueCounters *counters,
    glm::vec3 *image
) {
    uint32_t* counter;
    if constexpr (M == Material::Type::Lambertian) {
        counter = &counters->lambertian_count;
    } else if constexpr (M == Material::Type::Mirror) {
        counter = &counters->mirror_count;
    } else if constexpr (M == Material::Type::MetallicRoughness) {
        counter = &counters->metallic_roughness_count;
    } else if constexpr (M == Material::Type::Dielectric) {
        counter = &counters->dielectric_count;
    }

    int queue_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (queue_index >= *counter) return;

    int index = material_queue[queue_index];
    if (index >= num_paths) return;

    Intersection &intersection = path_segments.intersection[index];
    Material &material = scene.materials[intersection.material_id];

    // The direction we came from
    glm::vec3 w_out = -glm::normalize(path_segments.ray[index].direction);

    // Direct contribution from area sampling
    if (!(material.type == Material::Type::Mirror || material.type == Material::Type::Dielectric)) {
        LightSample light_sample = SampleSceneLight(scene, intersection, path_segments.sampler[index]);
        glm::vec3 w_in = glm::normalize(light_sample.position - intersection.pos);
        
        // Shoot a shadow ray, which will add a contribution
        Math::Ray shadow_ray;
        shadow_ray.origin = intersection.pos + Math::EPSILON * w_in;
        shadow_ray.direction = w_in;
        
        float dist = glm::distance(light_sample.position, intersection.pos);
        Intersection temp_inter;
        bool hit = IntersectScene<IntersectQueryMode::ANY>(scene, shadow_ray, temp_inter, dist - 1e-4);
        if (!hit) {
            float cos_surf = glm::dot(intersection.normal, w_in);
            float cos_light = glm::dot(light_sample.normal, -w_in);
            
            if (cos_surf > 0.0f && cos_light > 0.0f) {
                glm::vec3 bsdf = EvalMaterialBSDF(material, scene.texture_pool, intersection, w_in, w_out);
                float bsdf_pdf = MaterialPDF(material, scene.texture_pool, intersection, w_in, w_out);

                float light_pdf = light_sample.pdf * dist * dist / cos_light;
                float light_weight = Math::PowerHeuristic(light_pdf, bsdf_pdf);
                // float light_weight = 1.0f;

                path_segments.accum[index] += path_segments.throughput[index] * light_weight * bsdf * cos_surf * light_sample.radiance / light_pdf;
            }
        }
    }

    // Indirect contribution
    int &remaining_bounces = path_segments.remaining_bounces[index];
    BSDF_Sample sample = SampleMaterial(material, scene.texture_pool, intersection, w_out, path_segments.sampler[index]);
    float cos_surf = fabsf(glm::dot(intersection.normal, sample.w_in));

    if (sample.pdf < Math::EPSILON || cos_surf < Math::EPSILON || sample.type == BSDF_SampleType::Invalid) {
        return;
    }
    
    path_segments.throughput[index] *= sample.bsdf * cos_surf / sample.pdf;
    if (material.type == Material::Type::Mirror || material.type == Material::Type::Dielectric) {
        path_segments.last_pdf[index] = -1.0f;
    } else {
        path_segments.last_pdf[index] = sample.pdf;
    }
    path_segments.last_pos[index] = intersection.pos;
    
    // Russian roulette: After a few bounces, survive only with probability proportional to luminance
    float bounces = scene.max_depth - remaining_bounces;
    const int russian_roulette_start = 3;

    if (bounces >= russian_roulette_start) {
        const float survival_prob = glm::clamp(Math::Luminance(path_segments.throughput[index]), 0.05f, 0.95f);
        if (Sample1D(path_segments.sampler[index]) >= survival_prob) {
            remaining_bounces = 0;
            return;
        }

        // This factor ensures our expectation value accounting for Russian roulette doesn't change
        // -- that our estimate is still unbiased.
        path_segments.throughput[index] /= survival_prob;
    }

    Math::Ray ray;
    ray.origin = intersection.pos + Math::EPSILON * sample.w_in;
    ray.direction = sample.w_in;

    path_segments.ray[index] = ray;
    remaining_bounces--;

    if (remaining_bounces > 0) {
        uint32_t idx = atomicAdd(&counters->extension_count, 1u);
        extension_queue[idx] = index;
    }
}

__global__ void ShadowRayKernel(
    DeviceScene scene,
    PathSegments path_segments,
    uint32_t num_paths,
    uint32_t *shadow_ray_queue,
    QueueCounters *counters
) {
    int queue_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (queue_index >= counters->emissive_hit_count) return;

    int index = shadow_ray_queue[queue_index];
    if (index >= num_paths) return;

    glm::vec3 w_out = -glm::normalize(path_segments.ray[index].direction);

    Intersection &last_surface_intersect = path_segments.intersection[index];
    Material &material = scene.materials[last_surface_intersect.material_id];

    LightSample &light_sample = path_segments.light_sample[index];
    glm::vec3 w_in = glm::normalize(light_sample.position - last_surface_intersect.pos);
    float dist = glm::distance(light_sample.position, last_surface_intersect.pos);

    float tmax = dist - Math::EPSILON;
    Math::Ray ray;
    ray.origin = last_surface_intersect.pos + w_in * Math::EPSILON;
    ray.direction = w_in;

    // Check if this light is not occluded
    Intersection intersection;
    bool hit = IntersectScene<IntersectQueryMode::ANY>(scene, ray, intersection, tmax);
    if (hit) {
        // Accumulate contribution due to area sampling strategy
        float cos_surf = glm::dot(intersection.normal, w_in);
        float cos_light = glm::dot(light_sample.normal, -w_in);
        if (cos_surf > 0.0f && cos_light > 0.0f) {
            glm::vec3 bsdf = EvalMaterialBSDF(material, scene.texture_pool, last_surface_intersect, w_in, w_out);
            float bsdf_pdf = MaterialPDF(material, scene.texture_pool, intersection, w_in, w_out);

            float light_pdf = light_sample.pdf * dist * dist / cos_light;
            float light_weight = Math::PowerHeuristic(light_pdf, bsdf_pdf);
            path_segments.accum[index] += path_segments.throughput[index] * light_weight * bsdf * cos_surf * light_sample.radiance / light_pdf;
        }
    }
}

__global__ void EmissiveHitKernel(
    DeviceScene scene,
    PathSegments path_segments,
    uint32_t num_paths,
    uint32_t *emissive_queue,
    QueueCounters *counters,
    glm::vec3 *image
) {
    int queue_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (queue_index >= counters->emissive_hit_count) return;

    int index = emissive_queue[queue_index];
    if (index >= num_paths) return;

    if (path_segments.emissive_type[index] == EmissiveType::Infinite) {
        // const glm::vec3 background_color = glm::vec3(0.7, 0.7, 0.9f);
        const glm::vec3 background_color = glm::vec3(0.0f);
        Spectrum radiance = background_color;

        path_segments.accum[index] += path_segments.throughput[index] * radiance;
    } else if (path_segments.emissive_type[index] == EmissiveType::Area) {
        Intersection &intersection = path_segments.intersection[index];
        Material &material = scene.materials[intersection.material_id];
        GeometryInstance &geometry = scene.geometries[intersection.geometry_id];
        
        glm::vec3 color = SampleTexture<glm::vec3>(scene.texture_pool, material.emissive.color_texture, intersection.uv);
        Spectrum radiance = material.emissive.emittance * color;

        // if (scene.max_depth == path_segments.remaining_bounces[index]) {
        //     path_segments.accum[index] = radiance;
        // }

        // Accumulate contribution due to BSDF sampling strategy
        float weight_bsdf = 1.0f;
        if (path_segments.last_pdf[index] > 0.0f) {
            glm::vec3 surface_point = path_segments.last_pos[index];
            glm::vec3 light_point = path_segments.intersection[index].pos;
            
            AreaLight &light = scene.lights[geometry.area_light_id];
            float dist2 = glm::distance2(surface_point, light_point);
            
            glm::vec3 w_in = glm::normalize(surface_point - light_point);
            float cos_light = glm::dot(intersection.normal, w_in);
            
            if (cos_light > 0.0f) {
                float light_pdf = (light.power / scene.total_light_power) * (1.0f / geometry.total_area) * dist2 / cos_light;
                weight_bsdf = Math::PowerHeuristic(path_segments.last_pdf[index], light_pdf);
            }
        }

        path_segments.accum[index] += path_segments.throughput[index] * radiance * weight_bsdf;
    }
}

__global__ void FinalGather(PathSegments path_segments, glm::ivec2 resolution, glm::vec3 *image) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= resolution.x || y >= resolution.y) return;
    
    uint32_t index = x + resolution.x * y;

    glm::vec3 accum = path_segments.accum[index];
    if (isnan(accum.r) || isnan(accum.g) || isnan(accum.b)) {
        if (threadIdx.x) {
            printf("WARNING: NANs detected!");
        }
        accum = glm::vec3(0.0f);
    }

    image[index] += accum;
}

__global__ void PostprocessAndUpload(uchar4 *dest, glm::ivec2 resolution, int iterations, const glm::vec3 *image) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + y * resolution.x;

        glm::vec3 radiance = image[index] / static_cast<float>(iterations + 1);

        // Luminance-preserving tone-mapping
        // float luminance = Math::Luminance(radiance);
        // float ldr = ACESFilmicToneMap(luminance);

        // float scale = ldr / glm::max(luminance, 1e-6f);
        // glm::vec3 pixel = Math::Saturate(radiance * scale);

        glm::vec3 pixel = Math::ACESFilmicToneMap(radiance);

        glm::ivec3 color;
        color.x = glm::clamp((int)(pixel.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pixel.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pixel.z * 255.0), 0, 255);

        dest[index] = make_uchar4(color.x, color.y, color.z, 255);
    }
}

void PathIntegrator::Render(glm::ivec2 resolution, const DeviceScene &scene, uchar4 *data, int iteration, const IntegratorState &state) {
    size_t num_pixels = resolution.x * resolution.y;
    if (iteration == 0)
        CUDA_CHECK(cudaMemset(m_image, 0, num_pixels * sizeof(glm::vec3)));

    dim3 block_size(8, 8);
    dim3 block_count(
        (resolution.x + block_size.x - 1) / block_size.x,
        (resolution.y + block_size.y - 1) / block_size.y
    );

    cudaMemset(m_queue_counters, 0, sizeof(QueueCounters));
    cudaMemset(m_extension_queue, 0, sizeof(uint32_t) * num_pixels);

    GenerateRaysKernel<<<block_count, block_size>>>(scene, iteration, resolution, m_path_segements, m_extension_queue, m_queue_counters);
    
    uint32_t path_count = m_num_paths;
    for (uint32_t i = 0; i < scene.max_depth; ++i) {
        dim3 launch_count = (path_count + block_size_1d - 1) / block_size_1d;
        QueueCounters host_counters;
        CUDA_CHECK(cudaMemcpy(&path_count, &m_queue_counters->extension_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        if (path_count > 0) {
            IntersectClosestKernel<<<launch_count, block_size_1d>>>(scene, m_path_segements, m_num_paths, m_queue_counters, m_extension_queue, m_emissive_hit_queue, m_lambertian_queue, m_mirror_queue, m_metallic_roughness_queue, m_dielectric_queue);
            cudaMemset(&m_queue_counters->extension_count, 0, sizeof(uint32_t));

            MaterialShadeKernel<Material::Type::Lambertian><<<launch_count, block_size_1d>>>(scene, m_path_segements, m_num_paths, m_lambertian_queue, m_extension_queue, m_shadow_ray_queue, m_terminated_queue, m_queue_counters, m_image);
            MaterialShadeKernel<Material::Type::Mirror><<<launch_count, block_size_1d>>>(scene, m_path_segements, m_num_paths, m_mirror_queue, m_extension_queue, m_shadow_ray_queue, m_terminated_queue, m_queue_counters, m_image);
            MaterialShadeKernel<Material::Type::MetallicRoughness><<<launch_count, block_size_1d>>>(scene, m_path_segements, m_num_paths, m_metallic_roughness_queue, m_extension_queue, m_shadow_ray_queue, m_terminated_queue, m_queue_counters, m_image);
            MaterialShadeKernel<Material::Type::Dielectric><<<launch_count, block_size_1d>>>(scene, m_path_segements, m_num_paths, m_dielectric_queue, m_extension_queue, m_shadow_ray_queue, m_terminated_queue, m_queue_counters, m_image);
            cudaMemset(&m_queue_counters->lambertian_count, 0, sizeof(uint32_t));
            cudaMemset(&m_queue_counters->mirror_count, 0, sizeof(uint32_t));
            cudaMemset(&m_queue_counters->metallic_roughness_count, 0, sizeof(uint32_t));
            cudaMemset(&m_queue_counters->dielectric_count, 0, sizeof(uint32_t));
            
            cudaDeviceSynchronize();

            EmissiveHitKernel<<<launch_count, block_size_1d>>>(scene, m_path_segements, m_num_paths, m_emissive_hit_queue, m_queue_counters, m_image);
            cudaMemset(&m_queue_counters->emissive_hit_count, 0, sizeof(uint32_t));
        }
    }

    FinalGather<<<block_count, block_size>>>(m_path_segements, resolution, m_image);

    PostprocessAndUpload<<<block_count, block_size>>>(data, resolution, iteration, m_image);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}
