#pragma once

#include <glm/glm.hpp>
#include "common.h"
#include "integrator.h"

enum class EmissiveType {
    Area,
    Infinite
};

struct QueueCounters {
    uint32_t extension_count;
    uint32_t terminated_count;
    uint32_t emissive_hit_count;
    uint32_t shadow_ray_count;

    uint32_t lambertian_count;
    uint32_t mirror_count;
    uint32_t metallic_roughness_count;
    uint32_t dielectric_count;
};

struct PathSegments {
    Math::Ray *ray;
    LightSample *light_sample;
    float *last_pdf;

    glm::vec3 *last_pos;

    Spectrum *throughput;
    Spectrum *accum;
    int *pixel_index;
    int *remaining_bounces;

    Intersection *intersection;
    EmissiveType *emissive_type;

    Sampler *sampler;

    void Alloc(size_t count) {
        CUDA_CHECK(cudaMalloc((void **) &ray, sizeof(*ray) * count));
        CUDA_CHECK(cudaMalloc((void **) &light_sample, sizeof(*light_sample) * count));
        CUDA_CHECK(cudaMalloc((void **) &last_pdf, sizeof(*last_pdf) * count));
        CUDA_CHECK(cudaMalloc((void **) &last_pos, sizeof(*last_pos) * count));
        CUDA_CHECK(cudaMalloc((void **) &throughput, sizeof(*throughput) * count));
        CUDA_CHECK(cudaMalloc((void **) &accum, sizeof(*accum) * count));
        CUDA_CHECK(cudaMalloc((void **) &pixel_index, sizeof(*pixel_index) * count));
        CUDA_CHECK(cudaMalloc((void **) &remaining_bounces, sizeof(*remaining_bounces) * count));
        CUDA_CHECK(cudaMalloc((void **) &intersection, sizeof(*intersection) * count));
        CUDA_CHECK(cudaMalloc((void **) &emissive_type, sizeof(*emissive_type) * count));
        CUDA_CHECK(cudaMalloc((void **) &sampler, sizeof(*sampler) * count));
    }

    void Free() {
        CUDA_CHECK(cudaFree(ray));
        CUDA_CHECK(cudaFree(light_sample));
        CUDA_CHECK(cudaFree(last_pdf));
        CUDA_CHECK(cudaFree(last_pos));
        CUDA_CHECK(cudaFree(throughput));
        CUDA_CHECK(cudaFree(accum));
        CUDA_CHECK(cudaFree(pixel_index));
        CUDA_CHECK(cudaFree(remaining_bounces));
        CUDA_CHECK(cudaFree(intersection));
        CUDA_CHECK(cudaFree(emissive_type));
        CUDA_CHECK(cudaFree(sampler));

        ray = nullptr;
        light_sample = nullptr;
        last_pdf = nullptr;
        last_pos = nullptr;
        throughput = nullptr;
        accum = nullptr;
        pixel_index = nullptr;
        remaining_bounces = nullptr;
        intersection = nullptr;
        emissive_type = nullptr;
        sampler = nullptr;
    }
};

class PathIntegrator : public Integrator {
public:
    PathIntegrator();
    virtual ~PathIntegrator() override;

    virtual void Init(glm::ivec2 resolution) override;
    virtual void Destroy() override;
    virtual void Resize(glm::ivec2 resolution) override;
    virtual void Reset(glm::ivec2 resolution) override;

    virtual inline IntegratorType Type() override { return IntegratorType::Path; }

    virtual void Render(glm::ivec2 resolution, const DeviceScene &scene, uchar4 *data, int iteration, const IntegratorState &state) override;
private:
    glm::vec3 *m_image;
    DeviceScene *m_dev_scene;

    PathSegments m_path_segements;
    uint32_t m_num_paths;

    uint32_t *m_extension_queue;
    uint32_t *m_emissive_hit_queue;
    uint32_t *m_terminated_queue;

    uint32_t *m_lambertian_queue;
    uint32_t *m_mirror_queue;
    uint32_t *m_metallic_roughness_queue;
    uint32_t *m_dielectric_queue;

    QueueCounters *m_queue_counters;

    uint32_t *m_shadow_ray_queue;
private:
    bool LaunchDebugKernels(glm::ivec2 resolution, const DeviceScene &scene, uchar4 *data, int iteration, const IntegratorState &state);
};
