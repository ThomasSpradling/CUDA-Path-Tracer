#pragma once

#include "intersection.h"
#include "sampler.h"
#include "../math/sampling.h"
#include "texture.h"

struct LightSample {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 radiance;
    float pdf;
};

__device__ __forceinline__ LightSample SampleSceneLight(const DeviceScene &scene, const Intersection &intersection, Sampler &sampler) {
    uint32_t light_id = DiscreteSample1D(*scene.light_sampler, sampler);
    AreaLight &light = scene.lights[light_id];
    if (light.geometry_instance_index == -1)
        return {};
    
    float p_pick = light.power / scene.total_light_power;
    GeometryInstance &geometry = scene.geometries[light.geometry_instance_index];

    DeviceDiscreteSampler1D &triangle_sampler = scene.triangle_samplers[geometry.triangle_sampler_index];
    uint32_t tri = DiscreteSample1D(triangle_sampler, sampler);
    
    uint32_t i0 = scene.indices[geometry.triangle_mesh.first_index + tri*3 + 0];
    uint32_t i1 = scene.indices[geometry.triangle_mesh.first_index + tri*3 + 1];
    uint32_t i2 = scene.indices[geometry.triangle_mesh.first_index + tri*3 + 2];

    glm::vec3 p0 = geometry.transform * glm::vec4(scene.positions[i0], 1.0f);
    glm::vec3 p1 = geometry.transform * glm::vec4(scene.positions[i1], 1.0f);
    glm::vec3 p2 = geometry.transform * glm::vec4(scene.positions[i2], 1.0f);

    glm::vec3 n0 = geometry.inv_transpose * glm::vec4(scene.normals[i0], 0.0f);
    glm::vec3 n1 = geometry.inv_transpose * glm::vec4(scene.normals[i1], 0.0f);
    glm::vec3 n2 = geometry.inv_transpose * glm::vec4(scene.normals[i2], 0.0f);

    glm::vec2 texcoord0 = scene.uvs[i0];
    glm::vec2 texcoord1 = scene.uvs[i1];
    glm::vec2 texcoord2 = scene.uvs[i2];
    
    glm::vec3 uvw = Math::SampleTriangle(Sample2D(sampler));
    float u = uvw.x, v = uvw.y, w = uvw.z;
    
    glm::vec3 pos = u*p0 + v*p1 + w*p2;
    glm::vec3 normal = u*n0 + v*n1 + w*n2;
    glm::vec2 texcoord = u*texcoord0 + v*texcoord1 + w*texcoord2;

    Material &material = scene.materials[geometry.material_id];

    LightSample light_sample;
    light_sample.radiance = SampleTexture<glm::vec3>(scene.texture_pool, material.emissive.color_texture, texcoord) * material.emissive.emittance;
    light_sample.normal = normal;
    light_sample.position = pos;
    light_sample.pdf = p_pick * (1.0f / geometry.total_area);

    return light_sample;
}
