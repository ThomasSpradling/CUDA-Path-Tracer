#pragma once
#include <glm/glm.hpp>
#include "device_types.h"

struct DeviceScene {
    uint32_t max_depth;

    glm::vec3 *positions;
    glm::vec3 *normals;
    glm::vec2 *uvs;
    uint32_t vertex_count;

    uint32_t *indices;
    uint32_t index_count;
    
    GeometryInstance *geometries;
    uint32_t geometry_count;

    DeviceDiscreteSampler1D *triangle_samplers;
    uint32_t triangle_sampler_count;

    DeviceTLAS *tlas;

    Material *materials;
    uint32_t material_count;

    AreaLight *lights;
    uint32_t light_count;
    DeviceDiscreteSampler1D *light_sampler;

    DeviceTexturePool texture_pool;
    DeviceCamera camera;

    float total_light_power;
};
