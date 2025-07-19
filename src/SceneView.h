#pragma once

#define GLM_ENABLE_EXPERIMENTAL 
#include <glm/gtx/norm.hpp>
#include "Light.h"
#include "geometry.h"
#include "material.h"
#include "samplers.h"
#include "bvh.h"

// Device view of `Scene`
struct SceneView {
    // Triangle Geometry ----
    MeshVertex *vertices = nullptr;
    uint32_t vertex_count = 0;

    uint32_t *indices = nullptr;
    uint32_t index_count = 0;
    
    AreaLight *lights = nullptr;
    uint32_t light_count = 0;

    // BVH ----
// #if USE_BVH
    BVH::BVHNode *bvh_nodes = nullptr;
    uint32_t bvh_node_count = 0;
    
    uint32_t *bvh_tri_indices = nullptr;
    uint32_t bvh_indices_count = 0;
// #endif

    // Primitives and materials ----
    Geometry *geometries = nullptr;
    uint32_t geometry_count = 0;

    Material *materials = nullptr;
    uint32_t material_count = 0;

    float total_light_power = 0.0f;

    DiscreteSampler1DView light_sampler {};

    inline __device__ float AnyHit(const Ray &r, Intersection &intersection, float tmax = FLT_MAX) {
        for (int i = 0; i < geometry_count; ++i) {
            float t = -1.0f;
            Intersection temp{};

            if (geometries[i].type == GeometryType::Cube) {
                t = BoxIntersectionTest(geometries[i], r, temp);
            } else if (geometries[i].type == GeometryType::Sphere) {
                t = SphereIntersectionTest(geometries[i], r, temp);
            } else if (geometries[i].type == GeometryType::TriangleMesh) {
#if USE_BVH
                t = IntersectBVHClosest(geometries[i], *this, r, temp);
#else
                t = NaivePrimitiveIntersection(geometries[i], *this, r, temp);
#endif
            }

            if (t > 0.0f && t <= tmax) {
                temp.t = t;
                temp.material_id = geometries[i].material_id;
                intersection = temp;
                return t;
            }
        }

        intersection.t = -1.0f;
        return -1.0f;
    }

    inline __device__ float ClosestHit(const Ray &r, Intersection &intersection, float tmax = FLT_MAX) {
        float t_min = FLT_MAX;
        int hit_index = -1;
        Intersection best_intersection {};

        for (int i = 0; i < geometry_count; ++i) {
            float t = -1.0f;
            Intersection temp {};

            if (geometries[i].type == GeometryType::Cube) {
                t = BoxIntersectionTest(geometries[i], r, temp);
            } else if (geometries[i].type == GeometryType::Sphere) {
                t = SphereIntersectionTest(geometries[i], r, temp);
            } else if (geometries[i].type == GeometryType::TriangleMesh) {
#if USE_BVH
                t = IntersectBVHClosest(geometries[i], *this, r, temp);
#else
                t = NaivePrimitiveIntersection(geometries[i], *this, r, temp);
#endif
            }

            if (t > 0.0f && t < t_min && t < tmax) {
                t_min = t;
                hit_index = i;
                best_intersection = temp;
            }
        }

        if (hit_index < 0) {
            intersection.t = -1.0f;
            return -1.0f;
        }

        intersection = best_intersection;
        intersection.t = t_min;
        intersection.material_id = geometries[hit_index].material_id;

        return t_min;
    }

    inline __device__ LightSample SampleLight(const Intersection &intersection, thrust::default_random_engine &rng) {
        uint32_t light_id = light_sampler.Sample(rng);
        AreaLight &light = lights[light_id];

        float p_pick = light.power / total_light_power;
        if (light.geometry_id == -1)
            return {};
        Geometry &geometry = geometries[light.geometry_id];
        
        switch (geometry.type) {
            case GeometryType::Cube: {
                return {};
            }
            case GeometryType::Sphere: {
                return {};
            }
            case GeometryType::TriangleMesh: {
                uint32_t tri = geometry.triangle_sampler.Sample(rng);

                uint32_t i0 = indices[geometry.mesh.first_index + tri*3];
                uint32_t i1 = indices[geometry.mesh.first_index + tri*3 + 1];
                uint32_t i2 = indices[geometry.mesh.first_index + tri*3 + 2];
                
                glm::vec3 p0 = geometry.transform * glm::vec4(vertices[i0].position, 1.0f);
                glm::vec3 p1 = geometry.transform * glm::vec4(vertices[i1].position, 1.0f);
                glm::vec3 p2 = geometry.transform * glm::vec4(vertices[i2].position, 1.0f);

                glm::vec2 rand = Math::Sample2D(rng);
                
                float sqr = sqrt(rand.x);
                float u = 1.0f - sqr;
                float v = sqr * (1.0f - rand.y);
                float w = sqr * rand.y;

                glm::vec3 pos = u*p0 + v*p1 + w*p2;

                glm::vec3 normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
                Material &m = materials[geometry.material_id];

                LightSample light_sample;
                light_sample.radiance = m.mat.light.base_color * m.mat.light.emittance;
                light_sample.normal = normal;
                light_sample.position = pos;

                // Probability of choosing light * probability of choosing this point on surface
                light_sample.pdf = p_pick * (1.0f / light.area);

                return light_sample;
            }
        }
    }
};
