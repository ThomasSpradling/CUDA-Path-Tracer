#pragma once

#include "bvh.h"
#include "settings.h"
#include <cuda_runtime.h>

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL 
#include <glm/gtx/intersect.hpp>
#include "scene.h"

#include <thrust/random.h>

__host__ __device__ float BoxIntersectionTest(
    const Geometry &box,
    const Ray &r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

__host__ __device__
float SphereIntersectionTest(
    const Geometry& sphere,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

__host__ __device__
float TriangleIntersectionTest(
    const Ray &ray,
    const glm::vec3 &v0,
    const glm::vec3 &v1,
    const glm::vec3 &v2,
    glm::vec2 &uv);

__host__ __device__
float NaivePrimitiveIntersection(
    const Geometry &geom,
    const MeshVertex *vertices,
    const uint32_t *indices,
    const Ray &r,
    glm::vec3 &intersection_point,
    glm::vec3 &normal,
    bool &outside);

#if USE_BVH
__device__ float IntersectBVH(
    const Geometry &geom,
    const BVH::BVHNode *nodes,
    const uint32_t *triangle_index_map,
    const MeshVertex *vertex_array,
    const uint32_t *index_array,
    const Ray &r,
    glm::vec3 &hit_p,
    glm::vec3 &hit_n,
    bool &outside);

#endif
