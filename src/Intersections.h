#pragma once

#include "geometry.h"
#include "settings.h"
#include <cuda_runtime.h>

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL 
#include <glm/gtx/intersect.hpp>

#include <thrust/random.h>

struct Intersection {
    float t = 0.0f;
    glm::vec3 pos {};
    glm::vec3 normal {};
    glm::vec2 uv {};
    int material_id = -1;
};

__host__ __device__ float BoxIntersectionTest(
    const Geometry &box,
    const Ray &r,
    Intersection &intersection);

__host__ __device__
float SphereIntersectionTest(
    const Geometry& sphere,
    const Ray& r,
    Intersection &intersection);

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
    const SceneView &scene,
    const Ray &r,
    Intersection &intersection);

#if USE_BVH
__device__ float IntersectBVHAny(
    const Geometry &geom,
    const SceneView &scene,
    const Ray &r,
    Intersection &intersection,
    float tmax = FLT_MAX
);

__device__ float IntersectBVHClosest(
    const Geometry &geom,
    const SceneView &scene,
    const Ray &r,
    Intersection &intersection,
    float tmax = FLT_MAX
);

#endif
