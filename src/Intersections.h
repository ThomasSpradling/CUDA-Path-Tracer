#pragma once

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
