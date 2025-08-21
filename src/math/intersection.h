#pragma once

#include "../utils/cl_happy.h"
#include "aabb.h"
#include "constants.h"
#include "ray.h"

namespace Math {
    __host__ __device__ __forceinline__ bool IntersectAABB(const Ray &ray, const AABB &aabb, float &t_out, float t_max) {
        const glm::vec3 inv_d = 1.0f / ray.direction;

        const glm::vec3 aabb_min = aabb.min;
        const glm::vec3 aabb_max = aabb.max;
        const glm::vec3 origin = ray.origin;

        float t0 = (aabb_min.x - origin.x) * inv_d.x;
        float t1 = (aabb.max.x - origin.x) * inv_d.x;
        float tnear = fminf(t0, t1);
        float tfar = fmaxf(t0, t1);

        t0 = (aabb_min.y - origin.y) * inv_d.y;
        t1 = (aabb.max.y - origin.y) * inv_d.y;
        tnear = fmaxf(tnear, fminf(t0, t1));
        tfar = fminf(tfar, fmaxf(t0, t1));

        t0 = (aabb_min.z - origin.z) * inv_d.z;
        t1 = (aabb.max.z - origin.z) * inv_d.z;
        tnear = fmaxf(tnear, fminf(t0, t1));
        tfar = fminf(tfar, fmaxf(t0, t1));

        const bool hit = (tfar >= tnear) && (tnear < t_max) && (tfar > 0.0f);
        if (hit) t_out = tnear;

        return hit;
    }
    
    __host__ __device__ __forceinline__ bool IntersectTriangle(
        const Ray &ray,
        const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2,
        glm::vec2 &uv, float &t_out, float t_max
    ) {
        t_out = -1.0f;

        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 p = glm::cross(ray.direction, edge2);
        float a = glm::dot(edge1, p);
        if (fabs(a) < Math::SMALL_EPSILON)
            return false;
        float inv_a = 1.0f / a;

        glm::vec3 tvec = ray.origin - v0;
        float u = glm::dot(tvec, p) * inv_a;
        if (u < 0.0f || u > 1.0f)
            return false;

        glm::vec3 q = glm::cross(tvec, edge1);
        float v = glm::dot(ray.direction, q) * inv_a;
        if (v < 0.0f || u + v > 1.0f)
            return false;

        t_out = glm::dot(edge2, q) * inv_a;
        if (t_out <= Math::SMALL_EPSILON)
            return false;

        uv = glm::vec2(u, v);
        if (t_out > t_max)
            return false;

        return true;
    }
}
