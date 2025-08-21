#pragma once

#include "../utils/cl_happy.h"
#include <glm/glm.hpp>

namespace Math {

    struct AABB {
        glm::vec3 min {FLT_MAX};   
        glm::vec3 max {-FLT_MAX};

        __host__ __device__ __forceinline__ void Reset() {
            min = glm::vec3(FLT_MAX);
            max = glm::vec3(-FLT_MAX);
        }

        __host__ __device__ __forceinline__ void AddPoint(const glm::vec3 &point) {
            min = glm::min(min, point);
            max = glm::max(max, point);
        }

        __host__ __device__ __forceinline__ void Union(const AABB &other) {
            min = glm::min(min, other.min);
            max = glm::max(max, other.max);
        }

        __host__ __device__ __forceinline__ bool IsEmpty() const {
            return (min.x > max.x) || (min.y > max.y) || (min.z > max.z);
        }

        __host__ __device__ __forceinline__ glm::vec3 Extent() const {
            return max - min;
        }

        __host__ __device__ __forceinline__ float SurfaceArea() const {
            glm::vec3 d = max - min;
            d = glm::max(d, glm::vec3(0.0f));
            return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
        }

        __host__ __device__ __forceinline__ float Volume() const {
            glm::vec3 d = max - min;
            d = glm::max(d, glm::vec3(0.0f));
            return d.x * d.y * d.z;
        }

        __host__ __device__ __forceinline__ glm::vec3 Centroid() const {
            return (max + min) * 0.5f;
        }

        __host__ __device__ AABB Transform(const glm::mat4 &transform) const {
            glm::vec3 minv = min, maxv = max;
            Math::AABB out;
            for (int c = 0; c < 8; ++c) {
                const glm::vec3 p = {
                    (c & 1) ? maxv.x : minv.x,
                    (c & 2) ? maxv.y : minv.y,
                    (c & 4) ? maxv.z : minv.z
                };
                out.AddPoint(glm::vec3(transform * glm::vec4(p, 1.f)));
            }
            return out;
        }
    };

}
