#pragma once

#include "../math/ray.h"
#include "sampler.h"
#include "device_types.h"
#include <glm/glm.hpp>

// Samples a point on the camera aperture using sample.zw and a point within a pixel (i, j) using sample.xy. Then computes
// and returns the ray extending from the sampled aperture point to the film point.
__device__ __forceinline__ Math::Ray SpawnCameraRay(const DeviceCamera &camera, int i, int j, const glm::vec4 &sample) {    
    float x = ((i + sample.x) - 0.5f * camera.resolution.x) * camera.pixel_length.x;
    float y = ((j + sample.y) - 0.5f * camera.resolution.y) * camera.pixel_length.y;

    // Change of basis to world space
    glm::vec3 film_point = camera.position
                         + camera.front * camera.focal_length
                         + camera.right * x
                         - camera.up * y;

    glm::vec3 direction = glm::normalize(film_point - camera.position);
    return Math::Ray{ camera.position, direction };
}