#pragma once

#include "Intersections.h"
#include "bvh.h"
#include "settings.h"
#include "settings.h"
#include "scene.h"
#include <thrust/device_ptr.h>
#include <vector>
#include <vector_types.h>
#include <volk.h>
#include "GLTFModel.h"

struct PathTracerProperties;

class PathTracer {
public:
    PathTracer(const VkExtent2D &extent, Scene &scene);
    ~PathTracer();

    void Init();
    void Destroy();

    void Reset();

    void PathTrace(uchar4 *pbo, int frame, int iteration);
private:
    Scene &m_scene;

    const VkExtent2D &m_extent;
    glm::vec3 *md_image = nullptr;

    PathSegment *md_paths = nullptr;
    thrust::device_ptr<PathSegment> m_paths_thrust;
    
    PathSegment *md_terminated = nullptr;
    thrust::device_ptr<PathSegment> m_terminated_thrust;

#if SORT_BY_MATERIAL
    int *md_keys = nullptr;
    thrust::device_ptr<int> m_keys_thrust;
#endif

    Intersection *md_intersections = nullptr;
    thrust::device_ptr<Intersection> m_intersection_thrust;
};
