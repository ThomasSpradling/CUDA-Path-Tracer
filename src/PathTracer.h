#pragma once

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
    Geometry *md_geometries = nullptr;
    Material *md_materials = nullptr;

    PathSegment *md_paths = nullptr;
    thrust::device_ptr<PathSegment> m_paths_thrust;
    
    PathSegment *md_terminated = nullptr;
    thrust::device_ptr<PathSegment> m_terminated_thrust;

#if SORT_BY_MATERIAL
    int *md_keys = nullptr;
    thrust::device_ptr<int> m_keys_thrust;
#endif

    ShadableIntersection *md_intersections = nullptr;
    thrust::device_ptr<ShadableIntersection> m_intersection_thrust;

    MeshVertex *md_vertices;
    uint32_t *md_indices;

#if USE_BVH
    std::vector<BVH::BVHNode> m_bvh_nodes;
    BVH::BVHNode *md_bvh_nodes;

    std::vector<uint32_t> m_bvh_tri_indices;
    uint32_t *md_bvh_tri_indices;
#endif
private:

};
