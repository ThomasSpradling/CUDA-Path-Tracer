#pragma once

#include "scene.h"
#include <vector_types.h>
#include <volk.h>

class PathTracer {
public:
    PathTracer(const VkExtent2D &extent, Scene &scene);
    ~PathTracer();

    void Init();
    void Destroy();

    void PathTrace(uchar4 *pbo, int frame, int iteration);
private:
    Scene &m_scene;

    const VkExtent2D &m_extent;
    glm::vec3 *md_image = nullptr;
    Geometry *md_geometries = nullptr;
    Material *md_materials = nullptr;
    PathSegment *md_paths = nullptr;
    ShadableIntersection *md_intersections = nullptr;
};
