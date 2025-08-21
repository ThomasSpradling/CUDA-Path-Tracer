#pragma once

#include "kernel/device_scene.h"
#include "kernel/device_types.h"
#include <vector>

class Scene;

enum class SplitStrategy { SAH, Midpoint, Median };

struct Bin {
    Math::AABB bounds {};
    uint32_t prims {};
};

struct Primitive {
    Math::AABB aabb;
    glm::vec3 centroid;
};

class BLAS {
public:
    void Build(
        const Scene &scene,
        const TriangleMesh &triangle_mesh,
        SplitStrategy split_type
    );

    // object space bounds
    Math::AABB Bounds();

    const std::vector<BVHNode> &Nodes() const { return m_nodes; }
    const std::vector<uint32_t> &TriIndices() const { return m_tri_indices; }
private:
    std::vector<BVHNode> m_nodes;
    std::vector<uint32_t> m_tri_indices;

    std::vector<Primitive> m_primitives;

    SplitStrategy m_split_strategy;

    uint32_t m_current_node = 0;
private:
    void UpdateNodeBounds(uint32_t index);
    void Subdivide(uint32_t index);
    std::pair<uint32_t, float> FindSplitPlane(uint32_t index, float &cost);
};

class TLAS {
public:
    void Build(const Scene &scene, const std::vector<GeometryInstance> &instances, SplitStrategy split_type = SplitStrategy::SAH);
    void UpdateDevice(DeviceScene &scene);
    void FreeDevice(DeviceScene &scene);

    bool &Dirty() { return m_dirty; }
private:
    bool m_dirty = false;

    std::vector<TLASNode> m_tlas_nodes;

    std::vector<DeviceBLAS> m_device_blases;
    std::vector<BLAS> m_blases;

    std::vector<BVHNode> m_blas_nodes;
    std::vector<uint32_t> m_tri_indices;
};
