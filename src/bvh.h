#pragma once

#include "MathUtils.h"
#include "settings.h"
#include <functional>
#if USE_BVH
#include "scene.h"
#include <glm/glm.hpp>
#include <vector>
// Given geometry, verts, and indices, create bounding volume hierarchy

namespace BVH {
    struct Triangle {
        MeshVertex v0;
        MeshVertex v1;
        MeshVertex v2;
    };

    struct BVHNode {
        AABB bounds;
        uint32_t left_child;
        uint32_t first_tri, tri_count;

        __host__ __device__ inline bool IsLeaf() const { return tri_count > 0; }
        __host__ __device__ uint32_t RightChild() const { return left_child + 1; }
    };

    // Building BVH fills up m_bvh_nodes
    class BVH {
    public:
        BVH(const Geometry &geometry, const std::vector<MeshVertex> &vertices, const std::vector<uint32_t> &indices);

        void Build();
        const std::vector<BVHNode> &Nodes() const { return m_bvh_nodes; }
        const std::vector<uint32_t> &TriIndices() const { return m_tri_indices; }
    private:
        std::vector<BVHNode> m_bvh_nodes;
        std::vector<uint32_t> m_tri_indices;

        std::vector<glm::vec3> m_centroids;

        const Geometry &m_geometry;
        const std::vector<MeshVertex> &m_vertices;
        const std::vector<uint32_t> &m_indices;
    private:
        void UpdateNodeAABB(uint32_t node);
        void Subdivide(uint32_t node);
        void ForEachTri(std::function<void(uint32_t, const MeshVertex &, const MeshVertex &, const MeshVertex &)> callback);
    };

}
#endif
