#pragma once

#include "MathUtils.h"
#include "settings.h"
#include <functional>
#include "geometry.h"
#include <glm/glm.hpp>
#include <vector>

#if USE_BVH

// Given geometry, verts, and indices, create bounding volume hierarchy
namespace BVH {
    struct Bin {
        AABB bounds;
        uint32_t tri_count;
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
        std::vector<AABB> m_aabbs;

        const Geometry &m_geometry;
        const std::vector<MeshVertex> &m_vertices;
        const std::vector<uint32_t> &m_indices;
    private:
        void UpdateNodeAABB(uint32_t node);
        void Subdivide(uint32_t node);
        void ForEachTri(std::function<void(uint32_t, const MeshVertex &, const MeshVertex &, const MeshVertex &)> callback);
    
        float ComputeSplitPlane(BVHNode &node, uint32_t &best_axis, float &best_pos);
        float EvaluateSAH(BVHNode &node, uint32_t axis, float pos);
    };

}
#endif
