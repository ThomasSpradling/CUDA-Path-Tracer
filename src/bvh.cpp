#include "bvh.h"
#if USE_BVH

namespace BVH {

    BVH::BVH(const Geometry &geometry, const std::vector<MeshVertex> &vertices, const std::vector<uint32_t> &indices)
        : m_geometry(geometry)
        , m_vertices(vertices)
        , m_indices(indices)
    {}

    void BVH::Build() {
        uint32_t prim_count = m_geometry.mesh.index_count / 3u;
        if (prim_count == 0)
            return;
        
        m_tri_indices.resize(prim_count);
        for (uint32_t i = 0; i < prim_count; ++i)
            m_tri_indices[i] = i;
        
        m_centroids.resize(prim_count);
        ForEachTri([&](uint32_t i, const MeshVertex &v0, const MeshVertex &v1, const MeshVertex &v2) {
            m_centroids[i] = (v0.position + v1.position + v2.position) * 0.3333f;
        });
        
        m_bvh_nodes.clear();
        m_bvh_nodes.reserve(2 * prim_count - 1);
        m_bvh_nodes.emplace_back();

        BVHNode &root = m_bvh_nodes[0];
        root.left_child = UINT32_MAX;
        root.first_tri = 0;
        root.tri_count = prim_count;
        UpdateNodeAABB(0);
        Subdivide(0);
    }

    void BVH::Subdivide(uint32_t n) {
        BVHNode& node = m_bvh_nodes[n];
        if (node.tri_count <= 2)
            return;

        glm::vec3 extent = node.bounds.max - node.bounds.min;
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent[axis]) axis = 2;
        float splitPos = node.bounds.min[axis] + 0.5f * extent[axis];

        uint32_t i = node.first_tri;
        uint32_t j = i + node.tri_count - 1;
        while (i <= j) {
            uint32_t tri = m_tri_indices[i];
            if (m_centroids[tri][axis] < splitPos) ++i;
            else std::swap(m_tri_indices[i], m_tri_indices[j--]);
        }
        uint32_t leftCount = i - node.first_tri;
        if (leftCount == 0 || leftCount == node.tri_count)
            return;

        uint32_t left_index  = (uint32_t)m_bvh_nodes.size();
        m_bvh_nodes.emplace_back();
        m_bvh_nodes.emplace_back();
        uint32_t right_index = left_index + 1;

        BVHNode &left  = m_bvh_nodes[left_index];
        left.first_tri = node.first_tri;
        left.tri_count = leftCount;
        left.left_child  = UINT32_MAX;
        UpdateNodeAABB(left_index);

        BVHNode &right = m_bvh_nodes[right_index];
        right.first_tri = i;
        right.tri_count = node.tri_count - leftCount;
        right.left_child  = UINT32_MAX;
        UpdateNodeAABB(right_index);

        node.left_child  = left_index;
        node.tri_count = 0;

        Subdivide(left_index);
        Subdivide(right_index);
    }

    void BVH::UpdateNodeAABB(uint32_t n) {
        BVHNode& node = m_bvh_nodes[n];
        node.bounds.min = glm::vec3( 1e30f);
        node.bounds.max = glm::vec3(-1e30f);

        const uint32_t base_index = m_geometry.mesh.first_index;
        for (uint32_t i = 0; i < node.tri_count; ++i) {
            uint32_t tri = m_tri_indices[node.first_tri + i];
            uint32_t i0 = m_indices[base_index + tri*3 + 0];
            uint32_t i1 = m_indices[base_index + tri*3 + 1];
            uint32_t i2 = m_indices[base_index + tri*3 + 2];

            node.bounds.min = glm::min(node.bounds.min, m_vertices[i0].position);
            node.bounds.min = glm::min(node.bounds.min, m_vertices[i1].position);
            node.bounds.min = glm::min(node.bounds.min, m_vertices[i2].position);

            node.bounds.max = glm::max(node.bounds.max, m_vertices[i0].position);
            node.bounds.max = glm::max(node.bounds.max, m_vertices[i1].position);
            node.bounds.max = glm::max(node.bounds.max, m_vertices[i2].position);
        }
    }

    void BVH::ForEachTri(std::function<void(uint32_t, const MeshVertex &, const MeshVertex &, const MeshVertex &)> callback) {
        uint32_t prim_count = m_geometry.mesh.index_count / 3u;
        
        for (uint32_t i = 0; i < prim_count; ++i) {
            uint32_t i0 = m_indices[m_geometry.mesh.first_index + i*3 + 0];
            uint32_t i1 = m_indices[m_geometry.mesh.first_index + i*3 + 1];
            uint32_t i2 = m_indices[m_geometry.mesh.first_index + i*3 + 2];
            const auto &v0 = m_vertices[i0];
            const auto &v1 = m_vertices[i1];
            const auto &v2 = m_vertices[i2];

            callback(i, v0, v1, v2);
        }
    };

}
#endif
