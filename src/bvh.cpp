#include "bvh.h"
#include <array>
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

        m_aabbs.resize(prim_count);
        for(auto &b : m_aabbs) 
            b = AABB();
        ForEachTri([&](uint32_t i, const MeshVertex &v0, const MeshVertex &v1, const MeshVertex &v2) {
            m_aabbs[i].AddPoint(v0.position);
            m_aabbs[i].AddPoint(v1.position);
            m_aabbs[i].AddPoint(v2.position);
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
        BVHNode &node = m_bvh_nodes[n];
        if (node.tri_count <= 2)
            return;

        uint32_t axis;
        float split_pos;
        float cost = ComputeSplitPlane(node, axis, split_pos);
        
#if BVH_USE_SAH
        float parent_cost = node.tri_count * node.bounds.SurfaceArea();
        if (cost >= parent_cost)
            return;
#endif

        // Partition primitive (indices) by the splitting pane. Resulting
        // i is location of splitting pane
        uint32_t i = node.first_tri;
        uint32_t j = i + node.tri_count - 1;
        while (i <= j) {
            uint32_t tri = m_tri_indices[i];
            if (m_centroids[tri][axis] < split_pos) ++i;
            else std::swap(m_tri_indices[i], m_tri_indices[j--]);
        }
        uint32_t left_count = i - node.first_tri;
        if (left_count == 0 || left_count == node.tri_count)
            return;

        // If splitting pane in non-trivial position, create
        // each child node
        uint32_t left_index = m_bvh_nodes.size();
        m_bvh_nodes.emplace_back();
        m_bvh_nodes.emplace_back();
        uint32_t right_index = left_index + 1;

        BVHNode &left = m_bvh_nodes[left_index];
        left.first_tri = node.first_tri;
        left.tri_count = left_count;
        left.left_child = UINT32_MAX;
        UpdateNodeAABB(left_index);

        BVHNode &right = m_bvh_nodes[right_index];
        right.first_tri = i;
        right.tri_count = node.tri_count - left_count;
        right.left_child  = UINT32_MAX;
        UpdateNodeAABB(right_index);

        node.left_child = left_index;
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

    float BVH::ComputeSplitPlane(BVHNode &node, uint32_t &best_axis, float &best_pos) {
#if BVH_USE_SAH
        constexpr int bin_count = 8;
        float best_cost = FLT_MAX;
        
        AABB centroid_bounds;
        for (uint32_t i = 0; i < node.tri_count; ++i) {
            uint32_t tri = m_tri_indices[node.first_tri + i];
            centroid_bounds.AddPoint(m_centroids[tri]);
        }

        if (centroid_bounds.IsEmpty())
            return best_cost;

        std::array<Bin, bin_count> bins {};
        glm::vec3 centroid_extent = centroid_bounds.Extent();
        glm::vec3 bin_width = centroid_extent / static_cast<float>(bin_count);

        for (uint32_t axis = 0; axis < 3; ++axis) {
            if (centroid_extent[axis] <= 0.0f)
                continue;

            for (auto &b : bins) {
                b.tri_count = 0;
                b.bounds = AABB();
            }

            for (uint32_t i = 0; i < node.tri_count; ++i) {
                uint32_t tri = m_tri_indices[node.first_tri + i];

                // Which bin does this tri belong in?
                float scaled_offset = (m_centroids[tri][axis] - centroid_bounds.min[axis]) / bin_width[axis];
                int bin_index = glm::clamp(static_cast<int>(scaled_offset), 0, bin_count - 1);

                bins[bin_index].tri_count++;
                bins[bin_index].bounds.Union(m_aabbs[tri]);
            }

            // Compute areas and primitive counts on each side of
            // each candidate splitting plane
            std::array<float, bin_count - 1> left_area, right_area {};
            std::array<uint32_t, bin_count - 1> left_count, right_count {};

            AABB left_box, right_box;
            int left_sum = 0, right_sum = 0;
            for (int i = 0; i < bin_count - 1; ++i) {
                left_sum += bins[i].tri_count;
                left_count[i] = left_sum;
                left_box.Union(bins[i].bounds);
                left_area[i] = left_box.SurfaceArea();

                const uint32_t right_bin = bin_count - 1 - i;
                right_sum += bins[right_bin].tri_count;
                right_count[right_bin - 1] = right_sum;
                right_box.Union(bins[right_bin].bounds);
                right_area[right_bin - 1] = right_box.SurfaceArea();
            }

            for (uint32_t i = 0; i < bin_count - 1; ++i) {
                float plane_cost = left_count[i] * left_area[i] + right_count[i] * right_area[i];
                if (plane_cost < best_cost) {
                    best_axis = axis;
                    best_pos = centroid_bounds.min[axis] + bin_width[axis] * (i + 1);
                    best_cost = plane_cost;
                }
            }
        }
        return best_cost;
#else
        glm::vec3 extent = node.bounds.max - node.bounds.min;
        best_axis = 0;
        if (extent.y > extent.x) best_axis = 1;
        if (extent.z > extent[best_axis]) best_axis = 2;
        best_pos = node.bounds.min[best_axis] + 0.5f * extent[best_axis];
        return 0.0f;
#endif
    }

#if BVH_USE_SAH
    float BVH::EvaluateSAH(BVHNode &node, uint32_t axis, float pos) {
        AABB left_box {}, right_box {};
        int left_count = 0, right_count = 0;
        for (uint32_t i = 0; i < node.tri_count; ++i) {
            uint32_t tri = m_tri_indices[node.first_tri + i];
            if (m_centroids[tri][axis] < pos) {
                left_count++;
                left_box.Union(m_aabbs[tri]);
            } else {
                right_count++;
                right_box.Union(m_aabbs[tri]);
            }
        }

        float cost = left_count * left_box.SurfaceArea() + right_count * right_box.SurfaceArea();
        return cost > 0 ? cost : FLT_MAX;
    }
#endif

}
#endif
