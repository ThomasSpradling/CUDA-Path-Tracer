#include "bvh.h"
#include "scene.h"
#include <algorithm>

void BLAS::Build(
    const Scene &scene,
    const TriangleMesh &triangle_mesh,
    SplitStrategy split_type
) {
    m_current_node = 0;
    m_split_strategy = split_type;

    m_nodes.clear();
    m_tri_indices.clear();

    const uint32_t primitive_count = triangle_mesh.index_count / 3u;
    if (primitive_count == 0) return;
    
    m_nodes.reserve(2 * primitive_count - 1);
    m_tri_indices.resize(primitive_count);

    for (uint32_t i = 0; i < primitive_count; ++i)
        m_tri_indices[i] = i;

    m_primitives.resize(primitive_count);

    for (uint32_t t = 0; t < primitive_count; ++t) {
        uint32_t i0 = scene.Indices()[triangle_mesh.first_index + t*3 + 0];
        uint32_t i1 = scene.Indices()[triangle_mesh.first_index + t*3 + 1];
        uint32_t i2 = scene.Indices()[triangle_mesh.first_index + t*3 + 2];

        const glm::vec3 &v0 = scene.Positions()[i0];
        const glm::vec3 &v1 = scene.Positions()[i1];
        const glm::vec3 &v2 = scene.Positions()[i2];

        m_primitives[t].centroid = (v0 + v1 + v2) * (1.0f/3.0f);

        m_primitives[t].aabb.Reset();
        m_primitives[t].aabb.AddPoint(v0);
        m_primitives[t].aabb.AddPoint(v1);
        m_primitives[t].aabb.AddPoint(v2);
    }

    m_nodes.emplace_back();
    BVHNode &root = m_nodes[0];
    root.first_primitive = 0;
    root.primitive_count = primitive_count;

    UpdateNodeBounds(0);
    Subdivide(0);
}

void BLAS::UpdateNodeBounds(uint32_t index) {
    BVHNode &node = m_nodes[index];

    node.bounds.Reset();

    if (node.IsLeaf()) {
        for (uint32_t i = 0; i < node.primitive_count; ++i)
            node.bounds.Union(m_primitives[m_tri_indices[node.FirstPrimitive() + i]].aabb);
    } else {
        node.bounds = m_nodes[node.LeftChild()].bounds;
        node.bounds.Union(m_nodes[node.RightChild()].bounds);
    }
}

void BLAS::Subdivide(uint32_t index) {
    BVHNode &node = m_nodes[index];
    if (node.primitive_count <= 2)
        return;

    float cost;
    auto [axis, split_pos] = FindSplitPlane(index, cost);
    if (m_split_strategy == SplitStrategy::SAH) {
        float parent_cost = node.primitive_count * node.bounds.SurfaceArea();
        if (cost >= parent_cost) return;
    }

    // partition primitives by the splitting plane
    uint32_t i = node.first_primitive;
    uint32_t j = i + node.primitive_count - 1;
    while (i <= j) {
        uint32_t prim = m_tri_indices[i];
        if (m_primitives[prim].centroid[axis] < split_pos) ++i;
        else std::swap(m_tri_indices[i], m_tri_indices[j--]);
    }
    uint32_t left_count = i - node.first_primitive;
    if (left_count == 0 || left_count == node.primitive_count)
        return;

    // if splitting downa  non-trivial position, create
    // each child node
    uint32_t left_index = uint32_t(m_nodes.size());
    m_nodes.emplace_back();
    m_nodes.emplace_back();
    uint32_t right_index = left_index + 1;

    BVHNode &left = m_nodes[left_index];
    left.first_primitive = node.first_primitive;
    left.primitive_count = left_count;
    UpdateNodeBounds(left_index);
    
    BVHNode &right = m_nodes[right_index];
    right.first_primitive = node.first_primitive + left_count;
    right.primitive_count = node.primitive_count - left_count;
    UpdateNodeBounds(right_index);

    node.LeftChild() = left_index;
    node.primitive_count = 0;

    Subdivide(left_index);
    Subdivide(right_index);
}

std::pair<uint32_t, float> BLAS::FindSplitPlane(uint32_t index, float &cost) {
    BVHNode &node = m_nodes[index];

    if (m_split_strategy == SplitStrategy::Midpoint) {
        glm::vec3 extent = node.bounds.max - node.bounds.min;

        uint32_t axis = 0;
        float split_pos;

        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent[axis]) axis = 2;

        split_pos = node.bounds.min[axis] + 0.5f * node.bounds.max[axis];
        return std::make_pair(axis, split_pos);
    }

    if (m_split_strategy == SplitStrategy::SAH) {
        constexpr int bin_count = 8;
        float best_cost = FLT_MAX;
        uint32_t best_axis;
        float best_pos;
        
        Math::AABB centroid_bounds;
        for (uint32_t i = 0; i < node.primitive_count; ++i) {
            uint32_t tri = m_tri_indices[node.first_primitive + i];
            centroid_bounds.AddPoint(m_primitives[tri].centroid);
        }

        if (centroid_bounds.IsEmpty()) {
            cost = FLT_MAX;
            return {};
        }

        glm::vec3 centroid_extent = centroid_bounds.Extent();
        glm::vec3 inv_bin_width = 1.0f / glm::max(centroid_extent, glm::vec3(1e-30f));
        
        for (uint32_t axis = 0; axis < 3; ++axis) {
            if (centroid_extent[axis] <= 0.0f)
                continue;

            std::array<Bin, bin_count> bins {};
            for (uint32_t i = 0; i < node.primitive_count; ++i) {
                uint32_t primitive = m_tri_indices[node.first_primitive + i];

                // which bin does this tri belong in?
                float scaled_offset = (m_primitives[primitive].centroid[axis] - centroid_bounds.min[axis]) * inv_bin_width[axis];
                int bin_index = glm::clamp(static_cast<int>(scaled_offset * bin_count), 0, bin_count - 1);
            
                bins[bin_index].prims++;
                bins[bin_index].bounds.Union(m_primitives[primitive].aabb);
            }

            // Compute areas and primitive counts on each side of
            // each candidate splitting plane
            std::array<float, bin_count-1> left_area, right_area {};
            std::array<uint32_t, bin_count-1> left_count, right_count {};

            Math::AABB left_box, right_box;
            uint32_t left_sum = 0, right_sum = 0;
            for (int i = 0; i < bin_count - 1; ++i) {
                left_sum += bins[i].prims;
                left_count[i] = left_sum;
                left_box.Union(bins[i].bounds);
                left_area[i] = left_box.SurfaceArea();

                const int right_bin = bin_count - 1- i;
                right_sum += bins[right_bin].prims;
                right_count[right_bin-1] = right_sum;
                right_box.Union(bins[right_bin].bounds);
                right_area[right_bin-1] = right_box.SurfaceArea();
            }

            for (int i = 0; i < bin_count - 1; ++i) {
                // surface-area heuristic cost function
                float cost = left_count[i] * left_area[i] + right_count[i] * right_area[i];
                if (cost < best_cost) {
                    best_cost = cost;
                    best_axis = axis;
                    best_pos = centroid_bounds.min[axis] + centroid_extent[axis] * (float(i+1)/bin_count);
                }
            }
        }
        cost = best_cost;
        return std::make_pair(best_axis, best_pos);
    }
}

Math::AABB BLAS::Bounds() {
    return m_nodes[0].bounds;
}

void TLAS::Build(const Scene &scene, const std::vector<GeometryInstance> &instances, SplitStrategy split_type) {
    m_dirty = true;
    m_blases.clear();
    m_blas_nodes.clear();
    m_tri_indices.clear();
    m_tlas_nodes.clear();
    m_device_blases.clear();

    // -- BLAS setup --------

    // grab all unique blases
    int max_mesh_id = 0;
    for (auto &inst : instances)
        max_mesh_id = std::max(max_mesh_id, inst.blas_index);

    m_blases.resize(max_mesh_id + 1);

    // build each BLAS and store world-space bounds
    std::vector<Math::AABB> instance_bounds(instances.size());
    for (size_t i = 0; i < instances.size(); ++i) {
        const auto &inst = instances[i];
        uint32_t blas_id = inst.blas_index;

        if (m_blases[blas_id].Nodes().empty())
            m_blases[blas_id].Build(scene, inst.triangle_mesh, split_type);

        Math::AABB obj_bounds = m_blases[blas_id].Bounds();
        Math::AABB world_bounds = obj_bounds.Transform(inst.transform);
        
        instance_bounds[i] = world_bounds;
    }

    // compute device blases
    m_device_blases.resize(m_blases.size());
    for (uint32_t id = 0; id < m_blases.size(); ++id) {
        const auto& blas = m_blases[id];
        if (blas.Nodes().empty())
            continue;

        uint32_t node_offset = static_cast<uint32_t>(m_blas_nodes.size());

        for (const BVHNode& src : blas.Nodes()) {
            BVHNode dst = src;
            if (!dst.IsLeaf())
                dst.first_primitive += node_offset;
            m_blas_nodes.push_back(dst);
        }

        m_tri_indices.insert(m_tri_indices.end(),
                            blas.TriIndices().begin(),
                            blas.TriIndices().end());

        m_device_blases[id] = {
            node_offset,
            static_cast<uint32_t>(blas.Nodes().size()),
            static_cast<uint32_t>(m_tri_indices.size() - blas.TriIndices().size()),
            static_cast<uint32_t>(blas.TriIndices().size())
        };
    }

    // -- TLAS building --------
    m_tlas_nodes.reserve(2 * instances.size());

    // create a TLAS leaf node for each instance
    std::vector<uint32_t> instance_indices(instances.size());
    int node_count = int(instance_indices.size());

    m_tlas_nodes.emplace_back();
    for (uint32_t i = 0; i < instance_indices.size(); ++i) {
        instance_indices[i] = m_tlas_nodes.size();
        m_tlas_nodes.emplace_back();
        m_tlas_nodes.back().bounds = instance_bounds[i];
        m_tlas_nodes.back().instance = i;
        m_tlas_nodes.back().left_right = 0;
    }

    // Finds the best instance
    auto find_best_match = [&](int instance) {
        float best_area = FLT_MAX;
        int best_j = -1;
        for (int j = 0; j < node_count; ++j) {
            if (j == instance) continue;
            uint32_t idx_i = instance_indices[instance];
            uint32_t idx_j = instance_indices[j];

            Math::AABB merged = m_tlas_nodes[idx_i].bounds;
            merged.Union(m_tlas_nodes[idx_j].bounds);
            float area = merged.SurfaceArea();
            if (area < best_area) {
                best_area = area;
                best_j = j;
            }
        }
        return best_j;
    };

    // builds additional TLAS nodes by merging them together
    int a = 0;
    int b = find_best_match(a);
    while (node_count > 1) {
        int c = find_best_match(b);
        if (a == c) {
            int node_index_a = instance_indices[a];
            int node_index_b = instance_indices[b];
            TLASNode &node_a = m_tlas_nodes[instance_indices[a]];
            TLASNode &node_b = m_tlas_nodes[instance_indices[b]];
            
            // construct new node
            m_tlas_nodes.emplace_back();
            TLASNode &new_node = m_tlas_nodes.back();
            new_node.left_right = node_index_a
                                | (node_index_b << 16);
            new_node.bounds = node_a.bounds;
            new_node.bounds.Union(node_b.bounds);

            instance_indices[a] = m_tlas_nodes.size() - 1;
            instance_indices[b] = instance_indices[node_count - 1];
            --node_count;

            b = find_best_match(a);
        } else {
            a = b;
            b = c;
        }
    }
    m_tlas_nodes[0] = m_tlas_nodes[instance_indices[a]];
}

void TLAS::UpdateDevice(DeviceScene &scene) {
    if (!m_dirty) return;

    if (scene.tlas == nullptr)
        CUDA_CHECK(cudaMalloc((void**) &scene.tlas, sizeof(DeviceTLAS)));

    DeviceTLAS tlas {};
    tlas.tlas_node_count = CopyToDevice(tlas.tlas_nodes, m_tlas_nodes);
    tlas.blas_node_count = CopyToDevice(tlas.blas_nodes, m_blas_nodes);
    tlas.tri_index_count = CopyToDevice(tlas.tri_indices, m_tri_indices);

    tlas.blases = nullptr;
    tlas.tlas_node_count = uint32_t(m_tlas_nodes.size());
    tlas.blas_count = CopyToDevice(tlas.blases, m_device_blases);

    CUDA_CHECK(cudaMemcpy(scene.tlas, &tlas, sizeof(DeviceTLAS), cudaMemcpyHostToDevice));

    m_dirty = false;
}

void TLAS::FreeDevice(DeviceScene &scene) {
    if (scene.tlas == nullptr) return;
    DeviceTLAS tlas;
    CUDA_CHECK(cudaMemcpy(&tlas, scene.tlas, sizeof(DeviceTLAS), cudaMemcpyDeviceToHost));

    cudaFree(tlas.tlas_nodes);
    cudaFree(tlas.blas_nodes);
    cudaFree(tlas.tri_indices);
    cudaFree(tlas.blases);
    cudaFree(scene.tlas);
    scene.tlas = nullptr;
}
