#pragma once

#include "device_scene.h"
#include <glm/glm.hpp>
#include "../utils/stack.h"
#include "../math/ray.h"
#include "../math/intersection.h"
#include "../math/constants.h"

struct Intersection {
    float time;
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
    int material_id = -1;
    int geometry_id = -1;
};

struct StackItem {
    uint32_t node_index;
    float aabb_time;
};

struct NoOp {
  __device__ void operator()(int, bool) const {}
};

enum class IntersectQueryMode { CLOSEST, ANY };

template<IntersectQueryMode Mode, typename Callback = NoOp>
__device__ bool IntersectBLAS(const DeviceScene &scene, const GeometryInstance &geometry, const Math::Ray &ray, Intersection &intersection, float t_max = FLT_MAX, Callback callback = {}) {
    if (scene.tlas->blas_node_count <= 0) {
        intersection.time = -1.0f;
        return false;
    }
    
    DeviceBLAS &blas = scene.tlas->blases[geometry.blas_index];

    Stack<StackItem> stack;
    stack.Push({ blas.first_node, 0.0f });

    BVHNode *nodes = scene.tlas->blas_nodes;

    const Math::Ray local_ray = geometry.inv_transform * ray;

    // Best intersection results in object space
    float best_time = FLT_MAX;
    glm::vec2 best_uv;
    glm::vec3 best_pos;
    glm::vec3 best_normal;

    while (!stack.Empty()) {
        StackItem current_entry = stack.Pop();
        const BVHNode &node = nodes[current_entry.node_index];

        if constexpr (Mode == IntersectQueryMode::CLOSEST) {
            if (current_entry.aabb_time >= best_time) {
                continue;
            }
        }
        callback(node.primitive_count, true);

        if (node.IsLeaf()) {
            // For now, all geometries are triangle meshes. We may easily extend this
            // by adding other geometries here
            for (uint32_t i = 0; i < node.primitive_count; ++i) {
                uint32_t tri_index = scene.tlas->tri_indices[blas.first_index + node.first_primitive + i];
                uint32_t base_index = geometry.triangle_mesh.first_index + tri_index * 3u;

                uint32_t i0 = scene.indices[base_index + 0];
                uint32_t i1 = scene.indices[base_index + 1];
                uint32_t i2 = scene.indices[base_index + 2];

                glm::vec3 p0 = scene.positions[i0];
                glm::vec3 p1 = scene.positions[i1];
                glm::vec3 p2 = scene.positions[i2];

                glm::vec3 n0 = scene.normals[i0];
                glm::vec3 n1 = scene.normals[i1];
                glm::vec3 n2 = scene.normals[i2];

                glm::vec2 uv0 = scene.uvs[i0];
                glm::vec2 uv1 = scene.uvs[i1];
                glm::vec2 uv2 = scene.uvs[i2];
                
                glm::vec2 bary;
                float time;
                if (!Math::IntersectTriangle(local_ray, p0, p1, p2, bary, time, t_max)) {
                    continue;
                }

                if constexpr (Mode == IntersectQueryMode::CLOSEST) {
                    if (time < best_time) {
                        best_time = time;
                        
                        float w = 1.0f - bary.x - bary.y;
                        best_normal = glm::normalize(w*n0 + bary.x*n1 + bary.y*n2);
                        best_uv = w*uv0 + bary.x*uv1 + bary.y*uv2;
                        best_pos = w*p0 + bary.x*p1 + bary.y*p2;
                    }
                } else {
                    // Early-out if we managed to find ANY intersection
                    best_time = time;
                    
                    float w = 1.0f - bary.x - bary.y;
                    best_normal = glm::normalize(w*n0 + bary.x*n1 + bary.y*n2);
                    best_uv = w*uv0 + bary.x*uv1 + bary.y*uv2;
                    best_pos = w*p0 + bary.x*p1 + bary.y*p2;
                    break;
                }

            }
        } else {
            uint32_t left = nodes[current_entry.node_index].LeftChild();
            uint32_t right = nodes[current_entry.node_index].RightChild();

            float left_time;
            float right_time;

            bool hit_left = Math::IntersectAABB(local_ray, scene.tlas->blas_nodes[left].bounds, left_time, best_time);
            bool hit_right = Math::IntersectAABB(local_ray, scene.tlas->blas_nodes[right].bounds, right_time, best_time);
            
            if (hit_left && hit_right) {
                if (left_time < right_time) {
                    stack.Push({ right, right_time });
                    stack.Push({ left, left_time });
                } else {
                    stack.Push({ left, left_time });
                    stack.Push({ right, right_time });
                }
            } else if (hit_left) {
                stack.Push({ left, left_time });
            } else if (hit_right) {
                stack.Push({ right, right_time });
            }
        }
    }

    if (best_time < 0.0f || best_time == FLT_MAX) {
        intersection.time = -1.0;
        return false;
    }

    // Transform everything back to world space
    intersection.pos = glm::vec3(geometry.transform * glm::vec4(best_pos, 1.0f));
    intersection.uv = best_uv;
    intersection.time = glm::dot(intersection.pos - ray.origin, ray.direction);
    intersection.normal = glm::normalize(glm::vec3(geometry.inv_transpose * glm::vec4(best_normal, 0.0f)));
    intersection.material_id = geometry.material_id;

    return true;
}

template<IntersectQueryMode Mode, typename Callback = NoOp>
__device__ bool IntersectScene(const DeviceScene &scene, const Math::Ray &ray, Intersection &intersection, float t_max = FLT_MAX, Callback callback = {}) {
    if (scene.tlas->tlas_node_count <= 0) {
        intersection.time = -1.0f;
        return false;
    }

    const TLASNode *nodes = scene.tlas->tlas_nodes;
    if (float t = 0; !Math::IntersectAABB(ray, nodes[0].bounds, t, t_max)) {
        intersection.time = -1.0f;
        return false;
    }
    
    Stack<StackItem> stack;
    stack.Push({ 0 });

    float best_time = FLT_MAX;

    while (!stack.Empty()) {
        StackItem current_entry = stack.Pop();
        const TLASNode &node = nodes[current_entry.node_index];
        if constexpr (Mode == IntersectQueryMode::CLOSEST) {
            // `t` stores the AABB intersection, so we may early out
            if (current_entry.aabb_time >= best_time) {
                continue;
            }
        }
        callback(1, false);

        if (node.IsLeaf()) {
            GeometryInstance &geometry = scene.geometries[node.instance];

            Intersection temp_intersect;
            bool hit = IntersectBLAS<Mode>(scene, geometry, ray, temp_intersect, t_max, callback);
            temp_intersect.geometry_id = node.instance;

            if constexpr (Mode == IntersectQueryMode::CLOSEST) {
                if (temp_intersect.time < best_time && hit) {
                    best_time = temp_intersect.time;
                    intersection = temp_intersect;
                }
            } else {
                if (hit) {
                    best_time = temp_intersect.time;
                    intersection = temp_intersect;
                    break;
                }
            }
        } else {
            uint32_t left = nodes[current_entry.node_index].LeftChild();
            uint32_t right = nodes[current_entry.node_index].RightChild();

            float left_time;
            float right_time;

            bool hit_left = Math::IntersectAABB(ray, nodes[left].bounds, left_time, best_time);
            bool hit_right = Math::IntersectAABB(ray, nodes[right].bounds, right_time, best_time);
            
            if (hit_left && hit_right) {
                // Push on stack in order of who is closer, so we may early-out even earlier
                if (left_time < right_time) {
                    stack.Push({ right, right_time });
                    stack.Push({ left, left_time });
                } else {
                    stack.Push({ left, left_time });
                    stack.Push({ right, right_time });
                }
            } else if (hit_left) {
                stack.Push({ left, left_time });
            } else if (hit_right) {
                stack.Push({ right, right_time });
            }
        }
    }

    if (best_time < 0.0f || best_time == FLT_MAX) {
        intersection.time = -1.0f;
        return false;
    }

    return true;
}

template<IntersectQueryMode Mode>
__device__ bool IntersectSceneNaive(
    const DeviceScene &scene,
    const Math::Ray &ray,
    Intersection &intersection,
    float t_max = FLT_MAX)
{
    float best_time = FLT_MAX;
    glm::vec3 best_pos{0.0f};
    glm::vec3 best_normal{0.0f};
    glm::vec2 best_uv{0.0f};
    const GeometryInstance *best_geom = nullptr;

    for (uint32_t gi = 0; gi < scene.geometry_count; ++gi) {
        const GeometryInstance &geom = scene.geometries[gi];
        Math::Ray local_ray = geom.inv_transform * ray;

        uint32_t start = geom.triangle_mesh.first_index;
        uint32_t end   = start + geom.triangle_mesh.index_count;
        for (uint32_t idx = start; idx < end; idx += 3) {
            uint32_t i0 = scene.indices[idx + 0];
            uint32_t i1 = scene.indices[idx + 1];
            uint32_t i2 = scene.indices[idx + 2];

            glm::vec3 p0 = scene.positions[i0];
            glm::vec3 p1 = scene.positions[i1];
            glm::vec3 p2 = scene.positions[i2];

            glm::vec3 n0 = scene.normals[i0];
            glm::vec3 n1 = scene.normals[i1];
            glm::vec3 n2 = scene.normals[i2];

            glm::vec2 uv0 = scene.uvs[i0];
            glm::vec2 uv1 = scene.uvs[i1];
            glm::vec2 uv2 = scene.uvs[i2];

            glm::vec2 bary;
            float t;
            if (!Math::IntersectTriangle(local_ray, p0, p1, p2, bary, t, t_max))
                continue;

            if constexpr (Mode == IntersectQueryMode::CLOSEST) {
                if (t < best_time) {
                    best_time = t;
                    float w = 1.0f - bary.x - bary.y;
                    best_normal = glm::normalize(w*n0 + bary.x*n1 + bary.y*n2);
                    best_uv = w*uv0 + bary.x*uv1 + bary.y*uv2;
                    best_pos = w*p0 + bary.x*p1 + bary.y*p2;
                    best_geom = &geom;
                }
            } else {
                best_time = t;
                float w = 1.0f - bary.x - bary.y;
                best_normal = glm::normalize(w*n0 + bary.x*n1 + bary.y*n2);
                best_uv = w*uv0 + bary.x*uv1 + bary.y*uv2;
                best_pos = w*p0 + bary.x*p1 + bary.y*p2;
                best_geom = &geom;
                break;
            }
        }

        if constexpr (Mode == IntersectQueryMode::ANY) {
            if (best_geom) break;
        }
    }

    if (best_time == FLT_MAX || !best_geom) {
        intersection.time = -1.0f;
        return false;
    }

    intersection.pos = glm::vec3(best_geom->transform * glm::vec4(best_pos, 1.0f));
    intersection.uv = best_uv;
    intersection.time = glm::distance(intersection.pos, ray.origin);
    intersection.normal = glm::normalize(glm::vec3(best_geom->inv_transpose * glm::vec4(best_normal, 0.0f)));

    return true;
}
