#include "Intersections.h"
#include "utils.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>

__host__ __device__ float BoxIntersectionTest(
    const Geometry &box,
    const Ray &r,
    Intersection &intersection)
{
    Ray q;
    q.origin = glm::vec3(box.inv_transform * glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(glm::vec3(box.inv_transform * glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
        }

        intersection.pos = glm::vec3(box.transform * glm::vec4(q(tmin), 1.0f));
        intersection.normal = glm::normalize(glm::vec3(box.inv_transpose * glm::vec4(tmin_n, 0.0f)));
        intersection.t = glm::length(r.origin - intersection.pos);

        glm::vec3 localP = q.origin + q.direction * tmin;

        glm::vec2 uv;
        if (fabs(tmin_n.x) > 0.0f) {
            uv = glm::vec2(localP.z + 0.5f, localP.y + 0.5f);
        }
        else if (fabs(tmin_n.y) > 0.0f) {
            uv = glm::vec2(localP.x + 0.5f, localP.z + 0.5f);
        }
        else {
            uv = glm::vec2(localP.x + 0.5f, localP.y + 0.5f);
        }
        intersection.uv = uv;

        return intersection.t;
    }

    intersection.t = -1;
    return -1;
}

__host__ __device__
float SphereIntersectionTest(
    const Geometry& sphere,
    const Ray& r,
    Intersection &intersection)
{
    constexpr float radius = 0.5f;
    constexpr float radius2 = radius * radius;
    constexpr float kEpsilon = 1e-6f;

    glm::vec3 ro = glm::vec3(sphere.inv_transform * glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(glm::vec3(
        sphere.inv_transform * glm::vec4(r.direction, 0.0f)));

    float b = 2.0f * glm::dot(ro, rd);
    float c = glm::dot(ro, ro) - radius2;
    float disc = b * b - 4.0f * c;
    if (disc < 0.0f) {
        intersection.t = -1;
        return -1.0f;
    }

    float sqrtDisc = sqrtf(disc);
    float t0 = (-b - sqrtDisc) * 0.5f;
    float t1 = (-b + sqrtDisc) * 0.5f;

    float outside = (glm::dot(ro, ro) > radius2);
    float tObj = outside ? t0 : t1;
    if (tObj < kEpsilon) {
        intersection.t = -1;
        return -1.0f;
    }

    glm::vec3 pObj = ro + rd * tObj;

    intersection.pos = glm::vec3(
        sphere.transform * glm::vec4(pObj, 1.0f));

    intersection.normal = glm::normalize(glm::vec3(
        sphere.inv_transpose * glm::vec4(pObj, 0.0f)));
    intersection.t = glm::dot(intersection.pos - r.origin, r.direction);

    float u = 0.5f + atan2f(pObj.z, pObj.x) * c_INV_PI / 2.0f;
    float v = 0.5f - asinf(pObj.y / 0.5f) * c_INV_PI;
    intersection.uv = glm::vec2(u, v);

    return intersection.t;
}

__host__ __device__
float TriangleIntersectionTest(
    const Ray &ray,
    const glm::vec3 &v0,
    const glm::vec3 &v1,
    const glm::vec3 &v2,
    glm::vec2 &uv)
{
    const float EPSILON = 1e-8f;
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 p = glm::cross(ray.direction, edge2);
    float determinant = glm::dot(edge1, p);
    if (fabs(determinant) < EPSILON)
        return -1.0f;
    float inv_det = 1.0f / determinant;

    glm::vec3 tvec = ray.origin - v0;
    float u = glm::dot(tvec, p) * inv_det;
    if (u < 0.0f || u > 1.0f)
        return -1.0f;

    glm::vec3 q = glm::cross(tvec, edge1);
    float v = glm::dot(ray.direction, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f)
        return -1.0f;

    float t = glm::dot(edge2, q) * inv_det;
    if (t <= EPSILON)
        return -1.0f;

    uv = glm::vec2(u, v);
    return t;
}

__host__ __device__
float NaivePrimitiveIntersection(
    const Geometry & geom,
    const MeshVertex *vertices,
    const uint32_t *indices,
    const Ray &r,
    Intersection &intersection)
{
    Ray ray;
    ray.origin = glm::vec3(geom.inv_transform * glm::vec4(r.origin, 1.0f));
    ray.direction = glm::normalize(glm::vec3(geom.inv_transform * glm::vec4(r.direction, 0.0f)));

    float best_t = FLT_MAX;
    glm::vec3 best_normal(0.0f);
    glm::vec2 best_uv(0.0f);

    uint32_t start = geom.mesh.first_index;
    uint32_t end = start + geom.mesh.index_count;
    for (uint32_t i = start; i < end; i += 3) {
        uint32_t vi0 = indices[i + 0];
        uint32_t vi1 = indices[i + 1];
        uint32_t vi2 = indices[i + 2];

        const MeshVertex &v0 = vertices[vi0];
        const MeshVertex &v1 = vertices[vi1];
        const MeshVertex &v2 = vertices[vi2];

        glm::vec2 bary;
        float t_obj = TriangleIntersectionTest(ray, v0.position, v1.position, v2.position, bary);
        if (t_obj < 0.0f)
            continue;

        glm::vec3 p_obj = ray(t_obj);
        glm::vec3 p_world = glm::vec3(geom.transform * glm::vec4(p_obj, 1.f));
        float t_world = glm::dot(p_world - r.origin, r.direction);

        if (t_world < best_t) {
            best_t = t_world;

            glm::vec3 n0 = v0.normal, n1 = v1.normal, n2 = v2.normal;
            bool has_normals = (glm::length2(n0) > 1e-6f) && (glm::length2(n1) > 1e-6f) && (glm::length2(n2) > 1e-6f);

            if (has_normals) {
                float w = 1.0f - bary.x - bary.y;
                best_normal = glm::normalize(w * n0 + bary.x * n1 + bary.y * n2);
            } else {
                best_normal = glm::normalize(glm::cross(
                    v1.position - v0.position,
                    v2.position - v0.position));
            }

            glm::vec2 uv0 = v0.uv0, uv1 = v1.uv0, uv2 = v2.uv0;
            bool has_uvs = (glm::length2(uv0) > 1e-6f) && (glm::length2(uv1) > 1e-6f) && (glm::length2(uv2) > 1e-6f);

            if (has_uvs) {
                float w = 1.0f - bary.x - bary.y;
                best_uv = w*uv0 + bary.x*uv1 + bary.y*uv2;
            } else {
                best_uv = bary;
            }
        }
    }

    // no hit
    if (fabsf(best_t) < 1e-5 || best_t == FLT_MAX) {
        intersection.t = -1;
        return -1.0f;
    }

    // transform to world space
    intersection.pos = r(best_t);
    glm::vec3 normal_world = glm::normalize(glm::vec3(geom.inv_transpose * glm::vec4(best_normal, 0.0f)));
    intersection.normal = normal_world;
    intersection.uv = best_uv;
    intersection.t = best_t;

    return best_t;
}

#if USE_BVH

__host__ __device__
inline bool IntersectAABB(
    const Ray& r,
    const AABB &aabb,
    float t_max,
    float &t_out)
{
    const glm::vec3 inv_d = 1.0f / r.direction;

    float t0 = (aabb.min.x - r.origin.x) * inv_d.x;
    float t1 = (aabb.max.x - r.origin.x) * inv_d.x;
    float tnear = fminf(t0, t1);
    float tfar = fmaxf(t0, t1);

    t0 = (aabb.min.y - r.origin.y) * inv_d.y;
    t1 = (aabb.max.y - r.origin.y) * inv_d.y;
    tnear = fmaxf(tnear, fminf(t0, t1));
    tfar = fminf(tfar, fmaxf(t0, t1));

    t0 = (aabb.min.z - r.origin.z) * inv_d.z;
    t1 = (aabb.max.z - r.origin.z) * inv_d.z;
    tnear = fmaxf(tnear, fminf(t0, t1));
    tfar = fminf(tfar, fmaxf(t0, t1));

    const bool hit = (tfar >= tnear) && (tnear < t_max) && (tfar > 0.f);
    if (hit) t_out = tnear;
    return hit;
}

__device__ inline void FetchTri(const Geometry& g,
    uint32_t tri_id,
    const uint32_t *indices,
    const MeshVertex *vertices,
    glm::vec3 &p0,
    glm::vec3 &p1,
    glm::vec3 &p2)
{
    const uint32_t base = g.mesh.first_index + tri_id * 3u;
    const uint32_t i0 = indices[base + 0];
    const uint32_t i1 = indices[base + 1];
    const uint32_t i2 = indices[base + 2];
    p0 = vertices[i0].position;
    p1 = vertices[i1].position;
    p2 = vertices[i2].position;
}

struct StackEntry
{
    uint32_t node;
    float t_near;
};

__device__ inline void Push(StackEntry *s, uint32_t &top, uint32_t n, float t) {
    s[top].node = n;
    s[top].t_near = t;
    ++top;
}

__device__ float IntersectBVH(
    const Geometry &geom,
    const BVH::BVHNode *nodes,
    const uint32_t *tri_indices,
    const MeshVertex *vertices,
    const uint32_t *indices,
    const Ray &r,
    Intersection &intersection)
{
    Ray ray;
    ray.origin = glm::vec3(geom.inv_transform * glm::vec4(r.origin, 1.0f));
    ray.direction = glm::normalize(glm::vec3(geom.inv_transform * glm::vec4(r.direction, 0.0f)));

    StackEntry stack[64];
    uint32_t top = 0u;
    Push(stack, top, geom.first_bvh_node, 0.0f);

    float best_t_obj = FLT_MAX;
    float best_t_world = FLT_MAX;
    glm::vec3 best_normal(0.0f);
    glm::vec2 best_uv(0.0f);

    while (top > 0u) {
        StackEntry current_entry = stack[--top];
        const BVH::BVHNode &node_data = nodes[current_entry.node];

        if (current_entry.t_near >= best_t_obj) {
            continue;
        }

        if (node_data.IsLeaf()) {
            for (uint32_t tri_iter = 0u; tri_iter < node_data.tri_count; ++tri_iter) {
                uint32_t local_tri_idx = tri_indices[geom.first_tri_index
                                                          + node_data.first_tri
                                                          + tri_iter];
                uint32_t base_index = geom.mesh.first_index + local_tri_idx * 3u;

                uint32_t vi0 = indices[base_index + 0];
                uint32_t vi1 = indices[base_index + 1];
                uint32_t vi2 = indices[base_index + 2];

                const MeshVertex &v0 = vertices[vi0];
                const MeshVertex &v1 = vertices[vi1];
                const MeshVertex &v2 = vertices[vi2];

                glm::vec2 bary;
                float t_obj = TriangleIntersectionTest(ray, v0.position, v1.position, v2.position, bary);
                if (t_obj < 0.0f) {
                    continue;
                }

                glm::vec3 p_obj = ray(t_obj);
                glm::vec3 p_world = glm::vec3(geom.transform * glm::vec4(p_obj, 1.f));
                float t_world = glm::dot(p_world - r.origin, r.direction);

                if (t_world < best_t_world || best_t_world < 0.0f) {
                    best_t_world = t_world;
                    best_t_obj = t_obj;
                    
                    float w = 1.0f - bary.x - bary.y;
                    glm::vec3 n0 = v0.normal, n1 = v1.normal, n2 = v2.normal;
                    bool has_normals = (glm::length2(n0) > 1e-6f) && (glm::length2(n1) > 1e-6f) && (glm::length2(n2) > 1e-6f);

                    if (has_normals) {
                        best_normal = glm::normalize(w * n0 + bary.x * n1 + bary.y * n2);
                    } else {
                        best_normal = glm::normalize(glm::cross(
                            v1.position - v0.position,
                            v2.position - v0.position));
                    }

                    glm::vec2 uv0 = v0.uv0, uv1 = v1.uv0, uv2 = v2.uv0;
                    bool has_uvs = (glm::length2(uv0) > 1e-6f) && (glm::length2(uv1) > 1e-6f) && (glm::length2(uv2) > 1e-6f);

                    if (has_uvs) {
                        float w = 1.0f - bary.x - bary.y;
                        best_uv = w*uv0 + bary.x*uv1 + bary.y*uv2;
                    } else {
                        best_uv = bary;
                    }
                }
            }
        } else {
            uint32_t left_index  = node_data.left_child;
            uint32_t right_index = node_data.RightChild();

            float t_near_left  = 0.0f;
            float t_near_right = 0.0f;
            float t_limit = best_t_obj;

            bool hit_left  = IntersectAABB(ray, nodes[left_index].bounds,  t_limit, t_near_left);
            bool hit_right = IntersectAABB(ray, nodes[right_index].bounds, t_limit, t_near_right);

            if (hit_left && hit_right) {
                if (t_near_left < t_near_right) {
                    Push(stack, top, right_index, t_near_right);
                    Push(stack, top, left_index,  t_near_left);
                } else {
                    Push(stack, top, left_index,  t_near_left);
                    Push(stack, top, right_index, t_near_right);
                }
            } else if (hit_left) {
                Push(stack, top, left_index, t_near_left);
            } else if (hit_right) {
                Push(stack, top, right_index, t_near_right);
            }
        }
    }

    if (best_t_world < 1e-5 || best_t_world == FLT_MAX) {
        intersection.t = -1;
        return -1.0f;
    }

    // transform to world space
    intersection.pos = r(best_t_world);
    
    glm::vec3 normal_world = glm::normalize(glm::vec3(geom.inv_transpose * glm::vec4(best_normal, 0.0f)));
    intersection.normal = normal_world;
    intersection.uv = best_uv;
    intersection.t = best_t_world;

    return best_t_world;
}

#endif
