#include "Intersections.h"
#include "scene.h"
#include "utils.h"

__host__ __device__ float BoxIntersectionTest(
    const Geometry &box,
    const Ray &r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
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
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = glm::vec3(box.transform * glm::vec4(q(tmin), 1.0f));
        normal = glm::normalize(glm::vec3(box.inv_transpose * glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__
float SphereIntersectionTest(
    const Geometry& sphere,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    constexpr float radius = 0.5f;
    constexpr float radius2 = radius * radius;
    constexpr float kEpsilon = 1e-4f;

    glm::vec3 ro = glm::vec3(sphere.inv_transform * glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(glm::vec3(
        sphere.inv_transform * glm::vec4(r.direction, 0.0f)));

    float b = 2.0f * glm::dot(ro, rd);
    float c = glm::dot(ro, ro) - radius2;
    float disc = b * b - 4.0f * c;
    if (disc < 0.0f) {
        return -1.0f;
    }

    float sqrtDisc = sqrtf(disc);
    float t0 = (-b - sqrtDisc) * 0.5f;
    float t1 = (-b + sqrtDisc) * 0.5f;

    outside = (glm::dot(ro, ro) > radius2);
    float tObj = outside ? t0 : t1;
    if (tObj < kEpsilon) {
        return -1.0f;
    }

    glm::vec3 pObj = ro + rd * tObj;

    intersectionPoint = glm::vec3(
        sphere.transform * glm::vec4(pObj, 1.0f));

    normal = glm::normalize(glm::vec3(
        sphere.inv_transpose * glm::vec4(pObj, 0.0f)));

    return glm::dot(intersectionPoint - r.origin, r.direction);
}
