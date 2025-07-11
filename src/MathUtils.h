#pragma once

#define c_PI        3.14159265358979323846f
#define c_INV_PI    0.31830988618379067154f
#define c_INV_SQRT3 0.57735026918962576451f
#define c_SQRT3     1.73205080756887729353f 

#include <glm/glm.hpp>
#include <thrust/random.h>
#include <cuda_runtime.h>

namespace Math {

    /**
     * @brief Samples point in glm::vec<N>
     */
    template<glm::length_t N, glm::qualifier Q = glm::defaultp>
    __host__ __device__ glm::vec<N, float, Q> UniformSample(thrust::default_random_engine &rng) {
        thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
        glm::vec<N, float, Q> vec;
        for(glm::length_t i = 0; i < N; ++i)
            vec[i] = dist(rng);
        return vec;
    }

    __host__ __device__ inline float Sample1D(thrust::default_random_engine &rng) {
        thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(rng);
    }
    __host__ __device__ inline glm::vec2 Sample2D(thrust::default_random_engine &rng) { return UniformSample<2>(rng); }
    __host__ __device__ inline glm::vec3 Sample3D(thrust::default_random_engine &rng) { return UniformSample<3>(rng); }
    __host__ __device__ inline glm::vec4 Sample4D(thrust::default_random_engine &rng) { return UniformSample<4>(rng); }

    /**
     * @brief Assume `theta` in [0, \pi] and `phi` in [0, 2\pi]
     */
    __host__ __device__ glm::vec3 FromSphericalCoords(float theta, float phi);

    /**
     * @brief Samples hemisphere about `normal` assuming `uniform` are
     * independent random uniform variables on [0, 1].
     */
    __host__ __device__ glm::vec3 SampleUniformHemisphere(const glm::vec3 &normal, thrust::default_random_engine &rng);

    /**
     * @brief Same as above, but the PDF is cosine-weighted.
     */
    __host__ __device__ glm::vec3 SampleCosineHemisphere(const glm::vec3 &normal, thrust::default_random_engine &rng);

    __host__ __device__ glm::vec3 SampleGGX(
        const glm::vec3 &normal,
        float roughness,
        thrust::default_random_engine &rng
    );

    /**
     * @brief Computes
     * 
     */
    __host__ __device__ glm::vec3 Reflect(const glm::vec3 &vec, const glm::vec3 &normal);

    __host__ __device__ bool Refract(const glm::vec3 &vec, const glm::vec3 &normal, float eta, glm::vec3 &result);

    __host__ __device__ inline unsigned int JenkinsHash(unsigned int a) {
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a << 5);
        a = (a + 0xd3a2646c) ^ (a << 9);
        a = (a + 0xfd7046c5) + (a << 3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return a;
    }

    __host__ __device__ inline float Pow5(float base) {
        float base2 = base * base;
        return base2 * base2 * base;
    }
}
