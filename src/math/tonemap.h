#include "math.h"

namespace Math {

    template<typename T>
    __device__ __forceinline__ T ReinhardToneMap(T x) {
        return x / (1.0f + x);
    }

    template<typename T>
    __device__ __forceinline__ T ACESFilmicToneMap(T x) {
        const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
        return Math::Saturate((x*(a*x + b)) / (x*(c*x + d) + e));
    }

}
