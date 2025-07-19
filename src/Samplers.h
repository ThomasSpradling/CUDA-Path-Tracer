#pragma once

#include "MathUtils.h"
#include "exception.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include <stack>
#include <thrust/random.h>
#include <vector>
#include "CudaUtils.h"

// Discrete sampler designed for O(n log n) initialization on the CPU
// and O(1) sampling on the GPU
// Reference: https://en.wikipedia.org/wiki/Alias_method

// It works by evening out the discrete distribution by "pouring" higher probabilities
// into lower ones, effectively mapping the problem into N biased bernoulli tests.

struct DiscreteSampler1DView {
    uint32_t *alias_table = nullptr;
    float *probability_table = nullptr;

    uint32_t size = 0;

    __device__ uint32_t Sample(thrust::default_random_engine &rng) {
        float x = Math::Sample1D(rng);
        float scaled = x * float(size);

        uint32_t i = static_cast<uint32_t>(scaled);
        if (scaled >= size) {
            i = size - 1;
            scaled = float(i);
        }

        float y = scaled - static_cast<float>(i);

        return (y < probability_table[i]) ? i : alias_table[i];
    }
};

class DiscreteSampler1D {
public:
    DiscreteSampler1D(const std::vector<float> &weights) {
        constexpr float epsilon = 1e-6;
        const uint32_t n = weights.size();

        float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
        if (sum == 0.0f) {
            PT_ERROR("Cannot use discrete sampler if its CDF is 0!");
        }

        m_view.size = n;

        std::vector<float> probs(n);
        std::vector<uint32_t> alias(n, UINT_MAX);

        for (uint32_t i = 0; i < n; ++i) {
            probs[i] = n * weights[i] / sum;
        }

        auto compare_greater = [&](uint32_t a, uint32_t b) { return probs[a] < probs[b]; };
        auto compare_lesser = [&](uint32_t a, uint32_t b) { return probs[a] > probs[b]; };

        std::priority_queue<uint32_t, std::vector<uint32_t>, decltype(compare_greater)> overfull(compare_greater);
        std::priority_queue<uint32_t, std::vector<uint32_t>, decltype(compare_lesser)> underfull(compare_lesser);

        for (uint32_t i = 0; i < n; ++i) {
            if (probs[i] > 1.0f + epsilon) {
                overfull.push(i);
            } else if (probs[i] < 1.0f - epsilon) {
                underfull.push(i);
            } else {
                probs[i] = 1.0f;
            }
        }

        while (!overfull.empty() && !underfull.empty()) {
            // Grab largest overflower and smallest underflower
            uint32_t i = overfull.top(); overfull.pop();
            uint32_t j = underfull.top(); underfull.pop();

            // Fill j using excess probability from i
            alias[j] = i;
            probs[i] -= 1.0f - probs[j];

            if (probs[i] > 1.0f + epsilon) {
                overfull.push(i);
            } else if (probs[i] < 1.0f - epsilon) {
                underfull.push(i);
            } else {
                probs[i] = 1.0f;
            }
        }

        // Clamp any extras
        while (!overfull.empty()) {
            probs[overfull.top()] = 1.0f;
            overfull.pop();
        }

        while (!underfull.empty()) {
            probs[underfull.top()] = 1.0f;
            underfull.pop();
        }

        // Upload to GPU
        CopyToDevice(m_view.alias_table, alias);
        CopyToDevice(m_view.probability_table, probs);
    }

    const DiscreteSampler1DView &View() const { return m_view; }
private:
    DiscreteSampler1DView m_view;
};
