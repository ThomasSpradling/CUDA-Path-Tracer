#include "discrete_sampler.h"
#include "math/constants.h"
#include "utils/cuda_utils.h"
#include <numeric>
#include <queue>

// Discrete sampler designed for O(n log n) initialization on the CPU
// and O(1) sampling
// Reference: https://en.wikipedia.org/wiki/Alias_method

// It works by evening out the discrete distribution by "pouring" higher probabilities
// into lower ones, effectively mapping the problem into N biased bernoulli tests.

void DiscreteSampler1D::UpdateWeights(const std::vector<float> &weights) {
    const uint32_t n = weights.size();

    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    if (sum == 0.0f) {
        PT_ERROR("Cannot use discrete sampler if its CDF is 0!");
    }

    m_probability_table.resize(n);
    std::fill(m_probability_table.begin(), m_probability_table.end(), 0.0f);

    m_alias_table.resize(n);
    std::fill(m_alias_table.begin(), m_alias_table.end(), UINT_MAX);

    for (uint32_t i = 0; i < n; ++i) {
        m_probability_table[i] = n * weights[i] / sum;
    }

    auto compare_greater = [&](uint32_t a, uint32_t b) { return m_probability_table[a] < m_probability_table[b]; };
    auto compare_lesser = [&](uint32_t a, uint32_t b) { return m_probability_table[a] > m_probability_table[b]; };

    std::priority_queue<uint32_t, std::vector<uint32_t>, decltype(compare_greater)> overfull(compare_greater);
    std::priority_queue<uint32_t, std::vector<uint32_t>, decltype(compare_lesser)> underfull(compare_lesser);

    for (uint32_t i = 0; i < n; ++i) {
        if (m_probability_table[i] > 1.0f + Math::EPSILON) {
            overfull.push(i);
        } else if (m_probability_table[i] < 1.0f - Math::EPSILON) {
            underfull.push(i);
        } else {
            m_probability_table[i] = 1.0f;
        }
    }

    while (!overfull.empty() && !underfull.empty()) {
        // Grab largest overflower and smallest underflower
        uint32_t i = overfull.top(); overfull.pop();
        uint32_t j = underfull.top(); underfull.pop();

        // Fill j using excess probability from i
        m_alias_table[j] = i;
        m_probability_table[i] -= 1.0f - m_probability_table[j];

        if (m_probability_table[i] > 1.0f + Math::EPSILON) {
            overfull.push(i);
        } else if (m_probability_table[i] < 1.0f - Math::EPSILON) {
            underfull.push(i);
        } else {
            m_probability_table[i] = 1.0f;
        }
    }

    // Clamp any extras
    while (!overfull.empty()) {
        m_probability_table[overfull.top()] = 1.0f;
        overfull.pop();
    }

    while (!underfull.empty()) {
        m_probability_table[underfull.top()] = 1.0f;
        underfull.pop();
    }
}

void DiscreteSampler1D::UpdateDevice(DeviceDiscreteSampler1D &sampler) {
    sampler.size = CopyToDevice(sampler.alias_table, m_alias_table);
    CopyToDevice(sampler.probability_table, m_probability_table);
}

void DiscreteSampler1D::FreeDevice(DeviceDiscreteSampler1D &sampler) {
    cudaFree(sampler.alias_table);
    cudaFree(sampler.probability_table);
}
