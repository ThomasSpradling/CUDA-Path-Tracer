#pragma once

#include "kernel/device_types.h"

class DiscreteSampler1D {
public:
    void UpdateWeights(const std::vector<float> &weights);

    void UpdateDevice(DeviceDiscreteSampler1D &sampler);
    void FreeDevice(DeviceDiscreteSampler1D &sampler);
private:
    std::vector<uint32_t> m_alias_table;
    std::vector<float> m_probability_table;
};
