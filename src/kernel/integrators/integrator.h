#pragma once

#include <glm/glm.hpp>
#include "common.h"

enum class IntegratorType {
    Debug,
    Path
};

class Integrator {
public:
    Integrator() {}
    virtual ~Integrator() = default;

    virtual void Init(glm::ivec2 resolution) = 0;
    virtual void Destroy() = 0;
    virtual void Resize(glm::ivec2 resolution) = 0;
    virtual void Reset(glm::ivec2 resolution) = 0;

    virtual IntegratorType Type() = 0;

    virtual void Render(glm::ivec2 resolution, const DeviceScene &scene, uchar4 *data, int iteration, const IntegratorState &state) = 0;
};
