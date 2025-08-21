#pragma once

#include <cuda_device_runtime_api.h>
#include <glm/glm.hpp>
#include "common.h"
#include "integrator.h"

class DebugIntegrator : public Integrator {
public:
    DebugIntegrator();
    ~DebugIntegrator();

    virtual void Init(glm::ivec2 resolution) override;
    virtual void Destroy() override;
    virtual void Resize(glm::ivec2 resolution) override;
    virtual void Reset(glm::ivec2 resolution) override;

    virtual inline IntegratorType Type() override { return IntegratorType::Debug; }

    virtual void Render(glm::ivec2 resolution, const DeviceScene &scene, uchar4 *data, int iteration, const IntegratorState &state) override;
private:
    glm::vec3 *m_image;
    DeviceScene *m_dev_scene;
private:
    bool LaunchDebugKernels(glm::ivec2 resolution, const DeviceScene &scene, uchar4 *data, int iteration, const IntegratorState &state);
};
