#pragma once

#include "VulkanRenderer.h"

struct PathTracerProperties {
    RendererProperties renderer_properties;
};

class PathTracer {
public:
    PathTracer(const PathTracerProperties &props);

    void Run();
private:
    const PathTracerProperties &m_props;
    uint32_t m_iteration = 0;
    VulkanRenderer m_renderer;
};