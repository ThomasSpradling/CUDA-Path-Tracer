#pragma once

#include <vector_types.h>
#include <volk.h>

class PathTracer {
public:
    PathTracer(const VkExtent2D &extent);
    ~PathTracer() = default;

    void PathTrace(uchar4 *pbo);
private:
    const VkExtent2D &m_extent;
};
