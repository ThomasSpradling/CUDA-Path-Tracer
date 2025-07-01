#include "PathTracer.h"

__global__ void PaintRed(uchar4* dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    dst[y * w + x] = make_uchar4(255, 0, 0, 255);
}

PathTracer::PathTracer(const VkExtent2D &extent)
    : m_extent(extent)
{}

void PathTracer::PathTrace(uchar4 *data) {
    dim3 block_size(16,16);
    dim3 block_count(
        (m_extent.width + block_size.x - 1) / block_size.x,
        (m_extent.height + block_size.y - 1) / block_size.y
    );

    PaintRed<<<block_count, block_size>>>(data, m_extent.width, m_extent.height);
}
