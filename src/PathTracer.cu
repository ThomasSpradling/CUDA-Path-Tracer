#include <thrust/detail/sort.inl>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

#include "MathUtils.h"
#include "PathTracer.h"
#include "material.h"


#define GLM_ENABLE_EXPERIMENTAL 
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "exception.h"
#include "scene.h"
#include <cuda_runtime_api.h>
#include "Intersections.h"
#include <cuda.h>


__host__ __device__
thrust::default_random_engine MakeSeededRandomEngine(int iter, int index, int depth) {
    int h = Math::JenkinsHash((1 << 31) | (depth << 22) | iter) ^ Math::JenkinsHash(index);
    return thrust::default_random_engine(h);
}

__global__ void GenerateRayFromCamera(Camera cam, int iteration, int trace_depth, PathSegment *path_segments) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + y * cam.resolution.x;
        PathSegment &segment = path_segments[index];

        segment.ray.origin = cam.position;
        segment.throughput = glm::vec3(1.0f);

        thrust::default_random_engine rng = MakeSeededRandomEngine(iteration, index, trace_depth);
        glm::vec2 jitter = Math::Sample2D(rng);
        glm::vec2 pixel_sample = glm::vec2(x, y);
        pixel_sample += jitter;

        glm::vec3 direction = cam.front;
        direction -= cam.right * cam.pixel_length.x * (pixel_sample.x - (float)cam.resolution.x * 0.5f);
        direction -= cam.up * cam.pixel_length.y * (pixel_sample.y - (float)cam.resolution.y * 0.5f);

        segment.ray.direction = glm::normalize(direction);

        segment.pixel_index = index;
        segment.remaining_bounces = trace_depth;
        segment.color_accum = glm::vec3(0.0f);
        segment.hit = true;
    }
}
__global__ void ComputeIntersections(
    int depth,
    int num_paths,
    PathSegment *path_segments,
    Geometry *geometries,
    int geoms_size,
#if SORT_BY_MATERIAL
    int *sort_keys,
#endif
    MeshVertex *vertices,
    uint32_t *indices,
#if USE_BVH
    BVH::BVHNode *bvh_nodes,
    uint32_t *tri_indices,
#endif
    ShadableIntersection *intersections
) {
    int path_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (path_index >= num_paths)
        return;

    PathSegment path_segment = path_segments[path_index];

    float t;
    glm::vec3 intersect_point;
    glm::vec3 normal;
    float t_min = FLT_MAX;
    int hit_geom_index = -1;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    glm::vec3 temp_pos;

    bool hit_outside = true;

    for (int i = 0; i < geoms_size; i++) {
        Geometry &geom = geometries[i];

        if (geom.type == GeometryType::Cube) {
            t = BoxIntersectionTest(geom, path_segment.ray, tmp_intersect, tmp_normal, outside);
        } else if (geom.type == GeometryType::Sphere) {
            t = SphereIntersectionTest(geom, path_segment.ray, tmp_intersect, tmp_normal, outside);
        } else if (geom.type == GeometryType::GLTF_Primitive) {
#if USE_BVH
            t = IntersectBVH(geom, bvh_nodes, tri_indices, vertices, indices, path_segment.ray, tmp_intersect, tmp_normal, outside);
#else
            t = PrimitiveIntersectionTest(geom, vertices, indices, path_segment.ray, tmp_intersect, tmp_normal, outside);
#endif
        }

        if (t > 0.0f && t_min > t) {
            t_min = t;
            hit_geom_index = i;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
            hit_outside = outside;
        }
    }

    if (hit_geom_index == -1) {
        intersections[path_index].t = -1.0f;
        path_segments[path_index].hit = false;
        path_segments[path_index].remaining_bounces = 0;
        path_segments[path_index].pixel_index = -1;
#if SORT_BY_MATERIAL
        sort_keys[path_index] = -1;
#endif
    } else {
        intersections[path_index].t = t_min;
        intersections[path_index].material_id = geometries[hit_geom_index].material_id;
        intersections[path_index].pos = path_segments[path_index].ray(t_min);;
        intersections[path_index].normal = hit_outside
                                         ? normal
                                         : -normal;

        path_segments[path_index].hit = true;
#if SORT_BY_MATERIAL
        sort_keys[path_index] = intersections[path_index].material_id;
#endif
    }
}

__global__ void ShadeMaterial(
    int iteration,
    int num_paths,
    ShadableIntersection* shadable_intersections,
    PathSegment* path_segments,
    Material* materials
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    PathSegment path = path_segments[index];

    if (index >= num_paths)
        return;

    if (path_segments[index].remaining_bounces <= 0)
        return;

    ShadableIntersection intersection = shadable_intersections[index];
    if (intersection.t <= 1e-6) {
        path_segments[index].remaining_bounces = 0;
        return;
    }

    thrust::default_random_engine rng = MakeSeededRandomEngine(iteration, index, path_segments[index].remaining_bounces);
    Material material = materials[intersection.material_id];

    if (material.type == Material::Type::Light) {
        glm::vec3 radiance = material.base_color * 5.0f;
        // path_segments[index].throughput *= radiance;
        path_segments[index].color_accum += radiance * path_segments[index].throughput;
        path_segments[index].remaining_bounces = 0;
        return;
    } else {
        glm::vec3 w_out = -glm::normalize(path_segments[index].ray.direction);
        BSDFSample sample = material.Sample(intersection.normal, w_out, Math::Sample3D(rng));
        if (sample.type == InvalidSample) {
            path_segments[index].remaining_bounces = 0;
        } else if (sample.pdf < 1e-8f) {
            path_segments[index].remaining_bounces = 0;
        } else {
            float cos_theta = fabsf(glm::dot(intersection.normal, sample.w_in));
            float sign = glm::dot(sample.w_in, intersection.normal) > 0.0f ? 1.0f : -1.0f;
                
            path_segments[index].throughput *= sample.bsdf * cos_theta / sample.pdf;
            path_segments[index].ray = Ray::MakeOffseted(intersection.pos, sample.w_in);

            path_segments[index].remaining_bounces--;
        }
    }
}

__global__ void FinalGather(int path_count, glm::vec3 *image, PathSegment *iteration_paths) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= path_count)
        return;

    PathSegment &iteration_path = iteration_paths[index];
    if (iteration_paths->pixel_index >= 0 && iteration_path.remaining_bounces <= 0) {
        glm::vec3 &c = iteration_path.throughput;
        if (isnan(c.r) || isnan(c.g) || isnan(c.b))
            return;

        image[iteration_path.pixel_index] += iteration_path.color_accum;
    }

    // image[iteration_path.pixel_index] += iteration_path.color_accum;

}

__global__ void SendImageToVulkanBuffer(uchar4 *dest, glm::ivec2 resolution, int iteration, glm::vec3 *image) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + y * resolution.x;
        glm::vec3 pixel = image[index];

        int denom = iteration + 1;

        glm::ivec3 color;
        color.x = glm::clamp((int)(pixel.x / denom * 255.0), 0, 255);
        color.y = glm::clamp((int)(pixel.y / denom * 255.0), 0, 255);
        color.z = glm::clamp((int)(pixel.z / denom * 255.0), 0, 255);

        dest[index] = make_uchar4(color.x, color.y, color.z, 255);
    }
}

PathTracer::PathTracer(const VkExtent2D &extent, Scene &scene)
    : m_extent(extent)
    , m_scene(scene)
{
    Init();
}

PathTracer::~PathTracer() {
    Destroy();
}

void PathTracer::Reset() {
    const int pixelCount =
        m_scene.State().camera.resolution.x *
        m_scene.State().camera.resolution.y;

    cudaMemset(md_image, 0, pixelCount * sizeof(glm::vec3));
    cudaMemset(md_paths, 0, pixelCount * sizeof(PathSegment));
    cudaMemset(md_terminated, 0, pixelCount * sizeof(PathSegment));
    cudaMemset(md_intersections, 0, pixelCount * sizeof(ShadableIntersection));
}

void PathTracer::Init() {
    const Camera &cam = m_scene.State().camera;
    const int pixel_count = cam.resolution.x * cam.resolution.y;

    CUDA_CHECK(cudaMalloc(&md_image, pixel_count * sizeof(glm::vec3)));
    CUDA_CHECK(cudaMemset(md_image, 0, pixel_count * sizeof(glm::vec3)));

    CUDA_CHECK(cudaMalloc(&md_paths, pixel_count * sizeof(PathSegment)));
    m_paths_thrust = thrust::device_ptr<PathSegment>(md_paths);

    CUDA_CHECK(cudaMalloc(&md_terminated, pixel_count * sizeof(PathSegment)));
    m_terminated_thrust = thrust::device_ptr<PathSegment>(md_terminated);

    CUDA_CHECK(cudaMalloc(&md_materials, m_scene.Materials().size() * sizeof(Material)));
    CUDA_CHECK(cudaMemcpy(md_materials, m_scene.Materials().data(), m_scene.Materials().size() * sizeof(Material), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&md_intersections, pixel_count * sizeof(ShadableIntersection)));
    CUDA_CHECK(cudaMemset(md_intersections, 0, pixel_count * sizeof(ShadableIntersection)));
    m_intersection_thrust = thrust::device_ptr<ShadableIntersection>(md_intersections);

    CUDA_CHECK(cudaMalloc(&md_vertices, m_scene.VertexCount() * sizeof(MeshVertex)));
    CUDA_CHECK(cudaMemcpy(md_vertices, m_scene.Vertices().data(), m_scene.VertexCount() * sizeof(MeshVertex), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&md_indices, m_scene.IndexCount() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(md_indices, m_scene.Indices().data(), m_scene.IndexCount() * sizeof(uint32_t), cudaMemcpyHostToDevice));

#if SORT_BY_MATERIAL
    CUDA_CHECK(cudaMalloc(&md_keys, pixel_count * sizeof(uint32_t)));
    m_keys_thrust = thrust::device_ptr<int>(md_keys);
#endif

#if USE_BVH
    int geometry_count = m_scene.Geometries().size();
    for (int i = 0; i < geometry_count; ++i) {
        auto &geom = m_scene.Geometries()[i];
        if (geom.type != GeometryType::GLTF_Primitive)
            continue;
        
        BVH::BVH bvh(geom, m_scene.Vertices(), m_scene.Indices());
        bvh.Build();

        const uint32_t base_node = static_cast<uint32_t>(m_bvh_nodes.size());
        const uint32_t base_tri = static_cast<uint32_t>(m_bvh_tri_indices.size());

        for (const BVH::BVHNode &src : bvh.Nodes()) {
            BVH::BVHNode n = src;

            if (n.left_child != UINT32_MAX)
                n.left_child += base_node;

            m_bvh_nodes.push_back(n);
        }

        m_bvh_tri_indices.insert(m_bvh_tri_indices.end(), bvh.TriIndices().begin(), bvh.TriIndices().end());

        geom.first_bvh_node = base_node;
        geom.bvh_node_count = bvh.Nodes().size();

        geom.first_tri_index = base_tri;
        geom.tri_index_count = bvh.TriIndices().size();
    }

    CUDA_CHECK(cudaMalloc(&md_bvh_nodes, m_bvh_nodes.size() * sizeof(BVH::BVHNode)));
    CUDA_CHECK(cudaMemcpy(md_bvh_nodes, m_bvh_nodes.data(), m_bvh_nodes.size() * sizeof(BVH::BVHNode), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&md_bvh_tri_indices, m_bvh_tri_indices.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(md_bvh_tri_indices, m_bvh_tri_indices.data(), m_bvh_tri_indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
#endif

    CUDA_CHECK(cudaMalloc(&md_geometries, m_scene.Geometries().size() * sizeof(Geometry)));
    CUDA_CHECK(cudaMemcpy(md_geometries, m_scene.Geometries().data(), m_scene.Geometries().size() * sizeof(Geometry), cudaMemcpyHostToDevice));
}

void PathTracer::Destroy() {
    CUDA_CHECK(cudaFree(md_image));
    CUDA_CHECK(cudaFree(md_paths));
    CUDA_CHECK(cudaFree(md_terminated));
    CUDA_CHECK(cudaFree(md_geometries));
    CUDA_CHECK(cudaFree(md_materials));
    CUDA_CHECK(cudaFree(md_intersections));

#if SORT_BY_MATERIAL
    CUDA_CHECK(cudaFree(md_keys));
#endif
}

struct CompactTerminatedPaths {
    __host__ __device__
        bool operator()(const PathSegment& p) const {
        return !(p.pixel_index >= 0 && p.remaining_bounces <= 0);
    }
};

struct RemoveInvalidPaths {
    __host__ __device__
        bool operator()(const PathSegment& p) const {
        return (!p.hit) || (p.pixel_index < 0) || (p.remaining_bounces <= 0);
    }
};

void PathTracer::PathTrace(uchar4* data, int frame, int iteration) {
    const int trace_depth = m_scene.State().trace_depth;
    const Camera& cam = m_scene.State().camera;
    const int pixel_count = cam.resolution.x * cam.resolution.y;

    dim3 block_size(8, 8);
    dim3 block_count(
        (cam.resolution.x + block_size.x - 1) / block_size.x,
        (cam.resolution.y + block_size.y - 1) / block_size.y
    );

    const int block_size_1d = 128;

    using zip_it = thrust::zip_iterator<thrust::tuple<PathSegment*, ShadableIntersection*>>;

    GenerateRayFromCamera<<<block_count, block_size>>>(cam, iteration, trace_depth, md_paths);

    thrust::device_ptr<PathSegment> paths_begin = thrust::device_pointer_cast(md_paths);

    thrust::device_ptr<PathSegment> terminated_begin_base = thrust::device_pointer_cast(md_terminated);
    thrust::device_ptr<PathSegment> terminated_begin = thrust::device_pointer_cast(md_terminated);

    int num_paths = pixel_count;
    int depth = 0;

    bool iteration_complete = false;
    while (!iteration_complete) {
        cudaMemset(md_intersections, 0, pixel_count * sizeof(ShadableIntersection));

        dim3 num_blocks_segments = (num_paths + block_size_1d - 1) / block_size_1d;
        ComputeIntersections<<<num_blocks_segments, block_size_1d>>>(
            depth, num_paths, md_paths,
            md_geometries, m_scene.Geometries().size(),
#if SORT_BY_MATERIAL
            md_keys,
#endif
            md_vertices,
            md_indices,
#if USE_BVH
            md_bvh_nodes,
            md_bvh_tri_indices,
#endif
            md_intersections);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

#if SORT_BY_MATERIAL
        auto zipped = thrust::make_zip_iterator(thrust::make_tuple(m_intersection_thrust, m_paths_thrust));
        thrust::sort_by_key(m_keys_thrust, m_keys_thrust + num_paths, zipped);
        thrust::sort_by_key(m_keys_thrust, m_keys_thrust + num_paths, m_paths_thrust);
#endif

        ShadeMaterial<<<num_blocks_segments, block_size_1d>>>(iteration, num_paths, md_intersections, md_paths, md_materials);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        terminated_begin = thrust::remove_copy_if(
            thrust::device,
            paths_begin,
            paths_begin + num_paths,
            terminated_begin,
            CompactTerminatedPaths()
        );

        auto end = thrust::remove_if(
            thrust::device,
            paths_begin,
            paths_begin + num_paths,
            RemoveInvalidPaths()
        );

        num_paths = static_cast<std::size_t>(end - paths_begin);

        iteration_complete = (num_paths == 0) || (depth + 1 >= trace_depth);
        depth++;
    }

    num_paths = terminated_begin - terminated_begin_base;
    if (num_paths > 0) {
        dim3 num_blocks_pixels = (num_paths + block_size_1d - 1) / block_size_1d;
        FinalGather<<<num_blocks_pixels, block_size_1d>>>(num_paths, md_image, md_terminated);
    }
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    SendImageToVulkanBuffer<<<block_count, block_size>>>(data, cam.resolution, iteration, md_image);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(m_scene.State().image.Data(), md_image, pixel_count * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaGetLastError());
}

