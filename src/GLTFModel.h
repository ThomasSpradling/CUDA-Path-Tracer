#pragma once

#include <memory>
#include <vector>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "geometry.h"

#include "utils.h"

#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

struct MeshSettings {
    bool flat_shade = false;
};

struct ObjectSettings;

namespace GLTF {

    template <typename T> struct GltfType;
    template <> struct GltfType<float>      { static constexpr int gl_type = TINYGLTF_TYPE_SCALAR; static constexpr int comp_type = TINYGLTF_COMPONENT_TYPE_FLOAT; };
    template <> struct GltfType<glm::vec2>  { static constexpr int gl_type = TINYGLTF_TYPE_VEC2;   static constexpr int comp_type = TINYGLTF_COMPONENT_TYPE_FLOAT; };
    template <> struct GltfType<glm::vec3>  { static constexpr int gl_type = TINYGLTF_TYPE_VEC3;   static constexpr int comp_type = TINYGLTF_COMPONENT_TYPE_FLOAT; };
    template <> struct GltfType<glm::vec4>  { static constexpr int gl_type = TINYGLTF_TYPE_VEC4;   static constexpr int comp_type = TINYGLTF_COMPONENT_TYPE_FLOAT; };
    template <> struct GltfType<uint8_t>    { static constexpr int gl_type = TINYGLTF_TYPE_SCALAR; static constexpr int comp_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;  };
    template <> struct GltfType<uint16_t>   { static constexpr int gl_type = TINYGLTF_TYPE_SCALAR; static constexpr int comp_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT; };
    template <> struct GltfType<uint32_t>   { static constexpr int gl_type = TINYGLTF_TYPE_SCALAR; static constexpr int comp_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;  };


    struct Primitive {
        uint32_t first_index;
        uint32_t index_count;
        uint32_t vertex_count;
    };

    struct Mesh {
        std::vector<Primitive> primitives;
    };

    struct SceneNode {
        std::shared_ptr<Mesh> mesh;
        std::vector<std::shared_ptr<SceneNode>> children {};

        glm::vec3 translation {};
        glm::quat rotation {};
        glm::vec3 scale { 1, 1, 1 };

        glm::mat4 matrix { 1.0f };
        glm::mat4 world_transform { 1.0f };

        glm::mat4 LocalMatrix() {
            glm::mat4 main_mat = matrix;
            glm::mat4 trans_mat = glm::translate(glm::mat4(1.0f), translation);
            glm::mat4 rotation_mat = glm::mat4(rotation);
            glm::mat4 scale_mat = glm::scale(glm::mat4(1.0f), scale);

            glm::mat4 result = trans_mat * rotation_mat * scale_mat * main_mat;
            return result;
        }
    };

    class GLTFModel {
    public:
        GLTFModel(const MeshSettings &settings) : m_settings(settings) {};
        ~GLTFModel() = default;

        void LoadGLTF(const std::string &filename);

        size_t PrimitiveCount() const { return m_primitive_count; }
        void ForEachNode(const std::function<void(SceneNode &node)> &callback);

        const std::vector<MeshVertex> &Vertices() const { return m_vertices; } 
        const std::vector<uint32_t> &Indices() const { return m_indices; } 
    private:
        // std::vector<std::shared_ptr<SceneNode>> m_nodes;
        std::vector<std::shared_ptr<SceneNode>> m_root_nodes;
        // std::vector<std::shared_ptr<Mesh>> m_meshes;

        const MeshSettings &m_settings;

        std::vector<uint32_t> m_indices;
        std::vector<MeshVertex> m_vertices;
        size_t m_primitive_count;
    private:
        void LoadNode(const tinygltf::Model &model, const tinygltf::Scene &scene, const tinygltf::Node &parsed_node, std::shared_ptr<SceneNode> &node);

        template <typename T>
        static void VerifyAccessorCompatibility(const tinygltf::Accessor& accessor) {
            static_assert(std::is_trivially_copyable_v<T>, "ElementType must be a POD");
            assert(accessor.type == GltfType<T>::gl_type && "Accessor / ElementType shape mismatch");

            if constexpr (!std::is_arithmetic_v<T>) {
                assert(accessor.componentType == GltfType<T>::comp_type && "Accessor / ElementType component mismatch");
            }
        }

        template <typename T>
        static T ReadScalar(const uint8_t *data, int component_type) {
            switch (component_type) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                    return static_cast<T>(*reinterpret_cast<const uint8_t *>(data));
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                    return static_cast<T>(*reinterpret_cast<const uint16_t *>(data));
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                    return static_cast<T>(*reinterpret_cast<const uint32_t *>(data));
                case TINYGLTF_COMPONENT_TYPE_BYTE:
                    return static_cast<T>(*reinterpret_cast<const int8_t *>(data));
                case TINYGLTF_COMPONENT_TYPE_SHORT:
                    return static_cast<T>(*reinterpret_cast<const int16_t *>(data));
                case TINYGLTF_COMPONENT_TYPE_INT:
                    return static_cast<T>(*reinterpret_cast<const int32_t *>(data));
                case TINYGLTF_COMPONENT_TYPE_FLOAT:
                    return static_cast<T>(*reinterpret_cast<const float *>(data));
                default:
                    return T{};
            }
        }

        template <typename ElementType>
        void IterateAccessor(const tinygltf::Model &model,
                            const tinygltf::Accessor &accessor,
                            std::function<void(uint32_t, ElementType)> callback)
        {
            VerifyAccessorCompatibility<ElementType>(accessor);

            const tinygltf::BufferView &buffer_view = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer &buffer = model.buffers[buffer_view.buffer];

            const size_t stride = accessor.ByteStride(buffer_view) ?
                                accessor.ByteStride(buffer_view) :
                                (std::is_arithmetic_v<ElementType>
                                    ? tinygltf::GetComponentSizeInBytes(accessor.componentType)
                                    : sizeof(ElementType));

            const uint8_t *base = &buffer.data[buffer_view.byteOffset + accessor.byteOffset];

            auto read_dense = [&](uint32_t i) -> ElementType {
                if constexpr (std::is_arithmetic_v<ElementType>)
                    return ReadScalar<ElementType>(base + i * stride, accessor.componentType);
                else
                    return *reinterpret_cast<const ElementType *>(base + i * stride);
            };

            if (!accessor.sparse.isSparse) {
                for (uint32_t i = 0; i < accessor.count; ++i)
                    callback(i, read_dense(i));
                return;
            }

            const auto &sparse = accessor.sparse;

            const tinygltf::BufferView &sparse_index_buffer_view = model.bufferViews[sparse.indices.bufferView];
            const tinygltf::Buffer &sparse_index_buffer = model.buffers[sparse_index_buffer_view.buffer];
            const uint8_t *index_data = &sparse_index_buffer.data[sparse_index_buffer_view.byteOffset + sparse.indices.byteOffset];

            const tinygltf::BufferView &value_buffer_view = model.bufferViews[sparse.values.bufferView];
            const tinygltf::Buffer &value_buffer = model.buffers[value_buffer_view.buffer];
            const uint8_t *val_base = &value_buffer.data[value_buffer_view.byteOffset + sparse.values.byteOffset];

            uint32_t sparse_cursor = 0;

            for (uint32_t i = 0; i < accessor.count; ++i) {
                uint32_t next_sparse = (sparse_cursor < sparse.count)
                    ? ReadScalar<uint32_t>(
                        index_data + sparse_cursor *
                        tinygltf::GetComponentSizeInBytes(sparse.indices.componentType),
                        sparse.indices.componentType)
                    : UINT32_MAX;

                if (i == next_sparse) {
                    ElementType val;
                    if constexpr (std::is_arithmetic_v<ElementType>) {
                        val = ReadScalar<ElementType>(
                            val_base + sparse_cursor * tinygltf::GetComponentSizeInBytes(accessor.componentType),
                            accessor.componentType
                        );
                    }
                    else {
                        val = *reinterpret_cast<const ElementType *>(
                            val_base + sparse_cursor * sizeof(ElementType));

                    }

                    callback(i, val);
                    ++sparse_cursor;
                }
                else {
                    callback(i, read_dense(i));
                }
            }
        }
    };

}
