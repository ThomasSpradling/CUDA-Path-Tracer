#include "GLTFModel.h"
#include "exception.h"
#include <stack>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

namespace GLTF {

    void GLTFModel::LoadGLTF(const std::string &filename) {
        fs::path path { filename };
    
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;

        std::string warnings;
        std::string errors;

        std::string ext = path.extension().string();
        ext = ToLowercase(ext);

        bool ret = false;
        if (ext == ".glb") {
            ret = loader.LoadBinaryFromFile(&model, &errors, &warnings, path.string());
        } else if (ext == ".gltf") {
            ret = loader.LoadASCIIFromFile(&model, &errors, &warnings, path.string());
        } else {
            PT_ERROR("Invalid gltf filename!");
        }

        if (!warnings.empty()) {
            PT_ERROR(std::format("glTF Warning: {}", warnings));
        }

        if (!errors.empty()) {
            PT_ERROR(std::format("glTF Error: {}", errors));
        }

        uint32_t default_scene = model.defaultScene == -1 ? 0 : model.defaultScene;
        const tinygltf::Scene &scene = model.scenes[default_scene];

        m_root_nodes.resize(scene.nodes.size());
        for (uint32_t i = 0; i < scene.nodes.size(); ++i) {
            const tinygltf::Node &node = model.nodes[scene.nodes[i]];
            LoadNode(model, scene, node, m_root_nodes[i]);
        }
    }

    void GLTFModel::ForEachNode(const std::function<void(SceneNode &node)> &callback) {
        std::stack<std::shared_ptr<SceneNode>> stack;

        for (const auto &node : m_root_nodes) {
            stack.push(node);
        }

        while (!stack.empty()) {
            std::shared_ptr<SceneNode> current_node = stack.top();
            stack.pop();

            callback(*current_node);

            for (const auto &child : current_node->children) {
                if (child)
                    stack.push(child);
            }
        }
    }

    void GLTFModel::LoadNode(const tinygltf::Model &model, const tinygltf::Scene &scene, const tinygltf::Node &parsed_node, std::shared_ptr<SceneNode> &node) {
        node = std::make_shared<SceneNode>();

        if (!parsed_node.translation.empty())
            node->translation = glm::make_vec3(parsed_node.translation.data());
        if (!parsed_node.rotation.empty())
            node->rotation = glm::make_quat(parsed_node.rotation.data());
        if (!parsed_node.scale.empty())
            node->scale = glm::make_vec3(parsed_node.scale.data());
        if (!parsed_node.matrix.empty())
            node->matrix = glm::make_mat4(parsed_node.matrix.data());
        
        if (parsed_node.mesh != -1) {
            // has mesh

            node->mesh = std::make_shared<Mesh>();
            const tinygltf::Mesh &mesh = model.meshes[parsed_node.mesh];

            for (const auto &primitive : mesh.primitives) {
                m_primitive_count++;

                if (!primitive.attributes.contains("POSITION")) {
                    throw std::runtime_error("Invalid glTF model: All models must have position attributes.");
                }
                
                //// Position ////
                const tinygltf::Accessor &position_accessor = model.accessors[primitive.attributes.at("POSITION")];
                std::vector<glm::vec3> positions {};
                positions.reserve(position_accessor.count);

                IterateAccessor<glm::vec3>(model, position_accessor, [&](uint32_t i, const glm::vec3 &value) {
                    positions.push_back(value);
                });
                uint32_t vertex_count = positions.size();

                //// Normals ////
                std::vector<glm::vec3> normals(vertex_count, glm::vec3(0.0f));
                if (!m_settings.flat_shade && primitive.attributes.contains("NORMAL")) {
                    const tinygltf::Accessor &normal_accessor = model.accessors[primitive.attributes.at("NORMAL")];
                    IterateAccessor<glm::vec3>(model, normal_accessor, [&](uint32_t i, const glm::vec3 &value) {
                        normals[i] = value;
                    });
                }

                //// UVs ////
                std::vector<glm::vec2> uv0s(vertex_count, glm::vec2(0.0f));
                if (primitive.attributes.contains("TEXCOORD_0")) {
                    const tinygltf::Accessor &uv0_accessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];
                    IterateAccessor<glm::vec2>(model, uv0_accessor, [&](uint32_t i, const glm::vec2 &value) {
                        uv0s[i] = value;
                    });
                }
                
                //// Colors ////
                std::vector<glm::vec4> colors(vertex_count, glm::vec4(1.0f));
                if (primitive.attributes.contains("COLOR_0")) {
                    const tinygltf::Accessor &color_accessor = model.accessors[primitive.attributes.at("COLOR_0")];
                    IterateAccessor<glm::vec3>(model, color_accessor, [&](uint32_t i, const glm::vec3 &value) {
                        colors[i] = glm::vec4(value, 1.0f);
                    });
                }

                //// Indices ////
                uint32_t index_count = 0;
                uint32_t first_index = static_cast<uint32_t>(m_indices.size());
                int32_t base_vertex = static_cast<uint32_t>(m_vertices.size());
                
                if (primitive.indices != -1) {
                    const tinygltf::Accessor &accessor = model.accessors[primitive.indices];
                    index_count = static_cast<uint32_t>(accessor.count);
                    
                    IterateAccessor<uint32_t>(model, accessor, [&](uint32_t i, uint32_t idxValue) {
                        m_indices.push_back(idxValue + base_vertex);
                    });
                } else {
                    index_count = vertex_count;
                    for (uint32_t k = 0; k < index_count; ++k)
                        m_indices.push_back(base_vertex + k);
                }

                for (uint32_t v = 0; v < vertex_count; ++v) {
                    MeshVertex vertex {
                        .position = positions[v],
                        .normal = normals[v],
                        .uv0 = uv0s[v],
                        .color = colors[v],
                    };
                    m_vertices.push_back(vertex);
                }

                Primitive prim {
                    .first_index = first_index,
                    .index_count = index_count,
                    .vertex_count = vertex_count,
                };

                node->mesh->primitives.push_back(prim);
            }
        }

        node->children.resize(parsed_node.children.size());
        for (uint32_t i = 0; i < parsed_node.children.size(); ++i) {
            LoadNode(model, scene, model.nodes[parsed_node.children[i]], node->children[i]);
        }
    }

}
