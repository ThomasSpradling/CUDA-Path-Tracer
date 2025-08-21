#include "gltf_model.h"
#include "../utils/utils.h"
#include "../utils/exception.h"
#include <stack>
#include "../utils/stopwatch.h"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

namespace GLTF {

    glm::vec3 _GetVec3(std::vector<double> &vec) {
        if (vec.size() < 3) {
            std::cerr << "GLTF Loading: Cannot convert to Vec3" << std::endl;
            return glm::vec3(0.0f);
        }

        return { vec[0], vec[1], vec[2] };
    }
    void GLTFModel::LoadGLTF(const std::string &filename) {
        StopWatch load_timer;
        load_timer.Start();

        fs::path path { filename };
    
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;

        std::string warnings;
        std::string errors;

        std::string ext = path.extension().string();
        ext = Utils::ToLowercase(ext);

        bool ret = false;
        if (ext == ".glb") {
            ret = loader.LoadBinaryFromFile(&model, &errors, &warnings, path.string());
        } else if (ext == ".gltf") {
            ret = loader.LoadASCIIFromFile(&model, &errors, &warnings, path.string());
        } else {
            PT_ERROR("Invalid gltf filename!");
        }

        if (!warnings.empty()) {
            std::cerr << std::format("glTF warning: {}", warnings) << std::endl;
        }

        if (!errors.empty()) {
            PT_ERROR(std::format("glTF Error: {}", errors));
        }

        if (model.scenes.empty()) {
            PT_ERROR("glTF Model has no scenes!");
        }

        uint32_t default_scene = model.defaultScene == -1 ? 0 : model.defaultScene;
        const tinygltf::Scene &scene = model.scenes[default_scene];

        m_meshes.resize(model.meshes.size());
        for (uint32_t i = 0; i < model.meshes.size(); ++i) {
            const tinygltf::Mesh &mesh = model.meshes[i];
            m_meshes[i] = std::make_shared<PrimMesh>();
            m_meshes[i]->mesh_id = i;
            LoadMesh(model, mesh, *m_meshes[i]);
        }

        m_root_nodes.resize(scene.nodes.size());
        for (uint32_t i = 0; i < scene.nodes.size(); ++i) {
            const tinygltf::Node &node = model.nodes[scene.nodes[i]];
            LoadNode(model, scene, node, m_root_nodes[i], glm::mat4(1.0f));
        }

        // -- Load materials --------

        // Load each texture into the texture pool, using images as needed
        // use map to make sure we are not dupping

        // cache to avoid multiple textures that refer to same image
        std::unordered_map<int, int> albedo_map, metallic_map, roughness_map;
        for (const auto &mat : model.materials) {
            Material material;
            material.type = Material::MetallicRoughness;

            if (mat.pbrMetallicRoughness.baseColorTexture.index != -1) {
                const auto &albedo_texture = model.textures[mat.pbrMetallicRoughness.baseColorTexture.index];

                // -- Albedo texture --------
                if (!albedo_map.contains(albedo_texture.source)) {
                    int albedo_tex_id = m_texture_pool.textures3.size();
                    int albedo_image_id = m_texture_pool.images3.size();

                    // Emplace albedo texture
                    m_texture_pool.images3.emplace_back();
                    Image3 &image = m_texture_pool.images3.back();
                    image.SetColorSpace(ColorSpace::sRGB);
                    ParseImage(image, model.images[albedo_texture.source]);

                    Texture<glm::vec3> texture;
                    texture.type = TextureType::Image;
                    texture.image.image_id = albedo_image_id;
                    texture.image.scale = glm::vec2(1.0f);
                    texture.image.offset = glm::vec2(0.0f);
                    texture.image.width = image.Width();
                    texture.image.height = image.Height();
                    material.metallic_roughness.albedo_texture = albedo_tex_id;

                    m_texture_pool.textures3.push_back(texture);

                    albedo_map[albedo_texture.source] = albedo_tex_id;
                } else {
                    material.metallic_roughness.albedo_texture = albedo_map[albedo_texture.source];
                }
            }

            if (mat.pbrMetallicRoughness.metallicRoughnessTexture.index != -1) {
                const auto &pbr_texture = model.textures[mat.pbrMetallicRoughness.metallicRoughnessTexture.index];
                // -- Metallic texture --------
                if (!metallic_map.contains(pbr_texture.source)) {
                    int metallic_tex_id = m_texture_pool.textures1.size();
                    int metallic_image_id = m_texture_pool.images1.size();
                
                    // Emplace B component of PBR texture
                    m_texture_pool.images1.emplace_back();
                    Image1 &image = m_texture_pool.images1.back();
                    ParseImage(image, model.images[pbr_texture.source], 2);

                    Texture<float> texture;
                    texture.type = TextureType::Image;
                    texture.image.image_id = metallic_image_id;
                    texture.image.scale = glm::vec2(1.0f);
                    texture.image.offset = glm::vec2(0.0f);
                    texture.image.width = image.Width();
                    texture.image.height = image.Height();
                    material.metallic_roughness.metallic_texture = metallic_tex_id;

                    m_texture_pool.textures1.push_back(texture);
                    metallic_map[pbr_texture.source] = metallic_tex_id;
                } else {
                    material.metallic_roughness.metallic_texture = metallic_map[pbr_texture.source];
                }

                // -- Roughness texture --------
                if (!roughness_map.contains(pbr_texture.source)) {
                    int roughness_tex_id = m_texture_pool.textures1.size();
                    int roughness_image_id = m_texture_pool.images1.size();
                
                    // Emplace G component of PBR texture
                    m_texture_pool.images1.emplace_back();
                    Image1 &image = m_texture_pool.images1.back();
                    ParseImage(image, model.images[pbr_texture.source], 1);

                    Texture<float> texture;
                    texture.type = TextureType::Image;
                    texture.image.image_id = roughness_image_id;
                    texture.image.scale = glm::vec2(1.0f);
                    texture.image.offset = glm::vec2(0.0f);
                    texture.image.width = image.Width();
                    texture.image.height = image.Height();
                    material.metallic_roughness.roughness_texture = roughness_tex_id;

                    m_texture_pool.textures1.push_back(texture);
                    roughness_map[pbr_texture.source] = roughness_tex_id;
                } else {
                    material.metallic_roughness.roughness_texture = roughness_map[pbr_texture.source];
                }
            }

            m_materials.push_back(material);
        }

        load_timer.Finish(std::format("Loaded model '{}' [{:.2f} MB]", fs::path(filename).filename().string(), Utils::FileSize(filename)));
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

    void GLTFModel::LoadMesh(const tinygltf::Model &model, const tinygltf::Mesh &parsed_mesh, PrimMesh &mesh) {
        for (const auto &primitive : parsed_mesh.primitives) {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                std::cerr << "glTF Loader: skipping non-triangle primitive (mode=" << primitive.mode << ")\n";
                continue;
            }

            m_primitive_count++;

            if (!primitive.attributes.contains("POSITION")) {
                throw std::runtime_error("Invalid glTF model: All models must have position attributes.");
            }
                
            // -- Position --------
            const tinygltf::Accessor &position_accessor = model.accessors[primitive.attributes.at("POSITION")];
            std::vector<glm::vec3> positions {};
            positions.reserve(position_accessor.count);

            IterateAccessor<glm::vec3>(model, position_accessor, [&](uint32_t i, const glm::vec3 &value) {
                positions.push_back(value);
            });
            uint32_t vertex_count = positions.size();

            // -- Normals --------
            bool use_face_normals = m_settings.face_normals || !primitive.attributes.contains("NORMAL");
            std::vector<glm::vec3> normals(vertex_count, glm::vec3(0.0f));
            if (!use_face_normals) {
                const tinygltf::Accessor &normal_accessor = model.accessors[primitive.attributes.at("NORMAL")];
                IterateAccessor<glm::vec3>(model, normal_accessor, [&](uint32_t i, const glm::vec3 &value) {
                    normals[i] = glm::normalize(value);
                });
            }

            // -- Texture coords --------
            std::vector<glm::vec2> uv0s(vertex_count, glm::vec2(0.0f));
            if (primitive.attributes.contains("TEXCOORD_0")) {
                const tinygltf::Accessor &uv0_accessor = model.accessors[primitive.attributes.at("TEXCOORD_0")];
                IterateAccessor<glm::vec2>(model, uv0_accessor, [&](uint32_t i, const glm::vec2 &value) {
                    uv0s[i] = value;
                });
            }

                // -- Indices --------

            uint32_t index_count = 0;
            uint32_t first_index = static_cast<uint32_t>(m_mesh.indices.size());
            uint32_t base_vertex = static_cast<uint32_t>(m_mesh.positions.size());

            if (!use_face_normals) {
                if (primitive.indices != -1) {
                    const auto &accessor = model.accessors[primitive.indices];
                    index_count = static_cast<uint32_t>(accessor.count);

                    IterateAccessor<uint32_t>(model, accessor, [&](uint32_t, uint32_t idxValue) {
                        m_mesh.indices.push_back(idxValue + base_vertex);
                    });
                } else {
                    index_count = vertex_count;
                    for (uint32_t k = 0; k < index_count; ++k)
                        m_mesh.indices.push_back(base_vertex + k);
                }

                for (uint32_t v = 0; v < vertex_count; ++v) {
                    m_mesh.positions.push_back(positions[v]);
                    m_mesh.normals.push_back(normals[v]);
                    m_mesh.texcoords.push_back(uv0s[v]);
                }

            } else {
                std::vector<uint32_t> idxData;
                if (primitive.indices != -1) {
                    const auto &accessor = model.accessors[primitive.indices];
                    index_count = static_cast<uint32_t>(accessor.count);
                    idxData.reserve(index_count);
                    IterateAccessor<uint32_t>(model, accessor, [&](uint32_t, uint32_t x){ idxData.push_back(x); });
                } else {
                    index_count = vertex_count;
                    idxData.resize(index_count);
                    std::iota(idxData.begin(), idxData.end(), 0u);
                }

                for (size_t t = 0; t + 2 < idxData.size(); t += 3) {
                    uint32_t i0 = idxData[t+0],
                                i1 = idxData[t+1],
                                i2 = idxData[t+2];

                    const glm::vec3 &p0 = positions[i0];
                    const glm::vec3 &p1 = positions[i1];
                    const glm::vec3 &p2 = positions[i2];
                    glm::vec3 fn = glm::normalize(glm::cross(p1 - p0, p2 - p0));

                    auto emit = [&](uint32_t pi){
                        m_mesh.positions.push_back(positions[pi]);
                        m_mesh.normals.push_back(fn);
                        m_mesh.texcoords.push_back(uv0s[pi]);
                        m_mesh.indices.push_back(static_cast<uint32_t>(m_mesh.positions.size()) - 1);
                    };

                    emit(i0);
                    emit(i1);
                    emit(i2);
                }
            }
                
            if (m_settings.invert_normals) {
                for (auto &n : m_mesh.normals)
                    n = -n;
            }

            Primitive prim {
                .first_index = first_index,
                .index_count = index_count,
                .vertex_count = use_face_normals ? index_count : vertex_count,
            };
            prim.material_id = primitive.material >= 0 ? primitive.material : 0;

            mesh.primitives.push_back(prim);
        }
    }

    void GLTFModel::LoadNode(const tinygltf::Model &model, const tinygltf::Scene &scene, const tinygltf::Node &parsed_node, std::shared_ptr<SceneNode> &node, const glm::mat4 &parent_world) {
        node = std::make_shared<SceneNode>();

        if (!parsed_node.translation.empty())
            node->translation = glm::make_vec3(parsed_node.translation.data());
        if (!parsed_node.rotation.empty())
            node->rotation = glm::make_quat(parsed_node.rotation.data());
        if (!parsed_node.scale.empty())
            node->scale = glm::make_vec3(parsed_node.scale.data());
        if (!parsed_node.matrix.empty())
            node->matrix = glm::make_mat4(parsed_node.matrix.data());
        
        node->world_transform = parent_world * node->LocalMatrix(); 

        if (parsed_node.mesh != -1) {
            // has mesh
            node->mesh = m_meshes[parsed_node.mesh];
        }

        node->children.resize(parsed_node.children.size());
        for (uint32_t i = 0; i < parsed_node.children.size(); ++i) {
            const tinygltf::Node &child = model.nodes[parsed_node.children[i]];
            LoadNode(model, scene, child, node->children[i], node->world_transform);
        }
    }
}
