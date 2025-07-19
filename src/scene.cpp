#include "Scene.h"
#include "CudaUtils.h"
#include "GLTFModel.h"
#include "Light.h"
#include "Materials/Lambertian.h"
#include "Materials/RoughDielectric.h"
#include "Texture.h"
#include "exception.h"
#include "glm/gtc/matrix_inverse.hpp"
#include "material.h"
#include "samplers.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

// class Scene {
// public:
//     Scene(const std::string &filename);
//     ~Scene();
// private:
//     std::vector<Geometry> m_geometry;
//     std::vector<Material> m_materials;
//     RenderState m_state;
// private:
//     void LoadFromJSON(const std::string &name);
// };

Scene::Scene(const std::string &filename) {
    std::cout << "Reading scene from " << filename << std::endl;
    std::string ext = fs::path(filename).extension().string();
    PT_ASSERT(ext == ".json", std::format("Couldn't read from file '{}'.", filename));

    LoadFromJSON(filename);
}

void Scene::LoadFromJSON(const std::string &filename) {
    fs::path scene_path = filename;
    fs::path scene_dir = scene_path.parent_path();
    m_scene_dir = scene_dir;

    std::ifstream file(scene_path);

    PT_ASSERT(file.is_open(),
        std::format("ERROR: Could not open file '{}'.", filename));

    json data = json::parse(file);
    const auto &material_data = data["Materials"];
    m_materials.reserve(material_data.size());

    std::unordered_map<std::string, uint32_t> material_ids;

    for (const auto &[key, value] : material_data.items()) {
        uint32_t mat_id = static_cast<uint32_t>(m_materials.size());
        material_ids[key] = mat_id;

        m_materials.emplace_back();  
        Material &material = m_materials.back();

        if (value["TYPE"] == "Diffuse" || value["TYPE"] == "Lambertian") {
            material.type = Material::Type::Lambertian;
            new (&material.mat.lambert) Lambertian{};

            LoadTexture(material.mat.lambert.albedo, value["ALBEDO"]);

        } else if (value["TYPE"] == "Emitting" || value["TYPE"] == "Light") {
            const auto &color = value["COLOR"];

            new (&material.mat.light) Light{};
            material.type = Material::Type::Light;

            material.mat.light = Light{
                .base_color = glm::vec3(color[0], color[1], color[2]),
                .emittance = value["EMITTANCE"].is_number() ? value["EMITTANCE"].get<float>() : 1.0f,
            };

        } else if (value["TYPE"] == "Dielectric") {

            new (&material.mat.dielectric) PerfectDielectric{};
            material.type = Material::Type::Dielectric;

            LoadTexture(material.mat.dielectric.albedo, value["ALBEDO"]);
            material.mat.dielectric.ior = value["INDEX_OF_REFRACTION"];

        } else if (value["TYPE"] == "Mirror") {

            new (&material.mat.mirror) PerfectMirror{};
            material.type = Material::Type::Mirror;

            LoadTexture(material.mat.mirror.albedo, value["ALBEDO"]);

        } else if (value["TYPE"] == "MetallicRoughness") {

            new (&material.mat.metallic_roughness) MetallicRoughness{};
            material.type = Material::Type::Metallic_Roughness;

            LoadTexture(material.mat.metallic_roughness.albedo, value["ALBEDO"]);
            LoadTexture(material.mat.metallic_roughness.metallic_map, value["METALLIC"]);
            LoadTexture(material.mat.metallic_roughness.roughness_map, value["ROUGHNESS"]);
        } else if (value["TYPE"] == "RoughDielectric") {        
            
            new (&material.mat.rough_dielectric) RoughDielectric{};
            material.type = Material::Type::RoughDielectric;

            LoadTexture(material.mat.rough_dielectric.albedo, value["ALBEDO"]);
            LoadTexture(material.mat.rough_dielectric.roughness_map, value["ROUGHNESS"]);
            material.mat.dielectric.ior = value["INDEX_OF_REFRACTION"];
        }
        //  else if (value["TYPE"] == "Glass") {
        //     const auto &color = value["ALBEDO"];
        //     material.base_color = glm::vec3(color[0], color[1], color[2]);
        //     material.has_refractive = true;
        //     material.index_of_refraction = value["INDEX_OF_REFRACTION"];


        // } else if (value["TYPE"] == "Dielectric") {
        //     const auto &color = value["RGB"];
        //     material.color = glm::vec3(color[0], color[1], color[2]);
        //     material.has_reflective = true;
        //     material.has_refractive = true;
        //     material.index_of_refraction = value["INDEX_OF_REFRACTION"];

        //     std::cout << "DI: "<< material.index_of_refraction << std::endl;
        // }

        // material_ids[key] = m_materials.size();
        // m_materials.emplace_back(material);
    }

    const auto &object_data = data["Objects"];
    for (const auto &parsed_object : object_data) {
        const auto &type = parsed_object["TYPE"];
        Geometry geometry;
        if (type == "cube") {
            geometry.type = GeometryType::Cube;
        } else if (type == "sphere") {
            geometry.type = GeometryType::Sphere;
        } else if (type == "gltf") {
            geometry.type = GeometryType::TriangleMesh;
        } else {
            PT_ERROR("Invalid geometry type!");
        }

        MeshSettings obj_settings;

        if (parsed_object.contains("SETTINGS")) {
            if (!parsed_object["SETTINGS"].is_array()) {
                std::cerr << "Scene Warning: object contains SETTINGS but SETTINGS is not an array." << std::endl;
            }

            const auto &settings = parsed_object["SETTINGS"];
            for (const auto &setting : settings) {
                if (!setting.is_string()) {
                    continue;
                }

                if (setting == "FLAT_SHADE") {
                    if (geometry.type != GeometryType::TriangleMesh) {
                        std::cerr << "Scene Warning: Setting 'FLAT_SHADE' is incompatible with object of type '" << type << "'" << std::endl;
                    }

                    obj_settings.flat_shade = true;
                }
            }
        }

        geometry.material_id = material_ids[parsed_object["MATERIAL"]];
        const auto &trans = parsed_object["TRANS"];
        const auto &rotat = parsed_object["ROTAT"];
        const auto &scale = parsed_object["SCALE"];

        geometry.translation = glm::vec3(trans[0], trans[1], trans[2]);
        geometry.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        geometry.scale = glm::vec3(scale[0], scale[1], scale[2]);

        glm::mat4 trans_mat = glm::translate(glm::mat4(1.0f), geometry.translation);
        glm::mat4 rot_mat = glm::mat4(1.0f);
        rot_mat *= glm::rotate(glm::mat4(1.0f), glm::radians(geometry.rotation.x), glm::vec3(1, 0, 0));
        rot_mat *= glm::rotate(glm::mat4(1.0f), glm::radians(geometry.rotation.y), glm::vec3(0, 1, 0));
        rot_mat *= glm::rotate(glm::mat4(1.0f), glm::radians(geometry.rotation.z), glm::vec3(0, 0, 1));
        glm::mat4 scale_mat = glm::scale(glm::mat4(1.0f), geometry.scale);
        
        geometry.transform = trans_mat * rot_mat * scale_mat;
        geometry.inv_transform = glm::inverse(geometry.transform);
        geometry.inv_transpose = glm::inverseTranspose(geometry.transform);

        if (geometry.type != GeometryType::TriangleMesh) {
            m_geometry.push_back(geometry);
        } else {
            GLTF::GLTFModel model(obj_settings);
            const std::string &file = parsed_object["PATH"];
            if (file.empty()) {
                PT_ERROR("Missing PATH for gLTF object type!");
            }

            fs::path model_path = file;
            if (model_path.is_relative()) {
                model_path = scene_dir / model_path;
            }
            if (!fs::exists(model_path))
                PT_ERROR("glTF file not found: " + model_path.string());

            model.LoadGLTF(model_path.string());

            uint32_t base_vertex = static_cast<uint32_t>(m_vertices.size());
            uint32_t base_index = static_cast<uint32_t>(m_indices.size());
            m_vertices.insert(m_vertices.end(), model.Vertices().begin(), model.Vertices().end());
            for (auto idx : model.Indices())
                m_indices.push_back(idx + base_vertex);

            model.ForEachNode([&](GLTF::SceneNode &node) {
                if (!node.mesh)
                    return;

                glm::mat4 node_transform = geometry.transform * node.LocalMatrix();
                for (auto &primitive : node.mesh->primitives) {
                    Geometry geom_copy = geometry;
                    TriangleMesh tri_mesh {
                        .first_index = base_index + primitive.first_index,
                        .index_count = primitive.index_count,
                        .first_vertex = base_vertex,
                        .vertex_count = primitive.vertex_count,
                    };
                    geom_copy.transform = node_transform;
                    geom_copy.inv_transform = glm::inverse(node_transform);
                    geom_copy.inv_transpose = glm::inverseTranspose(node_transform);
                    geom_copy.mesh = tri_mesh;
                    m_geometry.push_back(geom_copy);
                    // std::cout << "ADDED geometry prim" << std::endl;
                }
            });

            
        }
    }

    const auto &camera_data = data["Camera"];
    Camera &camera = m_state.camera;
    RenderState &state = m_state;

    camera.resolution = { camera_data["RES"][0], camera_data["RES"][1] };
    float fovy = camera_data["FOVY"];
    state.iterations = camera_data["ITERATIONS"];
    state.trace_depth = camera_data["DEPTH"];
    state.output_name = camera_data["FILE"];
    const auto &pos = camera_data["EYE"];
    const auto &lookat = camera_data["LOOKAT"];
    const auto &up = camera_data["UP"];

    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.look_at = glm::vec3(lookat[0], lookat[1], lookat[2]);
    // camera.up = glm::vec3(up[0], up[1], up[2]);

    float yscaled = glm::tan(glm::radians(fovy));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;

    camera.front = glm::normalize(camera.look_at - camera.position);
    camera.right = glm::normalize(glm::cross(glm::vec3(0,1,0), camera.front));
    camera.up = glm::cross(camera.front, camera.right);

    camera.pixel_length = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    state.image.SetSize(camera.resolution.x, camera.resolution.y);

    camera.default_params.front = camera.front;
    camera.default_params.position = camera.position;
    camera.default_params.look_at = camera.look_at;

    InitGeometry();
    UploadToGPU();

    std::cout << "Finished loading scene!\n";
}

void Scene::UploadToGPU() {
    m_view.geometry_count = CopyToDevice(m_view.geometries, m_geometry);
    m_view.material_count = CopyToDevice(m_view.materials, m_materials);
    m_view.vertex_count = CopyToDevice(m_view.vertices, m_vertices);
    m_view.index_count = CopyToDevice(m_view.indices, m_indices);

    m_view.bvh_node_count = CopyToDevice(m_view.bvh_nodes, m_bvh_nodes);
    m_view.bvh_indices_count = CopyToDevice(m_view.bvh_tri_indices, m_bvh_tri_indices);
    m_view.light_count = CopyToDevice(m_view.lights, m_area_lights);
}

void Scene::InitGeometry() {
    int geometry_count = m_geometry.size();
    float scene_light_power = 0.0f;
    
    for (int i = 0; i < geometry_count; ++i) {
        auto &geom = m_geometry[i];
        if (m_materials[geom.material_id].type == Material::Type::Light) {
            float emittance = m_materials[geom.material_id].mat.light.emittance;
            switch (geom.type) {
                case GeometryType::TriangleMesh: {
                    float total_area = 0.0f;
                    std::vector<float> areas;
                    areas.reserve(geom.mesh.index_count / 3);
                    for (uint32_t j = 0; j < geom.mesh.index_count; j += 3) {
                        uint32_t j0 = m_indices[geom.mesh.first_index + j];
                        uint32_t j1 = m_indices[geom.mesh.first_index + j+1];
                        uint32_t j2 = m_indices[geom.mesh.first_index + j+2];
                        glm::vec3 p0 = m_vertices[j0].position;
                        glm::vec3 p1 = m_vertices[j1].position;
                        glm::vec3 p2 = m_vertices[j2].position;

                        glm::vec3 w0 = glm::vec3(geom.transform * glm::vec4(p0,1.0f));
                        glm::vec3 w1 = glm::vec3(geom.transform * glm::vec4(p1,1.0f));
                        glm::vec3 w2 = glm::vec3(geom.transform * glm::vec4(p2,1.0f));

                        glm::vec3 e0 = w1 - w0;
                        glm::vec3 e1 = w2 - w0;
                        float tri_area = 0.5f * glm::length(glm::cross(e0, e1));

                        total_area += tri_area;
                        areas.push_back(tri_area);
                    }
                    DiscreteSampler1D triangle_sampler(areas);
                    m_geometry[i].triangle_sampler = triangle_sampler.View();
                    
                    AreaLight light {
                        .geometry_id = i,
                        .power = total_area * emittance * c_PI,
                        .area = total_area,
                    };
                    scene_light_power += light.power;
                    m_materials[geom.material_id].mat.light.light_id = m_area_lights.size();
                    m_area_lights.push_back(light);
                    break;
                }
                case GeometryType::Sphere: {
                    AreaLight light {
                        .geometry_id = i,
                        .power = c_PI * emittance * c_PI,
                        .area = c_PI,
                    };
                    scene_light_power += light.power;
                    m_materials[geom.material_id].mat.light.light_id = m_area_lights.size();
                    m_area_lights.push_back(light);
                    break;
                }
                case GeometryType::Cube: {
                    AreaLight light {
                        .geometry_id = i,
                        .power = 6.0f * emittance * c_PI,
                        .area = 6.0f,
                    };
                    scene_light_power += light.power;
                    m_materials[geom.material_id].mat.light.light_id = m_area_lights.size();
                    m_area_lights.push_back(light);
                    break;
                }
            }
        }

        if (geom.type != GeometryType::TriangleMesh)
            continue;
        
        BVH::BVH bvh(geom, m_vertices, m_indices);
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

    m_view.total_light_power = scene_light_power;

    std::vector<float> powers(m_area_lights.size());
    for (uint32_t i = 0; i < m_area_lights.size(); ++i) {
        powers[i] = m_area_lights[i].power;
    }
    DiscreteSampler1D light_sampler(powers);
    m_view.light_sampler = light_sampler.View();
}
