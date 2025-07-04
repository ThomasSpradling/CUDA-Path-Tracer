#include "scene.h"
#include "exception.h"
#include <cstdint>
#include <fstream>
#include <iostream>
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
    std::string ext = filename.substr(filename.find_last_of('.'));
    PT_ASSERT(ext == ".json", std::format("Couldn't read from file '{}'.", filename));

    LoadFromJSON(filename);
}

void Scene::LoadFromJSON(const std::string &filename) {
    std::ifstream file(filename);
    PT_ASSERT(file.is_open(),
        std::format("ERROR: Could not open file '{}'.", filename));

    json data = json::parse(file);
    const auto &material_data = data["Materials"];
    std::unordered_map<std::string, uint32_t> material_ids;
    for (const auto &[key, value] : material_data.items()) {
        Material material {};
        if (value["TYPE"] == "Diffuse") {
            const auto &color = value["RGB"];
            material.color = glm::vec3(color[0], color[1], color[2]);
        } else if (value["TYPE"] == "Emitting") {
            const auto &color = value["RGB"];
            material.color = glm::vec3(color[0], color[1], color[2]);
            material.emittance = value["EMITTANCE"];
        } else if (value["TYPE"] == "Specular") {
            const auto &color = value["RGB"];
            material.color = glm::vec3(color[0], color[1], color[2]);
        }

        material_ids[key] = m_materials.size();
        m_materials.emplace_back(material);
    }

    const auto &object_data = data["Objects"];
    for (const auto &parsed_object : object_data) {
        const auto &type = parsed_object["TYPE"];
        Geometry geometry;
        if (type == "cube") {
            geometry.type = GeometryType::Cube;
        } else if (type == "sphere") {
            geometry.type = GeometryType::Sphere;
        } else {
            PT_ERROR("Invalid geometry type!");
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

        m_geometry.push_back(geometry);
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
}
