#include "wavefront_obj.h"
#include "mesh.h"
#include <array>
#include <tiny_obj_loader.h>
#include "../utils/stopwatch.h"

Mesh LoadWavefrontObjMesh(const fs::path &path, const MeshSettings &settings) {
    StopWatch load_timer;
    load_timer.Start();
    
    Mesh mesh;

    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = path.parent_path().string();

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path.string(), reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjLoader Error: " << reader.Error() << std::endl;
        }
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjLoader Warning: " << reader.Warning();
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();

    mesh.indices.reserve(attrib.vertices.size() / 3);
    mesh.positions.reserve(attrib.vertices.size() / 3);
    mesh.normals.reserve(attrib.vertices.size() / 3);
    mesh.texcoords.reserve(attrib.texcoords.size() / 2);

    bool compute_normals = attrib.normals.empty() || settings.face_normals;
    
    for (size_t s = 0; s < shapes.size(); ++s) {
        size_t index_offset = 0;
        
        // loop over faces
        bool signaled_error = false;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f) {
            size_t vertex_count = shapes[s].mesh.num_face_vertices[f];
            if (vertex_count != 3 && !signaled_error) {
                std::cerr << "Obj Loading: We expected all faces to be triangles! This render might look incorrect..." << std::endl;
                signaled_error = true;
                
                vertex_count = std::clamp(static_cast<int>(vertex_count), 0, 3);
            }
            
            std::array<glm::vec3, 3> pos;
            std::array<glm::vec3, 3> normals;
            std::array<glm::vec2, 3> uvs {};

            for (size_t v = 0; v < vertex_count; ++v) {
                tinyobj::index_t index = shapes[s].mesh.indices[index_offset + v];

                tinyobj::real_t v0 = attrib.vertices[3*index.vertex_index + 0];
                tinyobj::real_t v1 = attrib.vertices[3*index.vertex_index + 1];
                tinyobj::real_t v2 = attrib.vertices[3*index.vertex_index + 2];

                pos[v] = glm::vec3(v0, v1, v2);

                if (!compute_normals) {
                    tinyobj::real_t n0 = attrib.normals[3*index.normal_index + 0];
                    tinyobj::real_t n1 = attrib.normals[3*index.normal_index + 1];
                    tinyobj::real_t n2 = attrib.normals[3*index.normal_index + 2];

                    normals[v] = glm::vec3(n0, n1, n2);
                }

                if (index.texcoord_index >= 0) {
                    tinyobj::real_t uv0 = attrib.texcoords[2*index.texcoord_index + 0];
                    tinyobj::real_t uv1 = attrib.texcoords[2*index.texcoord_index + 1];

                    uvs[v] = glm::vec2(uv0, uv1);
                }
            }

            glm::vec3 face_normal;
            if (compute_normals) {
                glm::vec3 e1 = glm::normalize(pos[1] - pos[0]);
                glm::vec3 e2 = glm::normalize(pos[2] - pos[0]);

                face_normal = glm::normalize(glm::cross(e1, e2));
            }

            for (size_t v = 0; v < vertex_count; ++v) {
                mesh.positions.push_back(pos[v]);
                mesh.indices.push_back(mesh.positions.size() - 1);

                glm::vec3 normal = compute_normals ? face_normal : glm::normalize(normals[v]);
                if (settings.invert_normals) {
                    normal = -normal;
                }

                mesh.normals.push_back(normal);
                mesh.texcoords.push_back(uvs[v]);
            }

            index_offset += vertex_count;
        }
    }

    load_timer.Finish(std::format("Loaded model '{}' [{:.2f} MB]", path.filename().string(), Utils::FileSize(path)));
    return mesh;
}
