// #pragma once

// #include <glm/glm.hpp>

// struct Sphere {

// };

// struct MeshVertex {
//     glm::vec3 position;
//     glm::vec3 normal;
//     glm::vec2 uv0;
//     glm::vec4 color;
// };

// struct TriangleMesh {
//     uint32_t first_index = 0;
//     uint32_t index_count = 0;
//     uint32_t vertex_count = 0;

//     int first_bvh_node = -1;
//     int bvh_node_count = -1;

//     int first_tri_index = -1;
//     int tri_index_count = -1;
// };

// enum class ShapeType {
//     Sphere,
//     TriangleMesh
// };

// struct GeometryInstance {
//     ShapeType type;
//     int shape_id = -1;

//     int material_id = -1;

//     glm::mat4 obj_to_world {1.0f};
//     glm::mat4 world_to_obj {1.0f};
//     glm::mat4 frame_to_world {1.0f};
// };
