#include "scene.h"
#include "kernel/integrators/integrator.h"
#include "math/constants.h"
#include "math/geometry.h"
#include "mesh/wavefront_obj.h"
#include "utils/cuda_utils.h"
#include "utils/utils.h"
#include "utils/exception.h"
#include "math/transform.h"
#include <fstream>
#include <iostream>
#include <glm/gtc/matrix_inverse.hpp>
#include <memory>
#include <string>
#include <unordered_map>

Scene::Scene(const std::string &filename) {
    LoadScene(filename);
    std::memset(&m_device_scene, 0, sizeof(DeviceScene));
}

Scene::~Scene() {
    m_texture_pool->FreeDevice(m_device_scene);

    {
        uint32_t n = m_device_scene.triangle_sampler_count;
        std::vector<DeviceDiscreteSampler1D> triangle_samplers(n);
        cudaMemcpy(triangle_samplers.data(), m_device_scene.triangle_samplers, n * sizeof(DeviceDiscreteSampler1D), cudaMemcpyDeviceToHost);

        for (uint32_t i = 0; i < n; ++i) {
            m_triangle_samplers[i].FreeDevice(triangle_samplers[i]);
        }
        cudaFree(m_device_scene.triangle_samplers);
    }

    DeviceDiscreteSampler1D light_sampler;
    cudaMemcpy(&light_sampler, m_device_scene.light_sampler, sizeof(DeviceDiscreteSampler1D), cudaMemcpyDeviceToHost);
    m_light_sampler.FreeDevice(light_sampler);

    cudaFree(m_device_scene.positions);
    cudaFree(m_device_scene.normals);
    cudaFree(m_device_scene.uvs);
    cudaFree(m_device_scene.indices);
    cudaFree(m_device_scene.geometries);

    cudaFree(m_device_scene.materials);
    cudaFree(m_device_scene.lights);

    m_bvh->FreeDevice(m_device_scene);
}

bool Scene::UpdateDevice() {
    bool old_dirty = m_dirty;
    if (m_dirty) {
        // -- vertex data / geometries --------
        m_device_scene.vertex_count = CopyToDevice(m_device_scene.positions, m_positions);
        CopyToDevice(m_device_scene.normals, m_normals);
        CopyToDevice(m_device_scene.uvs, m_texcoords);

        m_device_scene.index_count = CopyToDevice(m_device_scene.indices, m_indices);

        m_device_scene.geometry_count = CopyToDevice(m_device_scene.geometries, m_geometries);
        m_device_scene.max_depth = m_max_depth;

        // Update discrete samplers
        m_device_scene.triangle_sampler_count = m_triangle_samplers.size();
        if (m_device_scene.triangle_sampler_count > 0) {
            std::vector<DeviceDiscreteSampler1D> triangle_samplers(m_triangle_samplers.size());
            for (int i = 0; i < m_triangle_samplers.size(); ++i) {
                m_triangle_samplers[i].UpdateDevice(triangle_samplers[i]);
            }

            CopyToDevice(m_device_scene.triangle_samplers, triangle_samplers);
        }
        
        // -- materials --------
        m_device_scene.material_count = CopyToDevice(m_device_scene.materials, m_materials);
        
        // -- lights --------
        m_device_scene.light_count = CopyToDevice(m_device_scene.lights, m_lights);
        m_device_scene.total_light_power = m_total_light_power;

        DeviceDiscreteSampler1D light_sampler;
        m_light_sampler.UpdateDevice(light_sampler);
        CUDA_CHECK(cudaMalloc((void **) &m_device_scene.light_sampler, sizeof(DeviceDiscreteSampler1D)));
        CUDA_CHECK(cudaMemcpy(m_device_scene.light_sampler, &light_sampler, sizeof(DeviceDiscreteSampler1D), cudaMemcpyHostToDevice));

        m_dirty = false;
    }
    
    m_texture_pool->UpdateDevice(m_device_scene);
    m_bvh->UpdateDevice(m_device_scene);
    m_camera->UpdateDevice(m_device_scene);

    return old_dirty;
}

IntegratorType _GetIntegratorType(const std::string &str) {
    std::string integrator = Utils::ToLowercase(str);
    if (integrator == "path") {
        return IntegratorType::Path;
    }

    std::cout << std::format("\tInvalid integrator type '{}'. Defaulting to 'path'.", integrator) << std::endl;
    return IntegratorType::Path;
}

CameraType _GetCameraType(const std::string &str) {
    std::string camera = Utils::ToLowercase(str);
    if (camera == "pinhole") {
        return CameraType::Pinhole;
    } else if (camera == "thinlens") {
        return CameraType::ThinLens;
    }

    std::cout << std::format("\tInvalid camera type '{}'. Defaulting to 'pinhole'.", camera) << std::endl;
    return CameraType::Pinhole;
}

SamplerType _GetSamplerType(const std::string &str) {
    std::string sampler = Utils::ToLowercase(str);
    if (sampler == "independent") {
        return SamplerType::Independent;
    } else if (sampler == "sobol") {
        return SamplerType::Sobol;
    }

    std::cout << std::format("\tInvalid sampler type '{}'. Defaulting to 'independent'.", sampler) << std::endl;
    return SamplerType::Independent;
}

ReconstructionFilterType _GetFilterType(const std::string &str) {
    std::string filter = Utils::ToLowercase(str);
    if (filter == "box") {
        return ReconstructionFilterType::Box;
    } else if (filter == "gaussian") {
        return ReconstructionFilterType::Gaussian;
    }

    std::cout << std::format("\tInvalid filter type '{}'. Defaulting to 'box'.", filter) << std::endl;
    return ReconstructionFilterType::Box;
}

ColorSpace _GetColorSpaceType(const std::string &str) {
    std::string colorspace = Utils::ToLowercase(str);
    if (colorspace == "rgb") {
        return ColorSpace::RGB;
    } else if (colorspace == "xyz") {
        return ColorSpace::XYZ;
    }

    std::cout << std::format("\tInvalid color space '{}'. Defaulting to 'rgb'.", colorspace) << std::endl;
    return ColorSpace::RGB;
}

void Scene::ParseIntegrator(const json &data) {
    const IntegratorType default_integrator = IntegratorType::Path;
    const int default_depth = 4;
    const int default_iterations = 1000;

    if (data.contains("integrator")) {
        const auto &integrator = data["integrator"];
        m_integrator = _GetIntegratorType(integrator.value("TYPE", "path"));
        m_max_depth = integrator.value("DEPTH", default_depth);
        m_max_iterations = integrator.value("ITERATIONS", default_iterations);
    } else {
        std::cout << "\tNo integrator chosen. Choosing a default." << std::endl;

        m_integrator = default_integrator;
        m_max_depth = default_depth;
        m_max_iterations = default_iterations;
    }
}

void Scene::ParseCamera(const json &data) {
    const float default_fovy = 45.0f;
    const glm::vec3 default_up = glm::vec3(0.0f, 1.0f, 0.0f);
    const glm::vec3 default_eye = glm::vec3(0.0f);
    const glm::vec3 default_look = glm::vec3(0.0f, 0.0f, -1.0f);

    const float default_near = 0.1f;
    const float default_far = 100.0f;

    const SamplerType default_sampler = SamplerType::Independent;
    const ReconstructionFilterType default_filter = ReconstructionFilterType::Box;
    const ColorSpace film_format = ColorSpace::RGB;

    if (!data.contains("camera")) {
        std::cerr << "Failed to parse scene: Missing camera!" << std::endl;
        return;
    }

    m_camera = std::make_unique<Camera>();

    const auto &camera = data["camera"];
    m_camera->m_type = _GetCameraType(camera.value("TYPE", "pinhole"));
    m_camera->m_default_params.fovy = camera.value("FOVY", default_fovy);
    m_camera->m_default_params.up = Utils::GetOrDefault<glm::vec3>(camera, "UP", default_up);
    m_camera->m_default_params.position = Utils::GetOrDefault<glm::vec3>(camera, "EYE", default_eye);
    m_camera->m_default_params.look_at = Utils::GetOrDefault<glm::vec3>(camera, "LOOKAT", default_look);
    m_camera->m_near = Utils::GetOrDefault<float>(camera, "NEAR", default_near);
    m_camera->m_far = Utils::GetOrDefault<float>(camera, "FAR", default_far);
    

    if (camera.contains("sampler")) {
        const auto &sampler = camera["sampler"];
        m_sampler = _GetSamplerType(sampler.value("TYPE", "independent"));
    } else {
        m_sampler = default_sampler;
    }

    if (camera.contains("reconstruction_filter")) {
        const auto &filter = camera["reconstruction_filter"];
        m_camera->m_film.m_reconstruction_filter = _GetFilterType(filter.value("TYPE", "box"));
    } else {
        m_camera->m_film.m_reconstruction_filter = default_filter;
    }

    if (!camera.contains("film")) {
        std::cerr << "Failed to parse scene: Missing camera film!" << std::endl;
        return;
    }

    const auto &film = camera["film"];
    const int width = film["WIDTH"];
    const int height = film["HEIGHT"];

    m_camera->m_film.m_resolution = { width, height };
    m_camera->m_film.m_color_space = _GetColorSpaceType(film.value("COLOR_SPACE", "rgb"));

    m_camera->ResetDefaults();

}

void Scene::InitDefaults() {
    {
        Texture<float> texture;
        texture.type = TextureType::Constant;
        texture.constant.value = 0.98f;
        
        m_texture_pool->textures1.push_back(texture);
    }

    {
        Texture<glm::vec3> texture;
        texture.type = TextureType::Constant;
        texture.constant.value = glm::vec3(0.98f);
        
        m_texture_pool->textures3.push_back(texture);
    }

    {
        // default material uses default texture
        Material default_material;
        default_material.type = Material::Type::Lambertian;
        default_material.lambertian.albedo_texture = 0;

        m_materials.push_back(default_material);
    }
}

bool Scene::ParseMaterial(const json &material) {
    if (!material.contains("TYPE")) {
        std::cerr << "Failed to parse scene: Material must contain a type!";
        return 1;
    }

    Material mat;

    // -- Emitting material --------
    if (Utils::ToLowercase(material["TYPE"]) == "emitting") {
        mat.type = Material::Type::Emissive;

        if (material.contains("RADIANCE")) {
            const auto &radiance = material["RADIANCE"];
            mat.emissive.color_texture = ParseTexture<glm::vec3>(radiance);
            mat.emissive.emittance = radiance.value("EMITTANCE", 1.0f);
        } else {
            mat.emissive.color_texture = 0;
            mat.emissive.emittance = 1.0f;
        }
    }

    // -- Diffuse material --------
    if (Utils::ToLowercase(material["TYPE"]) == "diffuse" || Utils::ToLowercase(material["TYPE"]) == "lambertian") {
        mat.type = Material::Type::Lambertian;
        mat.lambertian.albedo_texture = LoadTextureOrZero<glm::vec3>(material, "ALBEDO", 0, true);
    }

    // -- Mirror material --------
    if (Utils::ToLowercase(material["TYPE"]) == "mirror") {
        mat.type = Material::Type::Mirror;
        mat.mirror.albedo_texture = LoadTextureOrZero<glm::vec3>(material, "ALBEDO", 0, true);
    }

    // -- Dielectric material --------
    if (Utils::ToLowercase(material["TYPE"]) == "dielectric") {
        mat.type = Material::Type::Dielectric;
        
        if (!material.contains("IOR") && !material.contains("INDEX_OF_REFRACTION")) {
            std::cerr << "Warning: Dielectric materials must have assigned an index of refraction" << std::endl;
        } else {
            float ior;
            if (material.contains("INDEX_OF_REFRACTION")) {
                ior = material["INDEX_OF_REFRACTION"];
            } else {
                ior = material["IOR"];
            }

            mat.dielectric.albedo_texture = LoadTextureOrZero<glm::vec3>(material, "ALBEDO", 0, true);
            mat.dielectric.ior = ior;
        }
        
    }

    // -- Metallic roughness material --------
    if (Utils::ToLowercase(material["TYPE"]) == "metallic_roughness") {
        mat.type = Material::Type::MetallicRoughness;
        mat.metallic_roughness.albedo_texture = LoadTextureOrZero<glm::vec3>(material, "ALBEDO", 0, true);
        uint32_t roughness_texture = LoadTextureOrZero<float>(material, "ROUGHNESS");
        uint32_t metallic_texture = LoadTextureOrZero<float>(material, "METALLIC");

        if (metallic_texture == 0 && roughness_texture == 0 && material.contains("ORM")) {
            roughness_texture = LoadTextureOrZero<float>(material, "ORM", 1);
            metallic_texture = LoadTextureOrZero<float>(material, "ORM", 2);
        }

        mat.metallic_roughness.roughness_texture = roughness_texture;
        mat.metallic_roughness.metallic_texture = metallic_texture;
    }

    m_materials.push_back(mat);

    return 0;
}

std::unordered_map<std::string, int> Scene::ParseMaterials(const json &data) {
    std::unordered_map<std::string, int> material_ids {};

    if (data.contains("materials")) {
        for (const auto &[key, value] : data["materials"].items()) {
            if (material_ids.contains(key)) {
                std::cerr << std::format("Scene Error: Found multiple materials with name '{}'.", key) << std::endl;
                continue;
            }

            material_ids[key] = m_materials.size();
            bool err = ParseMaterial(value);
            if (err) {
                return {};
            }
        }
    }

    return material_ids;
}

std::vector<GeometryInstance> Scene::ParseObject(const json &parsed_object, const std::unordered_map<std::string, int> &material_ids) {
    if (!parsed_object.contains("TYPE")) {
        std::cout << "Scene Error: Every object in a scene must have a type!" << std::endl;
        return {};
    }

    // -- Wavefront OBJ format --------
    if (Utils::ToLowercase(parsed_object["TYPE"]) == "obj") {
        PT_ASSERT(parsed_object.contains("PATH"), "Scene Error: Every obj mesh must have a path to load from.");

        GeometryInstance instance;

        std::string raw_path = parsed_object["PATH"].get<std::string>();
        fs::path candidate(raw_path);
        fs::path file = candidate.is_absolute()
            ? candidate
            : (m_scene_dir / candidate);

        MeshSettings settings;
        settings.face_normals = parsed_object.value("FACE_NORMALS", false);
        settings.invert_normals = parsed_object.value("INVERT_NORMALS", false);

        Mesh mesh = LoadWavefrontObjMesh(file, settings);
        if (mesh.indices.size() == 0 || mesh.positions.size() == 0) {
            std::cerr << "Skipping object since it has zero vertices!" << std::endl;
            return {};
        }
        
        // Update indices and vertices
        {
            uint32_t base_vertex = static_cast<uint32_t>(m_positions.size());
            instance.triangle_mesh.first_index = m_indices.size();
            instance.triangle_mesh.index_count = mesh.indices.size();
            instance.triangle_mesh.vertex_count = mesh.positions.size();

            m_indices.reserve(m_indices.size() + mesh.indices.size());
            for (auto idx : mesh.indices) {
                m_indices.push_back(base_vertex + idx);
            }

            m_positions.insert(m_positions.end(), mesh.positions.begin(), mesh.positions.end());
            m_texcoords.insert(m_texcoords.end(), mesh.texcoords.begin(), mesh.texcoords.end());
            m_normals.insert(m_normals.end(), mesh.normals.begin(), mesh.normals.end());
        }

        // We do not use the MTL files with the object, so if users don't specify the material,
        // we use default
        if (!parsed_object.contains("MATERIAL") || !material_ids.contains(parsed_object["MATERIAL"])) {
            instance.material_id = 0;
        } else {
            instance.material_id = material_ids.at(parsed_object["MATERIAL"]);
        }

        glm::vec3 translate = Utils::GetOrDefault(parsed_object, "TRANS", glm::vec3(0.0f));
        glm::vec3 rotate = Utils::GetOrDefault(parsed_object, "ROTAT", glm::vec3(0.0f));
        glm::vec3 scale = Utils::GetOrDefault(parsed_object, "SCALE", glm::vec3(1.0f));

        instance.transform = Math::GetTransformMatrix(translate, rotate, scale);
        instance.inv_transform = glm::inverse(instance.transform);
        instance.inv_transpose = glm::inverseTranspose(instance.transform);

        instance.blas_index = m_current_blas_index++;

        // m_geometries.push_back(instance);
        return { instance };
    }

    // -- glTF format --------
    if (Utils::ToLowercase(parsed_object["TYPE"]) == "gltf") {
        PT_ASSERT(parsed_object.contains("PATH"), "Scene Error: Every gltf mesh must have a path to load from.");

        std::string raw_path = parsed_object["PATH"].get<std::string>();
        fs::path candidate(raw_path);
        fs::path file = candidate.is_absolute()
            ? candidate
            : (m_scene_dir / candidate);

        MeshSettings settings;
        settings.face_normals = parsed_object.value("FACE_NORMALS", false);
        settings.invert_normals = parsed_object.value("INVERT_NORMALS", false);

        GLTF::GLTFModel model(settings, *m_texture_pool);
        model.LoadGLTF(file.string());

        const Mesh &mesh = model.AggregateMesh();
        if (mesh.indices.size() == 0 || mesh.positions.size() == 0) {
            std::cerr << "Skipping glTF object since it has zero vertices!" << std::endl;
            return {};
        }

        uint32_t base_vertex = static_cast<uint32_t>(m_positions.size());
        uint32_t base_index = static_cast<uint32_t>(m_indices.size());

        // Update indices and vertices

        m_indices.reserve(m_indices.size() + mesh.indices.size());
        for (auto idx : mesh.indices) {
            m_indices.push_back(base_vertex + idx);
        }
        m_positions.insert(m_positions.end(), mesh.positions.begin(), mesh.positions.end());
        m_texcoords.insert(m_texcoords.end(), mesh.texcoords.begin(), mesh.texcoords.end());
        m_normals.insert(m_normals.end(), mesh.normals.begin(), mesh.normals.end());

        int base_material = m_materials.size();
        m_materials.insert(m_materials.end(), model.Materials().begin(), model.Materials().end());

        glm::vec3 translate = Utils::GetOrDefault(parsed_object, "TRANS", glm::vec3(0.0f));
        glm::vec3 rotate = Utils::GetOrDefault(parsed_object, "ROTAT", glm::vec3(0.0f));
        glm::vec3 scale = Utils::GetOrDefault(parsed_object, "SCALE", glm::vec3(1.0f));
        glm::mat4 instance_transform = Math::GetTransformMatrix(translate, rotate, scale);

        std::vector<GeometryInstance> geometries {};

        std::unordered_map<uint64_t, int> blas_indices;
        model.ForEachNode([&](const GLTF::SceneNode &node) {
            // Technically bad naming: `world_transform` here just means glTF model-space
            // as opposed to glTF primitive-space
            glm::mat4 world_transform = instance_transform * node.world_transform;

            if (!node.mesh) return;

            for (int i = 0; i < node.mesh->primitives.size(); ++i) {
                const GLTF::Primitive &primitive = node.mesh->primitives[i];

                GeometryInstance instance;
                instance.triangle_mesh.first_index = base_index + primitive.first_index;
                instance.triangle_mesh.index_count  = primitive.index_count;
                instance.triangle_mesh.vertex_count = primitive.vertex_count;

                // Scene-defined materials takes precedence
                if (!parsed_object.contains("MATERIAL") ||
                    !material_ids.contains(parsed_object["MATERIAL"])) {
                    instance.material_id = base_material + primitive.material_id;
                } else {
                    instance.material_id =
                        material_ids.at(parsed_object["MATERIAL"]);
                }

                instance.transform = world_transform;
                instance.inv_transform = glm::inverse(world_transform);
                instance.inv_transpose = glm::inverseTranspose(world_transform);

                uint64_t key = (uint64_t(node.mesh->mesh_id) << 32) | uint64_t(i);

                auto it = blas_indices.find(key);                               
                if (it == blas_indices.end()) {                                    
                    instance.blas_index = m_current_blas_index++;
                    blas_indices[key] = instance.blas_index;
                } else {
                    instance.blas_index = it->second;                                
                }

                geometries.push_back(instance);
            }
        });

        return geometries;
    }

    return {};
}

void Scene::ParseObjects(const json &data, const std::unordered_map<std::string, int> &material_ids) {
    if (!data.contains("objects")) {
        std::cout << "Scene Error: Every scene must contain an object!" << std::endl;
        return;
    }

    // Pass 1: Handle objects and generate base instances
    std::unordered_map<std::string, std::vector<GeometryInstance>> instances;
    for (const auto &parsed_object : data["objects"]) {
        
        // Insert all basic objects
        std::vector<GeometryInstance> geometry_instances = ParseObject(parsed_object, material_ids);
        m_geometries.insert(m_geometries.end(), geometry_instances.begin(), geometry_instances.end());

        // -- Handle instancing --------
        if (Utils::ToLowercase(parsed_object["TYPE"]) == "instance") {
            if (!(parsed_object.contains("NAME") && parsed_object.contains("INSTANCE")) && !parsed_object.contains("SOURCE")) {
                std::cerr << "Warning: Invalid instance object!" << std::endl;
                return;
            }

            // -- Instance base --------
            if (parsed_object.contains("NAME")) {
                std::string name = parsed_object["NAME"].get<std::string>();
                instances[name] = ParseObject(parsed_object["INSTANCE"], material_ids);
            }
        }
    }

    // Pass 2: Link all instances
    for (const auto &parsed_object : data["objects"]) {
        if (Utils::ToLowercase(parsed_object["TYPE"]) == "instance") {
            if (!(parsed_object.contains("NAME") && parsed_object.contains("INSTANCE")) && !parsed_object.contains("SOURCE")) {
                std::cerr << "Warning: Invalid instance object!" << std::endl;
                return;
            }

            // -- instance --------
            if (parsed_object.contains("SOURCE")) {
                std::string source_ref = parsed_object["SOURCE"].get<std::string>();

                glm::vec3 translate = Utils::GetOrDefault(parsed_object, "TRANS", glm::vec3(0.0f));
                glm::vec3 rotate = Utils::GetOrDefault(parsed_object, "ROTAT", glm::vec3(0.0f));
                glm::vec3 scale = Utils::GetOrDefault(parsed_object, "SCALE", glm::vec3(1.0f));

                glm::mat4 instance_transform = Math::GetTransformMatrix(translate, rotate, scale);

                for (const auto &instance : instances[source_ref]) {
                    GeometryInstance new_instance = instance;
                    new_instance.transform = instance_transform * new_instance.transform;
                    new_instance.inv_transform = glm::inverse(new_instance.transform);
                    new_instance.inv_transpose = glm::inverseTranspose(new_instance.transform);

                    m_geometries.push_back(new_instance);
                }
            }
        }
    }
}

void Scene::ComputeLightData() {
    uint32_t light_count = 0;
    for (int i = 0; i < m_geometries.size(); ++i) {
        if (m_materials[m_geometries[i].material_id].type == Material::Type::Emissive) {
            light_count++;
        }
    }

    std::vector<float> light_powers(light_count);
    m_lights.resize(light_count);
    m_triangle_samplers.resize(light_count);

    float total_light_power = 0.0f;
    uint32_t light_index = 0;
    for (int i = 0; i < m_geometries.size(); ++i) {
        if (m_materials[m_geometries[i].material_id].type != Material::Type::Emissive) {
            continue;
        }

        // Collect triangle areas.
        float total_area = 0.0f;
        std::vector<float> tri_areas;
        tri_areas.reserve(m_geometries[i].triangle_mesh.index_count / 3);

        // TODO: Might be able to save some work and memory for instanced
        // light sources if we re-use triangle samplers
        for (int j = 0; j < m_geometries[i].triangle_mesh.index_count; j += 3) {
            int i0 = m_indices[m_geometries[i].triangle_mesh.first_index + j + 0];
            int i1 = m_indices[m_geometries[i].triangle_mesh.first_index + j + 1];
            int i2 = m_indices[m_geometries[i].triangle_mesh.first_index + j + 2];

            glm::vec3 v0 = m_geometries[i].transform * glm::vec4(m_positions[i0], 1.0f);
            glm::vec3 v1 = m_geometries[i].transform * glm::vec4(m_positions[i1], 1.0f);
            glm::vec3 v2 = m_geometries[i].transform * glm::vec4(m_positions[i2], 1.0f);

            float area = Math::TriangleArea(v0, v1, v2);
            total_area += area;
            tri_areas.push_back(area);
        }

        // Update geometry and upload triangle sampler. This will be useful for sampling
        // a random triangle on the mesh with probability based on area
        m_geometries[i].total_area = total_area;
        std::cout << "TOTAL_AREA: " << total_area << std::endl;

        m_geometries[i].area_light_id = light_index;
        m_geometries[i].triangle_sampler_index = light_index;

        m_triangle_samplers[light_index].UpdateWeights(tri_areas);

        // Create the area light itself
        Material &material = m_materials[m_geometries[i].material_id];
        float power = total_area * material.emissive.emittance * Math::PI;

        AreaLight light {
            .power = power,
            .geometry_instance_index = i,
        };
        m_lights[light_index] = std::move(light);

        total_light_power += power;
        light_powers[light_index] = power;

        light_index++;
    }
    m_total_light_power = total_light_power;
    m_light_sampler.UpdateWeights(light_powers);
}

void Scene::LoadScene(const std::string &filename) {
    fs::path scene_path = filename;
    fs::path scene_dir = scene_path.parent_path();
    m_scene_dir = scene_dir;

    // -- Parse scene and load textures --------
    std::cout << "Parsing Scene..." << std::endl;

    std::ifstream file(scene_path);
    PT_ASSERT(file.is_open(), std::format("ERROR: Could not open file '{}'.", filename));

    json data = json::parse(file);
    std::string err = "";

    m_texture_pool = std::make_unique<TexturePool>();
    ParseIntegrator(data);
    ParseCamera(data);

    // Handles default textures and materials
    InitDefaults();

    std::unordered_map<std::string, int> material_ids = ParseMaterials(data);
    if (material_ids.empty())
        return;

    ParseObjects(data, material_ids);

    // -- Acceleration structures --------
    m_bvh = std::make_unique<TLAS>();
    m_bvh->Build(*this, m_geometries);

    // -- Lights --------
    ComputeLightData();

    m_dirty = true;
    m_texture_pool->dirty = true;
    m_bvh->Dirty() = true;
    m_camera->Dirty() = true;
}
