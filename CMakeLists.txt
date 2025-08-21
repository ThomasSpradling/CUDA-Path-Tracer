cmake_minimum_required(VERSION 3.24)
project(cuda_path_tracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Dev
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Dependencies

## Vulkan
find_package(Vulkan REQUIRED)

## CUDA
find_package(CUDAToolkit REQUIRED)

## GLFW
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/glfw)

## TinyGLTF
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL_VENDOR OFF CACHE BOOL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/tinygltf)

## TinyObjLoader
add_library(tinyobjloader
    third_party/tinyobjloader/tiny_obj_loader.cpp
)
target_include_directories(tinyobjloader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyobjloader)

## STB
add_library(stb
    third_party/stb/stb_image_write.cpp
    third_party/stb/stb_image.cpp
)
target_include_directories(stb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)

## GLM
set(GLM_BUILD_LIBRARY OFF CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/glm)

## Volk
add_subdirectory(third_party/volk)
if (WIN32)
    target_compile_definitions(volk PRIVATE VK_USE_PLATFORM_WIN32_KHR)
endif()

## ImGui
set(IMGUI_SRCS
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_demo.cpp
    third_party/imgui/backends/imgui_impl_glfw.cpp
    third_party/imgui/backends/imgui_impl_vulkan.cpp
)
add_library(imgui STATIC
    ${IMGUI_SRCS}
)
target_include_directories(imgui PUBLIC
    third_party/imgui
)
target_compile_definitions(imgui PRIVATE IMGUI_IMPL_VULKAN_USE_VOLK)
target_link_libraries(imgui volk glfw)

## Vulkan Memory Allocator
add_subdirectory(third_party/VulkanMemoryAllocator)

## json
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/json)

# Source

# -- CUDA Sources ----

file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS src/kernel/*.cu)

add_library(cuda_kernels STATIC ${CUDA_SOURCES})
set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_options(cuda_kernels PRIVATE
    --expt-relaxed-constexpr
    $<$<CONFIG:Debug>:-G;-src-in-ptx>
    $<$<CONFIG:RelWithDebInfo>:-lineinfo;-src-in-ptx>
)
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(cuda_kernels PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(cuda_kernels PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(cuda_kernels PROPERTIES CUDA_ARCHITECTURES native)
endif()

target_link_libraries(cuda_kernels PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    glm::glm
)

# -- C++ Sources ----

file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.h)

add_executable(${PROJECT_NAME} ${CORE_SOURCES})
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:
        $<$<CXX_COMPILER_ID:GNU,Clang>:-g>
        $<$<CXX_COMPILER_ID:MSVC>: /Zi>
    >
    $<$<CONFIG:RelWithDebInfo>:
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -gline-tables-only -fdebug-types-section>
        $<$<CXX_COMPILER_ID:MSVC>:/O2 /Zi /Oy->
    >
)

target_link_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:RelWithDebInfo>:
        $<$<CXX_COMPILER_ID:MSVC>:/DEBUG /OPT:REF /OPT:ICF>
    >
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pdb"
  PDB_NAME "${PROJECT_NAME}"
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pdb"
  COMPILE_PDB_NAME "${PROJECT_NAME}_obj"
)

target_link_libraries(${PROJECT_NAME}
    cuda_kernels
    stb
    glfw
    tinygltf
    tinyobjloader
    glm::glm
    volk::volk
    VulkanMemoryAllocator
    imgui
    nlohmann_json
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})
