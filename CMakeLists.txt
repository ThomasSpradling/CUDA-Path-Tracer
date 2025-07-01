cmake_minimum_required(VERSION 3.15)
project(cuda_path_tracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Dev
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

## XML
option(PUGIXML_INSTALL OFF)
add_subdirectory(third_party/pugixml)

## TinyObj
add_library(tinyobjloader third_party/tinyobjloader/tiny_obj_loader.cpp)
target_include_directories(tinyobjloader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyobjloader)

## TinyPly
add_library(tinyply STATIC third_party/tinyply/tinyply.cpp)
target_include_directories(tinyply PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyply)

## STB
add_library(stb
    third_party/stb/stb_image_write.cpp
    third_party/stb/stb_image.cpp
)
target_include_directories(stb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)

## GLM
set(GLM_BUILD_LIBRARY OFF CACHE BOOL "" FORCE)
set(GLM_TEST_ENABLE OFF CACHE BOOL "" FORCE)
set(GLM_ENABLE_CXX_20 ON CACHE BOOL "" FORCE)
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

# Source

file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    src/*.cpp src/*.cu src/*.h)

add_executable(${CMAKE_PROJECT_NAME} ${CORE_SOURCES})

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
endif()

target_link_libraries(${CMAKE_PROJECT_NAME}
    CUDA::cudart
    CUDA::cuda_driver
    stb
    glfw
    glm::glm
    tinyply
    tinyobjloader
    volk::volk
    VulkanMemoryAllocator
    imgui
)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")


set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

