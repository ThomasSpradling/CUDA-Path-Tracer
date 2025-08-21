#include "camera.h"
#include "utils/cuda_utils.h"
#include "utils/exception.h"
#include <iostream>

void Camera::Update() {
    m_front = glm::normalize(m_look_at - m_position);
    m_right = glm::normalize(glm::cross(m_front, m_up));
    m_up = glm::normalize(glm::cross(m_right, m_front));

    float theta = glm::radians(m_fovy);
    float half_height = std::tan(theta * 0.5f);
    float aspect = float(m_film.m_resolution.x) / float(m_film.m_resolution.y);
    float half_width = aspect * half_height;

    m_pixel_length.x = (2.0f * half_width) / float(m_film.m_resolution.x);
    m_pixel_length.y = (2.0f * half_height) / float(m_film.m_resolution.y);

    m_dirty = true;
}

void Camera::ResetDefaults() {
    m_look_at = m_default_params.look_at;
    m_fovy = m_default_params.fovy;
    m_position = m_default_params.position;
    m_up = m_default_params.up;

    Update();
}

void Camera::SetResolution(const glm::ivec2 &res) {
    m_film.m_resolution = res;
    Update();
}

void Camera::UpdateDevice(DeviceScene &scene) {
    if (!m_dirty) return;

    scene.camera.resolution = m_film.m_resolution;
    scene.camera.position = m_position;
    scene.camera.front = m_front;
    scene.camera.up = m_up;
    scene.camera.right = m_right;
    scene.camera.fovy = m_fovy;
    scene.camera.pixel_length = m_pixel_length;
    scene.camera.focal_length = 1.0f;
    scene.camera.near = m_near;
    scene.camera.far = m_far;

    m_dirty = false;
}
