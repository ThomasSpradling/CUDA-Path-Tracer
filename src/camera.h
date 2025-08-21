#pragma once

#include "film.h"
#include "kernel/device_scene.h"
#include "kernel/device_types.h"

class Scene;

class Camera {
    friend Scene;
public:
    inline const glm::ivec2 &Resolution() const { return m_film.m_resolution; }
    inline const glm::vec3 &LookAt() const { return m_look_at; }
    inline glm::vec3 &LookAt() { return m_look_at; }

    inline glm::vec3 &Position() { return m_position; }
    inline const glm::vec3 &Position() const { return m_position; }

    inline glm::vec3 &Front() { return m_front; }
    inline const glm::vec3 &Front() const { return m_front; }

    inline glm::vec3 &Up() { return m_up; }
    inline const glm::vec3 &Up() const { return m_up; }

    inline glm::vec3 &Right() { return m_right; }
    inline const glm::vec3 &Right() const { return m_right; }

    void SetResolution(const glm::ivec2 &res);

    void Update();
    void ResetDefaults();

    bool &Dirty() { return m_dirty; }
    void UpdateDevice(DeviceScene &scene);
private:
    bool m_dirty = false;
    CameraType m_type;

    struct Default {
        float fovy;
        glm::vec3 up;
        glm::vec3 look_at;
        glm::vec3 position;
    } m_default_params;

    glm::vec3 m_position;
    glm::vec3 m_look_at;

    glm::vec3 m_front;
    glm::vec3 m_up;
    glm::vec3 m_right;

    float m_near, m_far;

    float m_fovy;
    glm::vec2 m_pixel_length;

    Film m_film;
};
