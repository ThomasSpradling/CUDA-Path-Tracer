#pragma once

#include <string>
#include "utils.h"
#include <glm/glm.hpp>

using namespace std;

class Image {
public:
    Image();
    Image(uint32_t width, uint32_t height);
    ~Image();

    void SetSize(uint32_t width, uint32_t height);

    glm::vec3 *Data() { return m_data.data(); }

    void Clear(const glm::vec3 &value);

    glm::vec3 &operator()(uint32_t x);
    const glm::vec3 &operator()(uint32_t x) const;

    glm::vec3 &operator()(uint32_t x, uint32_t y);
    const glm::vec3 &operator()(uint32_t x, uint32_t y) const;

    static Image LoadFromFile(const fs::path &filename);

    void SavePNG(const std::string &filename) const;
    void SaveHDR(const std::string &filename) const;
private:
    uint32_t m_width;
    uint32_t m_height;

    std::vector<glm::vec3> m_data;
};
