#pragma once

#include <string>
#include "utils.h"
#include <glm/glm.hpp>

using namespace std;

template<typename T>
class Image {
public:
    Image() : Image(0, 0) {}
    Image(uint32_t width, uint32_t height)
        : m_width(width)
        , m_height(height)
        , m_data(m_width * m_height, T(0.0f))
    {}
    ~Image() = default;

    uint32_t Width() const { return m_width; }
    uint32_t Height() const { return m_height; }

    bool Empty() const { return m_width == 0 || m_height == 0 || m_data.empty(); }

    void SetSize(uint32_t width, uint32_t height) {
        m_width = width;
        m_height = height;
        m_data.resize(m_width * m_height);
        Clear(T(0.0f));
    }

    T *Data() { return m_data.data(); }
    const T *Data() const { return m_data.data(); }

    void Clear(const T &value) {
        std::fill(m_data.begin(), m_data.end(), value);
    }

    T &operator[](uint32_t x) { return m_data[x]; }
    const T &operator[](uint32_t x) const { return m_data[x]; }

    T &operator()(uint32_t x, uint32_t y) { return m_data[y * m_width + x]; }
    const T &operator()(uint32_t x, uint32_t y) const { return m_data[y * m_width + x]; }

    static Image LoadFromFile(const fs::path &filename);

    void SavePNG(const std::string &filename) const;
    void SaveHDR(const std::string &filename) const;
private:
    uint32_t m_width;
    uint32_t m_height;

    std::vector<T> m_data;
};

using Image1 = Image<float>;
using Image3 = Image<glm::vec3>;

void Load1DImageFromFile(Image1 &image, const fs::path &filename);
void Load3DImageFromFile(Image3 &image, const fs::path &filename);

void SavePNG(const Image3 &image, const std::string &filename);
void SaveHDR(const Image3 &image, const std::string &filename);
