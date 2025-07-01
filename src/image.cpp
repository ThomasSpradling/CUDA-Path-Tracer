#include "image.h"
#include <filesystem>
#include "glm/common.hpp"
#include "utils.h"
#include <format>
#include <iostream>
#include <stb_image_write.h>
#include <stb_image.h>
#include "exception.h"

Image::Image(uint32_t width, uint32_t height)
    : m_width(width)
    , m_height(height)
    , m_data(m_width * m_height, glm::vec3(0.0f))
{}

Image::~Image() {}

glm::vec3 &Image::operator()(uint32_t x) {
    return m_data[x];
}

const glm::vec3 &Image::operator()(uint32_t x) const {
    return m_data[x];
}

glm::vec3 &Image::operator()(uint32_t x, uint32_t y) {
    return m_data[y * m_width + x];
}

const glm::vec3 &Image::operator()(uint32_t x, uint32_t y) const {
    return m_data[y * m_width + x];
}

Image Image::LoadFromFile(const fs::path &filename) {
    std::string extension = ToLowercase(filename.extension().string());

    if (extension == ".jpg" ||
        extension == ".png" ||
        extension == ".tga" ||
        extension == ".bmp" ||
        extension == ".psd" ||
        extension == ".gif" ||
        extension == ".hdr" ||
        extension == ".pic"
    ) {
        int width, height, channels;
        float *data = stbi_loadf(filename.string().c_str(), &width, &height, &channels, 3);
        if (data == nullptr) {
            PT_ERROR(std::format("Failure loading image '{}'.", filename.string()));
        }

        Image image(width, height);

        int j = 0;
        for (uint32_t i = 0; i < width * height; ++i) {
            image(i).x = data[j++];
            image(i).y = data[j++];
            image(i).z = data[j++];
        }
        stbi_image_free(data);
        return image;
    } else {
        PT_ERROR(std::format("Unsupported file format for file '{}'", filename.string()));
    }
}

void Image::SavePNG(const std::string &filename) const {
    if (m_data.empty()) return;

    std::vector<uint8_t> output(3 * m_width * m_height);
    for (uint32_t j = 0; j < m_height; ++j) {
        for (uint32_t i = 0; i < m_width; ++i) {
            int idx = j * m_width + i;
            glm::vec3 pixel = glm::clamp(m_data[i], glm::vec3(), glm::vec3(1)) * 255.0f;
            output[3 * idx + 0] = pixel.x;
            output[3 * idx + 1] = pixel.y;
            output[3 * idx + 2] = pixel.z;
        }
    }

    stbi_write_png(filename.c_str(), m_width, m_height, 3, output.data(), m_width * 3);
    std::cout << "Saved" << filename << std::endl;
}

void Image::SaveHDR(const std::string &filename) const {
    if (m_data.empty()) return;

    stbi_write_hdr(filename.c_str(), m_width, m_height, 3, reinterpret_cast<const float *>(m_data.data()));
    std::cout << "Saved" << filename << std::endl;
}
