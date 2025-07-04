#include "image.h"
#include <filesystem>
#include "utils.h"
#include <format>
#include <iostream>
#include <stb_image_write.h>
#include <stb_image.h>
#include "exception.h"

Image::Image() : Image(0, 0) {}

Image::Image(uint32_t width, uint32_t height)
    : m_width(width)
    , m_height(height)
    , m_data(m_width * m_height, glm::vec3(0.0f))
{}

Image::~Image() {}

void Image::SetSize(uint32_t width, uint32_t height) {
    m_width = width;
    m_height = height;
    m_data.resize(m_width * m_height);
    Clear(glm::vec3(0.0f));
}

void Image::Clear(const glm::vec3 &value) {
    std::fill(m_data.begin(), m_data.end(), value);
}

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
            int index = j * m_width + i;
            glm::vec3 pixel = glm::clamp(m_data[index], glm::vec3(0), glm::vec3(1)) * 255.0f;
            output[3 * index + 0] = static_cast<uint8_t>(pixel.r);
            output[3 * index + 1] = static_cast<uint8_t>(pixel.g);
            output[3 * index + 2] = static_cast<uint8_t>(pixel.b);
        }
    }
    std::string result = filename + ".png";

    stbi_write_png(result.c_str(), m_width, m_height, 3, output.data(), m_width * 3);
    std::cout << "Saved " << result << std::endl;
}

void Image::SaveHDR(const std::string &filename) const {
    if (m_data.empty()) return;
    std::string result = filename + ".hdr";
    
    stbi_write_hdr(result.c_str(), m_width, m_height, 3, reinterpret_cast<const float *>(m_data.data()));
    std::cout << "Saved " << result << std::endl;
}
