#include "image.h"
#include <filesystem>
#include "utils.h"
#include <format>
#include <iostream>
#include <stb_image_write.h>
#include <stb_image.h>
#include "exception.h"

void Load1DImageFromFile(Image1 &image, const fs::path &filename) {
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
        float *data = stbi_loadf(filename.string().c_str(), &width, &height, &channels, 1);
        if (data == nullptr) {
            PT_ERROR(std::format("Failure loading image '{}'.", filename.string()));
        }
        image.SetSize(width, height);

        for (uint32_t i = 0; i < width * height; ++i) {
            image[i] = data[i];
        }
        stbi_image_free(data);
    } else {
        PT_ERROR(std::format("Unsupported file format for file '{}'", filename.string()));
    }
}

void Load3DImageFromFile(Image3 &image, const fs::path &filename) {
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
        image.SetSize(width, height);

        int j = 0;
        for (uint32_t i = 0; i < width * height; ++i) {
            image[i].x = data[j++];
            image[i].y = data[j++];
            image[i].z = data[j++];
        }
        stbi_image_free(data);
    } else {
        PT_ERROR(std::format("Unsupported file format for file '{}'", filename.string()));
    }
}

template<typename T>
Image<T> LoadImageFromFile(const fs::path &filename) {
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

        Image<T> image(width, height);

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

void SavePNG(const Image3 &image, const std::string &filename) {
    if (image.Empty()) return;

    std::vector<uint8_t> output(3 * image.Width() * image.Height());
    for (uint32_t j = 0; j < image.Height(); ++j) {
        for (uint32_t i = 0; i < image.Width(); ++i) {
            int index = j * image.Width() + i;
            glm::vec3 pixel = glm::clamp(image[index], glm::vec3(0), glm::vec3(1)) * 255.0f;
            output[3 * index + 0] = static_cast<uint8_t>(pixel.r);
            output[3 * index + 1] = static_cast<uint8_t>(pixel.g);
            output[3 * index + 2] = static_cast<uint8_t>(pixel.b);
        }
    }
    std::string result = filename + ".png";

    stbi_write_png(result.c_str(), image.Width(), image.Height(), 3, output.data(), image.Width() * 3);
    std::cout << "Saved " << result << std::endl;
}

void SaveHDR(const Image3 &image, const std::string &filename) {
    if (image.Empty()) return;
    std::string result = filename + ".hdr";
    
    stbi_write_hdr(result.c_str(), image.Width(), image.Height(), 3, reinterpret_cast<const float *>(image.Data()));
    std::cout << "Saved " << result << std::endl;
}
