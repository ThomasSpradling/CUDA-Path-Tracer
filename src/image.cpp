#include "image.h"
#include <filesystem>
#include <format>
#include <iostream>
#include <stb_image_write.h>
#include <stb_image.h>
#include "utils/stopwatch.h"
#include "utils/exception.h"
#include "utils/utils.h"

void Load1DImageFromFile(Image1 &image, const fs::path &filename, int channel) {
    StopWatch load_timer;
    load_timer.Start();

    if (!fs::exists(filename)) {
        std::cerr << "Could not open: " << filename.string() << std::endl;
        return;
    }

    std::string extension = Utils::ToLowercase(filename.extension().string());

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
        unsigned char *data = stbi_load(filename.string().c_str(), &width, &height, &channels, 0);
        if (data == nullptr) {
            std::cerr << "Failure loading image '{}'" << std::endl;
            return;
        }

        if (channel < 0 || channel >= channels) {
            std::cerr << std::format("Warning: requested channel {} but file has {} channels; using 0\n", channel, channels);
            channel = 0;
        }

        image.SetSize(width, height);

        const float inv_255 = 1.0f / 255.0f;
        for (uint32_t i = 0; i < width * height; ++i) {
            unsigned char v = data[i * channels + channel];
            image[i] = v * inv_255;
        }
        stbi_image_free(data);
    } else {
        PT_ERROR(std::format("Unsupported file format for file '{}'", filename.string()));
    }

    load_timer.Finish(std::format("\tLoaded texture '{}' [{:.2f} MB].", filename.filename().string(), Utils::FileSize(filename)));
}

void Load3DImageFromFile(Image3 &image, const fs::path &filename) {
    StopWatch load_timer;
    load_timer.Start();

    if (!fs::exists(filename)) {
        std::cerr << "Could not open: " << filename.string() << std::endl;
        return;
    }

    std::string extension = Utils::ToLowercase(filename.extension().string());

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
        unsigned char *data = stbi_load(filename.string().c_str(), &width, &height, &channels, 3);
        if (data == nullptr) {
            std::cerr << "Failure loading image '{}'" << std::endl;
            return;
        }
        image.SetSize(width, height);

        int j = 0;
        float inv_255 = 1.0f / 255.0f;
        for (uint32_t i = 0; i < width * height; ++i) {
            unsigned char r = data[j++];
            unsigned char g = data[j++];
            unsigned char b = data[j++];

            image[i].x = r * inv_255;
            image[i].y = g * inv_255;
            image[i].z = b * inv_255;
        }
        stbi_image_free(data);
    } else {
        PT_ERROR(std::format("Unsupported file format for file '{}'", filename.string()));
    }

    load_timer.Finish(std::format("\tLoaded texture '{}' [{:.2f} MB].", filename.filename().string(), Utils::FileSize(filename)));
}

void SavePNG(const Image3 &image, const std::string &filename) {
    if (image.Empty()) return;

    StopWatch save_timer;
    save_timer.Start();

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

    save_timer.Finish(std::format("Saved {} [{:.2f} MB].", filename, Utils::FileSize(result)));
}

void SaveHDR(const Image3 &image, const std::string &filename) {
    if (image.Empty()) return;

    StopWatch save_timer;
    save_timer.Start();

    std::string result = filename + ".hdr";
    
    stbi_write_hdr(result.c_str(), image.Width(), image.Height(), 3, reinterpret_cast<const float *>(image.Data()));

    save_timer.Finish(std::format("Saved {} [{:.2f} MB].", filename, Utils::FileSize(result)));
}
