#pragma once

#include "json.hpp"
#include <iostream>
#include <optional>
#include <string>
#include <filesystem>

#include <glm/glm.hpp>

namespace fs = std::filesystem;

namespace Utils {

    template<typename T>
    inline T AlignDown(T value, T alignment) {
        return value & ~(alignment - 1);
    }

    template<typename T>
    inline int AlignUp(T value, T alignment) {
        return AlignDown(value + alignment - 1, alignment);
    }

    inline std::string ToLowercase(const std::string &s) {
        std::string out = s;
        std::transform(s.begin(), s.end(), out.begin(), ::tolower);
        return out;
    }

    inline std::string CurrentTimeString() {
        time_t now;
        time(&now);
        char buf[sizeof "0000-00-00_00-00-00z"];
        strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
        return std::string(buf);
    }

    static std::optional<fs::path> ResolvePath(
        const fs::path &candidate,
        const fs::path &base_dir)
    {
        if (fs::exists(candidate)) {
            return fs::absolute(candidate);
        }

        fs::path joined = base_dir / candidate;
        if (fs::exists(joined)) {
            return fs::absolute(joined);
        }

        return std::nullopt;
    }

    template<int N, typename T>
    glm::vec<N, T> ParseVector(const nlohmann::json &data, const glm::vec<N, T> &def) {
        if (!data.is_array() || data.size() != N) {
            std::cerr << std::format("Expected an array of size {} for vec{}.\n", N, N);
            return def;
        }

        for (int i = 0; i < N; ++i) {
            if (!data[i].is_number()) {
                std::cerr << "Expected numeric components in vector\n";
                return def;
            }
        }

        glm::vec<N, T> result;
        for (int i = 0; i < N; ++i) {
            result[i] = data[i].get<T>();
        }
        return result;
    }

    template<typename> struct is_glm_vec : std::false_type {};

    template<glm::length_t L, typename T>
    struct is_glm_vec<glm::vec<L, T>> : std::true_type {};

    template<typename T>
    inline constexpr bool is_glm_vec_v = is_glm_vec<T>::value;

    template<typename T>
    T GetOrDefault(const nlohmann::json &data, const std::string &key, const T &def) {
        auto it = data.find(key);
        if (it == data.end()) {
            return def;
        }

        if constexpr (is_glm_vec_v<T>) {
            return ParseVector(*it, def);
        } else {
            return it->get<T>();
        }
    }

    
    static double FileSize(const fs::path &p) {
        auto bytes = std::filesystem::file_size(p);
        return double(bytes) / (1024.0 * 1024.0);
    }

}
