#pragma once

#include <sstream>
#include <stdexcept>

class Exception : public std::runtime_error {
public:
    Exception(const std::string &err, const std::string &filename, int line)
        : std::runtime_error(err)
    {
        std::ostringstream oss;
        oss << "[" << filename << ":" << line << "]: " << err;
        m_message = oss.str();
    }

    virtual const char *what() const noexcept override {
        return m_message.c_str();
    }
private:
    std::string m_message = "";
};

#define PT_ERROR(arg) throw Exception(arg, __FILE__, __LINE__);

#define PT_QASSERT(expr) \
    if (!(expr)) PT_ERROR("Assertion failed:\n\t " + std::string(#expr));

#define PT_ASSERT(expr, arg) \
    if (!(expr)) PT_ERROR("Assertion failed:\n\t " + std::string(#expr) + " -- " + std::string(arg));
