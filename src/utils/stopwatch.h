#pragma once

#include <chrono>
#include <iostream>

class StopWatch {
public:
    using clock_t = std::chrono::high_resolution_clock;
    using ms_t = std::chrono::duration<double, std::milli>;

    StopWatch() : m_running(false) {}

    void Start() {
        m_start = clock_t::now();
        m_running = true;
    }

    void Finish(const std::string &label) {
        if (!m_running) {
            std::cerr << "[StopWatch] Finish() called without Start()\n";
            return;
        }
        auto end = clock_t::now();
        double elapsed = std::chrono::duration_cast<ms_t>(end - m_start).count();
        std::cout << label << " (" << static_cast<long long>(elapsed) << " ms)\n";
        m_running = false;
    }

private:
    typename clock_t::time_point m_start;
    bool m_running;
};
