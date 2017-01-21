#include "PerformanceTimer.hpp"

using PerformanceTimer = pobr::utils::PerformanceTimer;

const void
PerformanceTimer::start()
{
    this->pointStart = std::chrono::high_resolution_clock::now();
}

const void
PerformanceTimer::stop()
{
    this->pointStop = std::chrono::high_resolution_clock::now();
}

const double
PerformanceTimer::getDurationNS()
const
{
    std::chrono::duration<double> diff = this->pointStop - this->pointStart;

    return std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();
} 
