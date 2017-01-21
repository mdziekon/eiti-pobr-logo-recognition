#ifndef POBR_UTILS_PERFORMANCETIMER_HPP
#define POBR_UTILS_PERFORMANCETIMER_HPP

#include <chrono>

namespace pobr::utils
{
    class PerformanceTimer
    {
    protected:
        std::chrono::high_resolution_clock::time_point pointStart;
        std::chrono::high_resolution_clock::time_point pointStop;

    public:
        const void start();
        const void stop();

        const double getDurationNS() const; 
    };
}

#endif
