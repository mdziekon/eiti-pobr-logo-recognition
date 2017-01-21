#ifndef POBR_IMGPROCESSING_STRUCTS_HPP
#define POBR_IMGPROCESSING_STRUCTS_HPP

#include <cstdint>

namespace pobr::imgProcessing::structs
{
    struct Segment
    {
    public:
        uint64_t xMin = 0;
        uint64_t xMax = 0;
        uint64_t yMin = 0;
        uint64_t yMax = 0;

        const void updateBoundaries(const uint64_t& x, const uint64_t& y)
        {
            if (x < this->xMin) {
                this->xMin = x;
            } else if (x > this->xMax) {
                this->xMax = x;
            }

            if (y < this->yMin) {
                this->yMin = y;
            } else if (y > this->yMax) {
                this->yMax = y;
            }
        }

        const void merge(const Segment& other)
        {
            this->updateBoundaries(other.xMin, other.yMin);
            this->updateBoundaries(other.xMax, other.yMax);
        }
    };
}

#endif
