#ifndef POBR_IMGPROCESSING_STRUCTS_SEGMENT_HPP
#define POBR_IMGPROCESSING_STRUCTS_SEGMENT_HPP

#include <cstdint>
#include <opencv2/core/core.hpp>

#include "../../utils/consts.hpp"

namespace consts = pobr::utils::consts;

namespace pobr::imgProcessing::structs
{
    struct Segment
    {
    public:
        uint64_t xMin = 0;
        uint64_t xMax = 0;
        uint64_t yMin = 0;
        uint64_t yMax = 0;

        cv::Mat_<cv::Vec3b> pixels;

        const void updateBoundaries(const uint64_t& x, const uint64_t& y);
        const void updatePixels(const cv::Mat_<cv::Vec3i>& segmentedImg, const int& segmentID);

        const void merge(const Segment& other);
    };
}

#endif
