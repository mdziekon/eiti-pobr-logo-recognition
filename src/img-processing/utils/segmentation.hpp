#ifndef POBR_IMGPROCESSING_UTILS_SEGMENTATION_HPP
#define POBR_IMGPROCESSING_UTILS_SEGMENTATION_HPP

#include <vector>
#include <opencv2/core/core.hpp>

#include "../structs/Segment.hpp"

namespace structs = pobr::imgProcessing::structs;

namespace pobr::imgProcessing::utils::segmentation
{
    std::vector<structs::Segment> getImageSegmentsScanMerge(
        const cv::Mat& img,
        const bool& useDiagonalDetection = true
    );
    std::vector<structs::Segment> getImageSegmentsFloodFill(
        const cv::Mat& img,
        const bool& diagDetection = false
    );
}

#endif
