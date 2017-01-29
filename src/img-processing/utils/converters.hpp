#ifndef POBR_IMGPROCESSING_UTILS_CONVERTERS_HPP
#define POBR_IMGPROCESSING_UTILS_CONVERTERS_HPP

#include <opencv2/core/core.hpp>

namespace pobr::imgProcessing::utils::converters
{
    cv::Vec3d rgb2HSV(const cv::Vec3b opencvRGB);
    cv::Mat grayscaleImage(const cv::Mat& img);
}

#endif
