#ifndef POBR_IMGPROCESSING_UTILS_ENHANCE_HPP
#define POBR_IMGPROCESSING_UTILS_ENHANCE_HPP

#include <opencv2/core/core.hpp>

namespace pobr::imgProcessing::utils::enhance
{
    cv::Mat erodeImage(const cv::Mat& img, const unsigned int& windowSize);
    cv::Mat dilateImage(const cv::Mat& img, const unsigned int& windowSize);
    cv::Mat unsharpMasking(const cv::Mat& img);
}

#endif
