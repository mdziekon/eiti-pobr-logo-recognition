#ifndef POBR_IMGPROCESSING_UTILS_BINARIZATION_HPP
#define POBR_IMGPROCESSING_UTILS_BINARIZATION_HPP

#include <opencv2/core/core.hpp>

namespace pobr::imgProcessing::utils::binarization
{
    cv::Mat binarizeImage(const cv::Mat& img, const unsigned int& threshold);
    cv::Mat binarizeImage(const cv::Mat& img, const cv::Vec3b& lowerBound, const cv::Vec3b& upperBound);
    cv::Mat invertBinaryImage(const cv::Mat& img);
    cv::Mat detectEdges(const cv::Mat& img);
}

#endif
