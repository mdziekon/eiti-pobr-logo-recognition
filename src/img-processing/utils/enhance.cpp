#include "./enhance.hpp"

#include <algorithm>

#include "../../utils/consts.hpp"
#include "./matrix-ops.hpp"

namespace consts = pobr::utils::consts;
namespace matrixOps = pobr::imgProcessing::utils::matrixOps;

namespace enhance = pobr::imgProcessing::utils::enhance;

cv::Mat
enhance::erodeImage(const cv::Mat& img, const unsigned int& windowSize)
{
    auto resultImg = img.clone();

    auto kernel = cv::Mat(
        windowSize,
        windowSize,
        CV_64F
    );

    double erosionThreshold = 1.0 * (windowSize * windowSize) * consts::colors::white;

    // Initialize kernel values
    for (unsigned int i = 0; i < windowSize; i++) {
        for (unsigned int j = 0; j < windowSize; j++) {
            kernel.at<double>(j, i) = 1;
        }
    }

    resultImg = matrixOps::applyKernel<cv::Vec3b, double, double>(
        resultImg,
        kernel,
        0.0,
        [](const uint64_t& x, const uint64_t& y, double& accumulator, const cv::Vec3b& pixel, const double& kernelValue) -> double
        {
            return accumulator + (kernelValue * pixel[0]);
        },
        [&erosionThreshold](const uint64_t& x, const uint64_t& y, double& accumulator, cv::Vec3b& pixel, const cv::Mat& img) -> void
        {
            double value = consts::colors::black;

            if (accumulator == erosionThreshold) {
                value = consts::colors::white;
            }

            pixel[0] = value;
            pixel[1] = value;
            pixel[2] = value;
        }
    );

    return resultImg;
}

cv::Mat
enhance::dilateImage(const cv::Mat& img, const unsigned int& windowSize)
{
    auto resultImg = img.clone();

    auto kernel = cv::Mat(
        windowSize,
        windowSize,
        CV_64F
    );

    // Initialize kernel values
    for (unsigned int i = 0; i < windowSize; i++) {
        for (unsigned int j = 0; j < windowSize; j++) {
            kernel.at<double>(j, i) = 1;
        }
    }

    resultImg = matrixOps::applyKernel<cv::Vec3b, double, double>(
        resultImg,
        kernel,
        0.0,
        [](const uint64_t& x, const uint64_t& y, double& accumulator, const cv::Vec3b& pixel, const double& kernelValue) -> double
        {
            return accumulator + (kernelValue * pixel[0]);
        },
        [](const uint64_t& x, const uint64_t& y, double& accumulator, cv::Vec3b& pixel, const cv::Mat& img) -> void
        {
            double value = consts::colors::black;

            if (accumulator > 0) {
                value = consts::colors::white;
            }

            pixel[0] = value;
            pixel[1] = value;
            pixel[2] = value;
        }
    );

    return resultImg;
}

cv::Mat
enhance::unsharpMasking(const cv::Mat& img)
{
    auto resultImg = img.clone();

    double kernelValues[25] = {
        1,  4,    6,  4, 1,
        4, 16,   24, 16, 4,
        6, 24, -476, 24, 6,
        4, 16,   24, 16, 4,
        1,  4,    6,  4, 1        
    };

    auto kernel = cv::Mat(
        5,
        5,
        CV_64F,
        kernelValues
    );

    std::array<double, 3> accumulatorInit = { { 0.0, 0.0, 0.0 } };

    resultImg = matrixOps::applyKernel<cv::Vec3b, std::array<double, 3>, double>(
        resultImg,
        kernel,
        accumulatorInit,
        [](const uint64_t& x, const uint64_t& y, std::array<double, 3>& accumulator, const cv::Vec3b& pixel, const double& kernelValue) -> std::array<double, 3>
        {
            accumulator[0] = accumulator[0] + (kernelValue * pixel[0]);
            accumulator[1] = accumulator[1] + (kernelValue * pixel[1]);
            accumulator[2] = accumulator[2] + (kernelValue * pixel[2]);

            return accumulator;
        },
        [](const uint64_t& x, const uint64_t& y, std::array<double, 3>& accumulator, cv::Vec3b& pixel, const cv::Mat& img) -> void
        {
            accumulator[0] = accumulator[0] / 256 * -1;
            accumulator[1] = accumulator[1] / 256 * -1;
            accumulator[2] = accumulator[2] / 256 * -1;

            accumulator[0] = std::min(accumulator[0], 255.0);
            accumulator[0] = std::max(accumulator[0], 0.0);
            accumulator[1] = std::min(accumulator[1], 255.0);
            accumulator[1] = std::max(accumulator[1], 0.0);
            accumulator[2] = std::min(accumulator[2], 255.0);
            accumulator[2] = std::max(accumulator[2], 0.0);

            pixel[0] = accumulator[0];
            pixel[1] = accumulator[1];
            pixel[2] = accumulator[2];
        }
    );

    return resultImg;
}
