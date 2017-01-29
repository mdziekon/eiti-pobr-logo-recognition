#include "./binarization.hpp"

#include "../../utils/consts.hpp"
#include "./converters.hpp"
#include "./matrix-ops.hpp"

namespace consts = pobr::utils::consts;
namespace converters = pobr::imgProcessing::utils::converters;
namespace matrixOps = pobr::imgProcessing::utils::matrixOps;

namespace binarization = pobr::imgProcessing::utils::binarization;

cv::Mat
binarization::binarizeImage(const cv::Mat& img, const unsigned int& threshold)
{
    auto resultImg = img.clone();

    matrixOps::forEachPixel(
        img,
        [&](const uint64_t& x, const uint64_t& y) -> void
        {
            auto& thisPixel = img.at<cv::Vec3b>(y, x);

            uint8_t value;
            cv::Vec3d hsv = converters::rgb2HSV(thisPixel);

            if (thisPixel[0] > threshold && thisPixel[1] > threshold && thisPixel[2] > threshold)
            {
                value = 255;
            }
            else if (hsv[0] > 85 && hsv[0] < 195 && hsv[2] > 50)
            {
                value = 255;
            }
            else
            {
                value = 0;
            }

            resultImg.at<cv::Vec3b>(y, x)[0] = value;
            resultImg.at<cv::Vec3b>(y, x)[1] = value;
            resultImg.at<cv::Vec3b>(y, x)[2] = value;
        }
    );

    return resultImg;
}

cv::Mat
binarization::binarizeImage(const cv::Mat& img, const cv::Vec3b& lowerBound, const cv::Vec3b& upperBound)
{
    auto resultImg = img.clone();

    matrixOps::forEachPixel(
        img,
        [&](const uint64_t& x, const uint64_t& y) -> void
        {
            auto& thisPixel = resultImg.at<cv::Vec3b>(y, x);

            uint8_t value = consts::colors::white;

            if (thisPixel[0] < lowerBound[0] || thisPixel[0] > upperBound[0]) {
                value = consts::colors::black;
            }
            if (thisPixel[1] < lowerBound[1] || thisPixel[1] > upperBound[1]) {
                value = consts::colors::black;
            }
            if (thisPixel[2] < lowerBound[2] || thisPixel[2] > upperBound[2]) {
                value = consts::colors::black;
            }

            thisPixel[0] = value;
            thisPixel[1] = value;
            thisPixel[2] = value;
        }
    );

    return resultImg;
}

cv::Mat
binarization::invertBinaryImage(const cv::Mat& img)
{
    auto resultImg = img.clone();

    matrixOps::forEachPixel(
        img,
        [&](const uint64_t& x, const uint64_t& y) -> void
        {
            auto& thisPixel = resultImg.at<cv::Vec3b>(y, x);
            uint8_t value = 255 - thisPixel[0];

            thisPixel[0] = value;
            thisPixel[1] = value;
            thisPixel[2] = value;
        }
    );

    return resultImg;
}
