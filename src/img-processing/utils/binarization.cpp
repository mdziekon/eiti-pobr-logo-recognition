#include "./binarization.hpp"

#include "../../utils/consts.hpp"
#include "./converters.hpp"
#include "./matrix-ops.hpp"

namespace consts = pobr::utils::consts;
namespace converters = pobr::imgProcessing::utils::converters;
namespace matrixOps = pobr::imgProcessing::utils::matrixOps;

namespace binarization = pobr::imgProcessing::utils::binarization;

cv::Mat
binarization::mixImageColors(const cv::Mat& img, const cv::Vec3i& coefficients, const bool& preserveLuminosity)
{
    auto resultImg = img.clone();

    matrixOps::forEachPixel(
        img,
        [&](const uint64_t& x, const uint64_t& y) -> void
        {
            auto& thisPixel = img.at<cv::Vec3b>(y, x);

            int value = (
                (((double) thisPixel[0]) * (((double) coefficients[0])) / 100) +
                (((double) thisPixel[1]) * (((double) coefficients[1])) / 100) +
                (((double) thisPixel[2]) * (((double) coefficients[2])) / 100)
            );

            if (preserveLuminosity) {
                // TODO: not working properly, flipping the sign
                double scaleNominator = ((double) (coefficients[0] + coefficients[1] + coefficients[2])) / 100;
                double scale = 1 / scaleNominator;

                value = value * scale;
            }

            value = std::min(value, 255);
            value = std::max(value, 0);

            resultImg.at<cv::Vec3b>(y, x)[0] = value;
            resultImg.at<cv::Vec3b>(y, x)[1] = value;
            resultImg.at<cv::Vec3b>(y, x)[2] = value;
        }
    );

    return resultImg;
}

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

            if (thisPixel[0] > threshold && thisPixel[1] > threshold && thisPixel[2] > threshold)
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

cv::Mat
binarization::detectEdges(const cv::Mat& img)
{
    auto resultImg = img.clone();

    double kernelValues[9] = {
        -1, -1, -1,
        -1, 8, -1,
        -1, -1, -1
    };

    auto kernel = cv::Mat(
        3,
        3,
        CV_64F,
        kernelValues
    );

    resultImg = converters::grayscaleImage(resultImg);

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
            double value = accumulator;

            value = std::min(value, 255.0);
            value = std::max(value, 0.0);

            pixel[0] = value;
            pixel[1] = value;
            pixel[2] = value;
        }
    );

    return resultImg;
}
