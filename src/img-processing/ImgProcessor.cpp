#include "ImgProcessor.hpp"

#include <array>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../utils/consts.hpp"
#include "../utils/error-handler/ErrorHandler.hpp"

namespace consts = pobr::utils::consts;
using ErrorHandler = pobr::utils::ErrorHandler;
using ImgProcessor = pobr::imgProcessing::ImgProcessor;

const bool
ImgProcessor::isReady()
const
{
    return !(this->img.empty());
}

const void
ImgProcessor::assertIsReady()
const
{
    if (this->isReady()) {
        return;
    }

    ErrorHandler::error("ImgProcessor has no image loaded yet!");
}

const cv::Mat&
ImgProcessor::forEachPixel(
    const cv::Mat& img,
    const std::function<void(const uint64_t& x, const uint64_t& y)>& operation
)
const
{
    for (uint64_t x = 0; x < img.cols; x++) {
        for (uint64_t y = 0; y < img.rows; y++) {
            operation(x, y);
        }
    }

    return img;
}

template<class Acc>
Acc
ImgProcessor::reduceEachPixel(
    const cv::Mat& img,
    Acc accumulator,
    const std::function<Acc(const uint64_t& x, const uint64_t& y, Acc& accumulator)>& operation
)
const
{
    for (uint64_t x = 0; x < img.cols; x++) {
        for (uint64_t y = 0; y < img.rows; y++) {
            accumulator = operation(x, y, accumulator);
        }
    }

    return accumulator;
}

template<class PixelClass, class Acc, class KernelValue>
cv::Mat
ImgProcessor::applyKernel(
    const cv::Mat& img,
    const cv::Mat& kernel,
    Acc accumulatorInit,
    const std::function<Acc(const uint64_t& x, const uint64_t& y, Acc& accumulator, const PixelClass& pixel, const KernelValue& kernelValue)>& reducer,
    const std::function<void(const uint64_t& x, const uint64_t& y, Acc& accumulator, PixelClass& pixel, const cv::Mat& img)>& applicator
)
const
{
    // Uses edge cropping
    auto resultImg = img.clone();

    const unsigned int initRow = 0 + ((kernel.rows - 1) / 2);
    const unsigned int initCol = 0 + ((kernel.cols - 1) / 2);

    const unsigned int endRow = (img.rows - 1) - initRow;
    const unsigned int endCol = (img.cols - 1) - initCol;

    const unsigned int kernelOffsetY = 0 + ((kernel.rows - 1) / 2);
    const unsigned int kernelOffsetX = 0 + ((kernel.cols - 1) / 2);

    this->forEachPixel(
        resultImg,
        [&](const uint64_t& x, const uint64_t& y) -> void
        {
            if (x < initCol) {
                return;
            }
            if (y < initRow) {
                return;
            }
            if (x > endCol) {
                return;
            }
            if (y > endRow) {
                return;
            }

            Acc value = this->reduceEachPixel<Acc>(
                kernel,
                accumulatorInit,
                [&](const uint64_t& kernelX, const uint64_t& kernelY, Acc& accumulator) -> Acc
                {
                    auto& thisKernelValue = kernel.at<KernelValue>(kernelY, kernelX);
                    auto& adjacentPixel = img.at<PixelClass>(
                        y - kernelOffsetY + kernelY,
                        x - kernelOffsetX + kernelX
                    );

                    return reducer(x, y, accumulator, adjacentPixel, thisKernelValue);
                }
            );

            auto& thisPixel = resultImg.at<PixelClass>(y, x);

            applicator(x, y, value, thisPixel, img);
        }
    );

    return resultImg;
}

const void
ImgProcessor::loadImg(const std::string& imgPath)
{
    this->img = cv::imread(imgPath);

    if (this->img.empty()) {
        ErrorHandler::warning("Could not properly load image \"" + imgPath + "\"...");
    }
}

const void
ImgProcessor::process()
const
{
    this->assertIsReady();

    cv::imshow("test", this->unsharpMasking(this->img));

    cv::waitKey(-1);
}

cv::Mat
ImgProcessor::binarizeImage(const cv::Mat& img, const unsigned int& threshold)
const
{
    auto resultImg = img.clone();

    this->forEachPixel(
        img,
        [&](const uint64_t& x, const uint64_t& y) -> void
        {
            auto& thisPixel = img.at<cv::Vec3b>(y, x);

            uint8_t value;
            cv::Vec3d hsv = this->rgb2HSV(thisPixel);

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
ImgProcessor::binarizeImage(const cv::Mat& img, const cv::Vec3b& lowerBound, const cv::Vec3b& upperBound)
const
{
    auto resultImg = img.clone();

    this->forEachPixel(
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
ImgProcessor::grayscaleImage(const cv::Mat& img)
const
{
    auto resultImg = img.clone();

    this->forEachPixel(
        img,
        [&](const uint64_t& x, const uint64_t& y) -> void
        {
            auto& thisPixel = img.at<cv::Vec3b>(y, x);

            uint8_t value = (thisPixel[0] + thisPixel[1] + thisPixel[2]) / 3;

            resultImg.at<cv::Vec3b>(y, x)[0] = value;
            resultImg.at<cv::Vec3b>(y, x)[1] = value;
            resultImg.at<cv::Vec3b>(y, x)[2] = value;
        }
    );

    return resultImg;
}

cv::Mat
ImgProcessor::detectEdges(const cv::Mat& img)
const
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

    resultImg = this->grayscaleImage(resultImg);

    resultImg = this->applyKernel<cv::Vec3b, double, double>(
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

cv::Mat
ImgProcessor::unsharpMasking(const cv::Mat& img)
const
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

    std::array<double, 3> accumulatorInit = { 0.0, 0.0, 0.0 };

    resultImg = this->applyKernel<cv::Vec3b, std::array<double, 3>, double>(
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
