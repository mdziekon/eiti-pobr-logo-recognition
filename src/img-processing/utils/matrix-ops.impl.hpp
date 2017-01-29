#ifndef POBR_IMGPROCESSING_UTILS_MATRIXOPS_IMPL_HPP
#define POBR_IMGPROCESSING_UTILS_MATRIXOPS_IMPL_HPP

#include "./matrix-ops.hpp"

namespace matrixOps = pobr::imgProcessing::utils::matrixOps;

template<class Acc>
Acc
matrixOps::reduceEachPixel(
    const cv::Mat& img,
    Acc accumulator,
    const std::function<Acc(const uint64_t& x, const uint64_t& y, Acc& accumulator)>& operation
)
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
matrixOps::applyKernel(
    const cv::Mat& img,
    const cv::Mat& kernel,
    Acc accumulatorInit,
    const std::function<Acc(const uint64_t& x, const uint64_t& y, Acc& accumulator, const PixelClass& pixel, const KernelValue& kernelValue)>& reducer,
    const std::function<void(const uint64_t& x, const uint64_t& y, Acc& accumulator, PixelClass& pixel, const cv::Mat& img)>& applicator
)
{
    // Uses edge cropping
    auto resultImg = img.clone();

    const unsigned int initRow = 0 + ((kernel.rows - 1) / 2);
    const unsigned int initCol = 0 + ((kernel.cols - 1) / 2);

    const unsigned int endRow = (img.rows - 1) - initRow;
    const unsigned int endCol = (img.cols - 1) - initCol;

    const unsigned int kernelOffsetY = 0 + ((kernel.rows - 1) / 2);
    const unsigned int kernelOffsetX = 0 + ((kernel.cols - 1) / 2);

    matrixOps::forEachPixel(
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

            Acc value = matrixOps::reduceEachPixel<Acc>(
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

#endif
