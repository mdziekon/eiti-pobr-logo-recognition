#ifndef POBR_IMGPROCESSING_UTILS_MATRIXOPS_HPP
#define POBR_IMGPROCESSING_UTILS_MATRIXOPS_HPP

#include <cstdint>
#include <functional>
#include <opencv2/core/core.hpp>

namespace pobr::imgProcessing::utils::matrixOps
{
    const cv::Mat& forEachPixel(
        const cv::Mat& img,
        const std::function<void(const uint64_t& x, const uint64_t& y)>& operation
    );

    template<class Acc>
    Acc reduceEachPixel(
        const cv::Mat& img,
        Acc accumulator,
        const std::function<Acc(const uint64_t& x, const uint64_t& y, Acc& accumulator)>& operation
    );

    template<class PixelClass, class Acc, class KernelValue>
    cv::Mat applyKernel(
        const cv::Mat& img,
        const cv::Mat& kernel,
        Acc accumulatorInit,
        const std::function<Acc(const uint64_t& x, const uint64_t& y, Acc& accumulator, const PixelClass& pixel, const KernelValue& kernelValue)>& reducer,
        const std::function<void(const uint64_t& x, const uint64_t& y, Acc& accumulator, PixelClass& pixel, const cv::Mat& img)>& applicator
    );
}

#include "./matrix-ops.impl.hpp"

#endif
