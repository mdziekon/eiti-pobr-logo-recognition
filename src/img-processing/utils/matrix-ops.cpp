#include "./matrix-ops.hpp"

namespace matrixOps = pobr::imgProcessing::utils::matrixOps;

const cv::Mat&
matrixOps::forEachPixel(
    const cv::Mat& img,
    const std::function<void(const uint64_t& x, const uint64_t& y)>& operation
)
{
    for (uint64_t x = 0; x < img.cols; x++) {
        for (uint64_t y = 0; y < img.rows; y++) {
            operation(x, y);
        }
    }

    return img;
}
