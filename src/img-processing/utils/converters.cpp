#include "./converters.hpp"

#include <algorithm>
#include "./matrix-ops.hpp"

namespace matrixOps = pobr::imgProcessing::utils::matrixOps;

namespace converters = pobr::imgProcessing::utils::converters;

cv::Vec3d
converters::rgb2HSV(const cv::Vec3b opencvRGB)
{
    double hue = -1;
    double sat = -1;
    double val = -1;

    uint8_t red = opencvRGB[2];
    uint8_t green = opencvRGB[1];
    uint8_t blue = opencvRGB[0];

    double tmp = std::min(std::min(red, green), blue);

    val = std::max(std::max(red, green), blue);

    if (tmp == val) {
        hue = 0;
    } else {
        if (red == val) {
            hue = 0 + ((green - blue) * 60 / (val - tmp));
        } else if (green == val) {
            hue = 120 + ((blue - red) * 60 / (val - tmp));
        } else {
            hue = 240 + ((red - green) * 60 / (val - tmp));
        }
    }

    if (hue < 0) {
        hue += 360;
    }

    if (val == 0) {
        sat = 0;
    } else {
        sat = (val - tmp) * 100 / val;
    }

    val = (100 * val) / 256;

    cv::Vec3d hsv;

    hsv[0] = hue;
    hsv[1] = sat;
    hsv[2] = val;

    return hsv;
}

cv::Mat
converters::grayscaleImage(const cv::Mat& img)
{
    auto resultImg = img.clone();

    matrixOps::forEachPixel(
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
