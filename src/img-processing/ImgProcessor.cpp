#include "ImgProcessor.hpp"

#include <opencv2/highgui/highgui.hpp>

#include "../utils/consts.hpp"
#include "../utils/logger/Logger.hpp"
#include "../utils/performance-timer/PerformanceTimer.hpp"
#include "./utils/converters.hpp"
#include "./utils/matrix-ops.hpp"
#include "./utils/binarization.hpp"
#include "./utils/enhance.hpp"
#include "./utils/segmentation.hpp"
#include "./utils/detection.hpp"

namespace consts = pobr::utils::consts;
namespace converters = pobr::imgProcessing::utils::converters;
namespace matrixOps = pobr::imgProcessing::utils::matrixOps;
namespace binarization = pobr::imgProcessing::utils::binarization;
namespace enhance = pobr::imgProcessing::utils::enhance;
namespace segmentation = pobr::imgProcessing::utils::segmentation;
namespace detection = pobr::imgProcessing::utils::detection;

using Logger = pobr::utils::Logger;
using PerformanceTimer = pobr::utils::PerformanceTimer;
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

    Logger::error("ImgProcessor has no image loaded yet!");
}

const void
ImgProcessor::loadImg(const std::string& imgPath)
{
    this->img = cv::imread(imgPath);

    if (this->img.empty()) {
        Logger::warning("Could not properly load image \"" + imgPath + "\"...");
    }
}

const cv::Mat&
ImgProcessor::getImg()
const
{
    this->assertIsReady();

    return this->img;
}

const cv::Mat
ImgProcessor::getBinarizedImg()
const
{
    this->assertIsReady();

    auto img = this->img;

    img = this->processPreEnhance(img);
    img = this->processBinarize(img);
    img = this->processBinaryEnhance(img);

    return img;
}

const std::vector<structs::Segment>
ImgProcessor::process()
const
{
    this->assertIsReady();

    auto img = this->img;

    img = this->processPreEnhance(img, true);
    img = this->processBinarize(img, true);
    img = this->processBinaryEnhance(img, true);

    auto segments = this->processSegmentation(img, true);
    auto candidates = this->processFilterCandidates(segments, true);
    auto letterSegments = this->processDetection(candidates, true);

    return letterSegments;
}

cv::Mat
ImgProcessor::processPreEnhance(const cv::Mat& img, const bool& isProfiling)
const
{
    auto resultImg = img;

    PerformanceTimer profiler;

    profiler.start();

    // Note: disabled, as not needed
    //       supplied images are rather sharp

    // resultImg = enhance::unsharpMasking(resultImg);

    profiler.stop();

    if (isProfiling) {
        Logger::notice(
            std::string("PreEnhance phase took: ") +
            std::to_string(profiler.getDurationNS() / 1000000) +
            std::string("ms")
        );
    }

    return resultImg;
}

cv::Mat
ImgProcessor::processBinarize(const cv::Mat& img, const bool& isProfiling)
const
{
    auto resultImg = img;

    PerformanceTimer profiler;

    profiler.start();

    // Note: previous method of image binarization
    //       replaced by much better "color mixer + thresholding" method

    // resultImg = binarization::binarizeImage(
    //     resultImg,
    //     cv::Vec3b(0, 0, 75),
    //     cv::Vec3b(180, 120, 255)
    // );
    // resultImg = binarization::invertBinaryImage(resultImg);

    resultImg = binarization::mixImageColors(
        resultImg,
        { -125, -140, 180 },
        false
    );
    resultImg = binarization::binarizeImage(
        resultImg,
        50
    );

    profiler.stop();

    if (isProfiling) {
        Logger::notice(
            std::string("Binarize phase took: ") +
            std::to_string(profiler.getDurationNS() / 1000000) +
            std::string("ms")
        );
    }

    return resultImg;
}

cv::Mat
ImgProcessor::processBinaryEnhance(const cv::Mat& img, const bool& isProfiling)
const
{
    auto resultImg = img;

    PerformanceTimer profiler;

    profiler.start();

    // Note: currently disabled, as it's:
    //       1. not needed in here
    //       2. breaks some logos recognition

    // resultImg = enhance::erodeImage(
    //     resultImg,
    //     3
    // );
    // resultImg = enhance::dilateImage(
    //     resultImg,
    //     3
    // );

    profiler.stop();

    if (isProfiling) {
        Logger::notice(
            std::string("BinaryEnhance phase took: ") +
            std::to_string(profiler.getDurationNS() / 1000000) +
            std::string("ms")
        );
    }

    return resultImg;
}

std::vector<structs::Segment>
ImgProcessor::processSegmentation(const cv::Mat& img, const bool& isProfiling)
const
{
    auto resultImg = img;

    PerformanceTimer profiler;

    profiler.start();

    auto segments = segmentation::getImageSegmentsFloodFill(resultImg, false);

    profiler.stop();

    if (isProfiling) {
        Logger::notice(
            std::string("Segmentation phase took: ") +
            std::to_string(profiler.getDurationNS() / 1000000) +
            std::string("ms")
        );
    }

    return segments;
}

std::vector<structs::Segment>
ImgProcessor::processFilterCandidates(
    const std::vector<structs::Segment>& segments,
    const bool& isProfiling
)
const
{
    PerformanceTimer profiler;

    profiler.start();

    std::vector<structs::Segment> filteredSegments;

    for (auto& segment: segments) {
        if (!segment.isClassifiedAsLetter()) {
            continue;
        }

        filteredSegments.push_back(segment);
    }

    profiler.stop();

    if (isProfiling) {
        Logger::notice(
            std::string("FilterCandidates phase took: ") +
            std::to_string(profiler.getDurationNS() / 1000000) +
            std::string("ms")
        );
    }

    return filteredSegments;
}

std::vector<structs::Segment>
ImgProcessor::processDetection(
    const std::vector<structs::Segment>& segments,
    const bool& isProfiling
)
const
{
    PerformanceTimer profiler;

    profiler.start();

    auto groupedSegments = detection::groupLetters(segments);

    profiler.stop();

    if (isProfiling) {
        Logger::notice(
            std::string("Detection phase took: ") +
            std::to_string(profiler.getDurationNS() / 1000000) +
            std::string("ms")
        );
    }

    return groupedSegments;
}

cv::Mat
ImgProcessor::drawSegmentsBBoxes(
    const cv::Mat& img,
    const std::vector<structs::Segment>& segments,
    const cv::Vec3b& borderColor,
    const unsigned int& borderSize
)
const
{
    auto resultImg = img.clone();

    for (auto& segment: segments) {
        for (int64_t x = segment.xMin - borderSize; x < segment.xMin; x++) {
            for (int64_t y = segment.yMin - borderSize; y <= segment.yMax + borderSize; y++) {
                if (x < 0 || y < 0 || x >= img.cols || y >= img.rows) {
                    continue;
                }

                auto& point = resultImg.at<cv::Vec3b>(y, x);

                point[0] = borderColor[0];
                point[1] = borderColor[1];
                point[2] = borderColor[2];
            }
        }
        for (int64_t x = segment.xMax + 1; x < segment.xMax + 1 + borderSize; x++) {
            for (int64_t y = segment.yMin - borderSize; y <= segment.yMax + borderSize; y++) {
                if (x < 0 || y < 0 || x >= img.cols || y >= img.rows) {
                    continue;
                }

                auto& point = resultImg.at<cv::Vec3b>(y, x);

                point[0] = borderColor[0];
                point[1] = borderColor[1];
                point[2] = borderColor[2];
            }
        }
        for (int64_t y = segment.yMin - borderSize; y < segment.yMin; y++) {
            for (int64_t x = segment.xMin - borderSize; x <= segment.xMax + borderSize; x++) {
                if (x < 0 || y < 0 || x >= img.cols || y >= img.rows) {
                    continue;
                }

                auto& point = resultImg.at<cv::Vec3b>(y, x);

                point[0] = borderColor[0];
                point[1] = borderColor[1];
                point[2] = borderColor[2];
            }
        }
        for (int64_t y = segment.yMax + 1; y < segment.yMax + 1 + borderSize; y++) {
            for (int64_t x = segment.xMin - borderSize; x <= segment.xMax + borderSize; x++) {
                if (x < 0 || y < 0 || x >= img.cols || y >= img.rows) {
                    continue;
                }

                auto& point = resultImg.at<cv::Vec3b>(y, x);

                point[0] = borderColor[0];
                point[1] = borderColor[1];
                point[2] = borderColor[2];
            }
        }
    }

    return resultImg;
}
