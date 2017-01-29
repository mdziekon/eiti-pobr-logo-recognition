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

const std::vector<structs::Segment>
ImgProcessor::process()
const
{
    this->assertIsReady();

    auto img = this->img;

    img = this->processPreEnhance(img);
    img = this->processBinarize(img);
    img = this->processBinaryEnhance(img);

    auto segments = this->processSegmentation(img);
    auto candidates = this->processFilterCandidates(segments);
    auto letterSegments = this->processDetection(candidates);

    return letterSegments;
}

cv::Mat
ImgProcessor::processPreEnhance(const cv::Mat& img)
const
{
    auto resultImg = img;

    PerformanceTimer profiler;

    profiler.start();

    // Do nothing

    profiler.stop();

    Logger::notice(
        std::string("PreEnhance phase took: ") +
        std::to_string(profiler.getDurationNS() / 1000000) +
        std::string("ms")
    );

    return resultImg;
}

cv::Mat
ImgProcessor::processBinarize(const cv::Mat& img)
const
{
    auto resultImg = img;

    PerformanceTimer profiler;

    profiler.start();

    resultImg = binarization::binarizeImage(
        resultImg,
        cv::Vec3b(0, 0, 75),
        cv::Vec3b(180, 120, 255)
    );
    // resultImg = binarization::invertBinaryImage(resultImg);

    profiler.stop();

    Logger::notice(
        std::string("Binarize phase took: ") +
        std::to_string(profiler.getDurationNS() / 1000000) +
        std::string("ms")
    );

    return resultImg;
}

cv::Mat
ImgProcessor::processBinaryEnhance(const cv::Mat& img)
const
{
    auto resultImg = img;

    PerformanceTimer profiler;

    // TODO: Make sure this is in proper order
    profiler.start();

    // resultImg = enhance::erodeImage(
    //     resultImg,
    //     3
    // );
    // resultImg = enhance::dilateImage(
    //     resultImg,
    //     3
    // );

    profiler.stop();

    Logger::notice(
        std::string("BinaryEnhance phase took: ") +
        std::to_string(profiler.getDurationNS() / 1000000) +
        std::string("ms")
    );

    return resultImg;
}

std::vector<structs::Segment>
ImgProcessor::processSegmentation(const cv::Mat& img)
const
{
    auto resultImg = img;

    PerformanceTimer profiler;

    profiler.start();

    auto segments = segmentation::getImageSegmentsFloodFill(resultImg);

    profiler.stop();

    Logger::notice(
        std::string("Segmentation phase took: ") +
        std::to_string(profiler.getDurationNS() / 1000000) +
        std::string("ms")
    );

    return segments;
}

std::vector<structs::Candidate>
ImgProcessor::processFilterCandidates(
    const std::vector<structs::Segment>& segments
)
const
{
    PerformanceTimer profiler;

    profiler.start();

    auto candidates = detection::findImageLogoCandidates(segments);

    std::vector<structs::Candidate> matchingCandidates;

    for (auto& candidate: candidates) {
        // Logger::notice(cand.segment.classify());

        if (!candidate.segment.isClassifiedAsLetter()) {
            continue;
        }

        matchingCandidates.push_back(candidate);

        // Logger::notice("-------");
        // Logger::notice("segment " + std::to_string(cand.segment.xMin) + "x" + std::to_string(cand.segment.yMin));
        // Logger::notice("hu 1 = " + std::to_string(cand.segment.getHuMomentInvariant(1)));
        // Logger::notice("hu 2 = " + std::to_string(cand.segment.getHuMomentInvariant(2)));
        // Logger::notice("hu 3 = " + std::to_string(cand.segment.getHuMomentInvariant(3)));
        // Logger::notice("hu 4 = " + std::to_string(cand.segment.getHuMomentInvariant(4)));
        // Logger::notice("hu 5 = " + std::to_string(cand.segment.getHuMomentInvariant(5)));
        // Logger::notice("hu 6 = " + std::to_string(cand.segment.getHuMomentInvariant(6)));
        // Logger::notice("hu 7 = " + std::to_string(cand.segment.getHuMomentInvariant(7)));

        // Logger::notice("-------");
        // Logger::notice("segment " + std::to_string(cand.segment.xMin) + "x" + std::to_string(cand.segment.yMin));
        // Logger::notice("classification = " + cand.segment.classify());
    }

    profiler.stop();

    Logger::notice(
        std::string("FilterCandidates phase took: ") +
        std::to_string(profiler.getDurationNS() / 1000000) +
        std::string("ms")
    );

    return matchingCandidates;
}

std::vector<structs::Segment>
ImgProcessor::processDetection(const std::vector<structs::Candidate>& candidates)
const
{
    PerformanceTimer profiler;

    profiler.start();

    std::vector<structs::Segment> segments;

    for (auto& candidate: candidates) {
        segments.push_back(candidate.segment);
    }

    profiler.stop();

    Logger::notice(
        std::string("Detection phase took: ") +
        std::to_string(profiler.getDurationNS() / 1000000) +
        std::string("ms")
    );

    return segments;
}

cv::Mat
ImgProcessor::drawSegmentsBBoxes(
    const cv::Mat& img,
    const std::vector<structs::Segment>& segments,
    const cv::Vec3b& borderColor
)
const
{
    auto resultImg = img.clone();

    for (auto& segment: segments) {
        // TODO: Safeguards maybe?

        for (uint64_t x = segment.xMin - 1; x <= segment.xMax + 1; x++) {
            auto& point1 = resultImg.at<cv::Vec3b>(segment.yMin - 1, x);
            auto& point2 = resultImg.at<cv::Vec3b>(segment.yMax + 1, x);

            point1[0] = point2[0] = borderColor[0];
            point1[1] = point2[1] = borderColor[1];
            point1[2] = point2[2] = borderColor[2];
        }
        for (uint64_t y = segment.yMin - 1; y <= segment.yMax + 1; y++) {
            auto& point1 = resultImg.at<cv::Vec3b>(y, segment.xMin - 1);
            auto& point2 = resultImg.at<cv::Vec3b>(y, segment.xMax + 1);

            point1[0] = point2[0] = borderColor[0];
            point1[1] = point2[1] = borderColor[1];
            point1[2] = point2[2] = borderColor[2];
        }
    }

    return resultImg;
}
