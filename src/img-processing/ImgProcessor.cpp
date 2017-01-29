#include "ImgProcessor.hpp"

#include <algorithm>
#include <functional>
#include <array>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../utils/consts.hpp"
#include "../utils/error-handler/ErrorHandler.hpp"
#include "../utils/performance-timer/PerformanceTimer.hpp"
#include "./utils/matrix-ops.hpp"

namespace consts = pobr::utils::consts;
namespace matrixOps = pobr::imgProcessing::utils::matrixOps;

using ErrorHandler = pobr::utils::ErrorHandler;
using PerformanceTimer = pobr::utils::PerformanceTimer;
using Segment = pobr::imgProcessing::structs::Segment;
using Candidate = pobr::imgProcessing::structs::Candidate;
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

cv::Vec3d
ImgProcessor::rgb2HSV(const cv::Vec3b opencvRGB)
const
{
    double hue = -1;
    double sat = -1;
    double val = -1;

    uint8_t red = opencvRGB[2];
    uint8_t green = opencvRGB[1];
    uint8_t blue = opencvRGB[0];

    double tmp = std::min(std::min(red, green), blue);

    val = std::max(std::max(red, green), blue);

    if (tmp == val)
    {
        hue = 0;
    }
    else
    {
        if (red == val)
        {
            hue = 0 + ((green - blue) * 60 / (val - tmp));
        }
        else if (green == val)
        {
            hue = 120 + ((blue - red) * 60 / (val - tmp));
        }
        else
        {
            hue = 240 + ((red - green) * 60 / (val - tmp));
        }
    }

    if (hue < 0)
    {
        hue += 360;
    }

    if (val == 0)
    {
        sat = 0;
    }
    else
    {
        sat = (val - tmp) * 100 / val;
    }

    val = (100 * val) / 256;

    cv::Vec3d hsv;

    hsv[0] = hue;
    hsv[1] = sat;
    hsv[2] = val;

    return hsv;
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

    auto img = this->img;

    img = this->processPreEnhance(img);
    img = this->processBinarize(img);
    img = this->processBinaryEnhance(img);

    auto segments = this->processSegmentation(img);
    auto candidates = this->processFilterCandidates(img, segments);
    auto letterSegments = this->processDetection(candidates);

    img = this->drawSegmentsBBoxes(img, letterSegments);

    cv::imshow("test", img);

    cv::waitKey(-1);
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

    ErrorHandler::notice(
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

    resultImg = this->binarizeImage(
        resultImg,
        cv::Vec3b(0, 0, 75),
        cv::Vec3b(180, 120, 255)
    );
    // resultImg = this->invertBinaryImage(resultImg);

    profiler.stop();

    ErrorHandler::notice(
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

    // resultImg = this->erodeImage(
    //     resultImg,
    //     3
    // );
    // resultImg = this->dilateImage(
    //     resultImg,
    //     3
    // );

    profiler.stop();

    ErrorHandler::notice(
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

    auto segments = this->getImageSegmentsFloodFill(resultImg);

    profiler.stop();

    ErrorHandler::notice(
        std::string("Segmentation phase took: ") +
        std::to_string(profiler.getDurationNS() / 1000000) +
        std::string("ms")
    );

    return segments;
}

std::vector<structs::Candidate>
ImgProcessor::processFilterCandidates(const cv::Mat& img, const std::vector<structs::Segment>& segments)
const
{
    PerformanceTimer profiler;

    profiler.start();

    auto candidates = this->findImageLogoCandidates(img, segments);

    std::vector<structs::Candidate> matchingCandidates;

    for (auto& candidate: candidates) {
        // ErrorHandler::notice(cand.segment.classify());

        if (!candidate.segment.isClassifiedAsLetter()) {
            continue;
        }

        matchingCandidates.push_back(candidate);

        // ErrorHandler::notice("-------");
        // ErrorHandler::notice("segment " + std::to_string(cand.segment.xMin) + "x" + std::to_string(cand.segment.yMin));
        // ErrorHandler::notice("hu 1 = " + std::to_string(cand.segment.getHuMomentInvariant(1)));
        // ErrorHandler::notice("hu 2 = " + std::to_string(cand.segment.getHuMomentInvariant(2)));
        // ErrorHandler::notice("hu 3 = " + std::to_string(cand.segment.getHuMomentInvariant(3)));
        // ErrorHandler::notice("hu 4 = " + std::to_string(cand.segment.getHuMomentInvariant(4)));
        // ErrorHandler::notice("hu 5 = " + std::to_string(cand.segment.getHuMomentInvariant(5)));
        // ErrorHandler::notice("hu 6 = " + std::to_string(cand.segment.getHuMomentInvariant(6)));
        // ErrorHandler::notice("hu 7 = " + std::to_string(cand.segment.getHuMomentInvariant(7)));

        // ErrorHandler::notice("-------");
        // ErrorHandler::notice("segment " + std::to_string(cand.segment.xMin) + "x" + std::to_string(cand.segment.yMin));
        // ErrorHandler::notice("classification = " + cand.segment.classify());
    }

    profiler.stop();

    ErrorHandler::notice(
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

    ErrorHandler::notice(
        std::string("Detection phase took: ") +
        std::to_string(profiler.getDurationNS() / 1000000) +
        std::string("ms")
    );

    return segments;
}


cv::Mat
ImgProcessor::binarizeImage(const cv::Mat& img, const unsigned int& threshold)
const
{
    auto resultImg = img.clone();

    matrixOps::forEachPixel(
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
ImgProcessor::invertBinaryImage(const cv::Mat& img)
const
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
ImgProcessor::erodeImage(const cv::Mat& img, const unsigned int& windowSize)
const
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
ImgProcessor::dilateImage(const cv::Mat& img, const unsigned int& windowSize)
const
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
ImgProcessor::grayscaleImage(const cv::Mat& img)
const
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

std::vector<Segment>
ImgProcessor::getImageSegmentsScanMerge(const cv::Mat& img, const bool& useDiagonalDetection)
const
{
    auto segmentsIDs = cv::Mat(
        img.rows,
        img.cols,
        CV_64F,
        0.0
    );
    std::vector<Segment> segmentsBoundaries;
    std::unordered_map<double, std::unordered_set<double>> touchingSegments;

    double lookupKernelValues[9] = {
        0, 1, 0,
        1, 0, 1,
        0, 1, 0     
    };

    if (useDiagonalDetection) {
        lookupKernelValues[0] = 1;
        lookupKernelValues[2] = 1;
        lookupKernelValues[6] = 1;
        lookupKernelValues[8] = 1;
    }

    auto lookupKernel = cv::Mat(
        3,
        3,
        CV_64F,
        lookupKernelValues
    );

    // TODO: Kernel application skips img edges
    //       replace with forEachPixel maybe?
    matrixOps::applyKernel<double, double, double>(
        segmentsIDs,
        lookupKernel,
        0,
        [&](const uint64_t& x, const uint64_t& y, double& accumulator, const double& segmentID, const double& kernelValue) -> double
        {
            const auto& imgPixel = img.at<cv::Vec3b>(y, x);

            if (imgPixel[0] == consts::colors::black) {
                // Short-circuit as there is nothing to do here
                return 0;
            }

            if (kernelValue == 0) {
                return accumulator;
            }

            if (accumulator != 0) {
                if (segmentID == 0 || accumulator == segmentID) {
                    return accumulator;
                }

                // Segments touching, store that information for merging phase
                touchingSegments.at(accumulator).insert(segmentID);
                touchingSegments.at(segmentID).insert(accumulator);
            }

            return segmentID;
        },
        [&](const uint64_t& x, const uint64_t& y, double& accumulator, double& setSegmentID, const cv::Mat& tt) -> void
        {
            const auto& imgPixel = img.at<cv::Vec3b>(y, x);

            if (imgPixel[0] == consts::colors::black) {
                // Short-circuit as there is nothing to do here
                return;
            }

            if (accumulator == 0) {
                // Not IDed yet, create new segment

                const auto thisSegmentID = segmentsBoundaries.size() + 1;
                Segment newSegment;

                newSegment.xMin = x;
                newSegment.xMax = x;
                newSegment.yMin = y;
                newSegment.yMax = y;

                segmentsIDs.at<double>(y, x) = thisSegmentID;
                segmentsBoundaries.push_back(newSegment);
                touchingSegments.insert({ thisSegmentID, {} });
            } else {
                // Segment exists, update boundaries

                const auto thisSegmentID = accumulator;
                auto& thisSegment = segmentsBoundaries.at(thisSegmentID - 1);

                segmentsIDs.at<double>(y, x) = thisSegmentID;
                thisSegment.updateBoundaries(x, y);
            }
        }
    );

    // Merge touching segments
    std::vector<std::reference_wrapper<Segment>> segmentsRefs;

    for (auto& segment: segmentsBoundaries) {
        segmentsRefs.push_back(std::ref(segment));
    }

    for (uint64_t segmentIdx = 0; segmentIdx < segmentsRefs.size(); segmentIdx++) {
        const auto& segmentRef = segmentsRefs.at(segmentIdx);
        auto& segment = segmentRef.get();
        const auto& touching = touchingSegments.at(segmentIdx + 1);

        for (const auto& touchesIdx: touching) {
            const auto& touched = segmentsRefs.at(touchesIdx - 1);

            segment.merge(touched);
            segmentsRefs.at(touchesIdx - 1) = segmentRef;
        }
    }

    std::unordered_set<Segment*> uniqSegments;
    for (const auto& segment: segmentsRefs) {
        Segment* segmentPtr = &(segment.get());

        uniqSegments.insert(segmentPtr);
    }

    std::vector<Segment> segments;
    for (const auto& segmentPtr: uniqSegments) {
        segments.push_back((*segmentPtr));
    }

    return segments;
}

std::vector<Segment>
ImgProcessor::getImageSegmentsFloodFill(const cv::Mat& img)
const
{
    cv::Mat_<cv::Vec3i> segmentedImg = img.clone();

    int currentSegmentID = 1;

    matrixOps::forEachPixel(
        img,
        [&](const uint64_t& x, const uint64_t& y) -> void
        {
            if (segmentedImg(y, x)[0] != consts::colors::white) {
                return;
            }

            std::stack<std::pair<int, int>> neighbours;

            neighbours.push({ x, y });

            // FloodFill
            while (!neighbours.empty()) {
                int neighbourX = neighbours.top().first;
                int neighbourY = neighbours.top().second;

                neighbours.pop();

                segmentedImg(neighbourY,neighbourX)[0] = currentSegmentID;

                for (int adjacentY = -1; adjacentY <= 1; ++adjacentY) {
                    for (int adjacentX = -1; adjacentX <= 1; ++adjacentX) {
                        if (adjacentY == 0 && adjacentX == 0) {
                            continue;
                        }
                        if (neighbourY + adjacentY < 0 || neighbourY + adjacentY >= img.rows) {
                            continue;
                        }
                        if (neighbourX + adjacentX < 0 || neighbourX + adjacentX >= img.cols) {
                            continue;
                        }
                        if (segmentedImg(neighbourY + adjacentY, neighbourX + adjacentX)[0] != consts::colors::white) {
                            continue;
                        }

                        neighbours.push({
                            neighbourX + adjacentX,
                            neighbourY + adjacentY
                        });
                    }
                }
            }

            // Segmentation pixels can still hold WHITE (255) or BLACK (0) values
            // make sure we do not use those
            if ((currentSegmentID + 2) % 256 == 0) {
                currentSegmentID += 3;
            } else {
                ++currentSegmentID;
            }
        }
    );

    std::unordered_map<int, Segment> segmentsMap;

    matrixOps::forEachPixel(
        img,
        [&](const uint64_t& x, const uint64_t& y) -> void
        {
            if (x == 0 || x == img.cols - 1 || y == 0 || y == img.rows - 1) {
                return;
            }

            const auto& thisSegmentID = segmentedImg(y, x)[0];

            if (thisSegmentID == consts::colors::black) {
                return;
            }

            if (segmentsMap.count(thisSegmentID) == 0) {
                Segment seg;

                seg.xMin = x;
                seg.xMax = x;
                seg.yMin = y;
                seg.yMax = y;

                segmentsMap.insert({ thisSegmentID, seg });
            }

            segmentsMap.at(thisSegmentID).updateBoundaries(x, y);
        }
    );

    for (auto& segment: segmentsMap) {
        segment.second.updatePixels(segmentedImg, segment.first);
    }

    std::vector<Segment> segments;

    for (auto& segment: segmentsMap) {
        segments.push_back(segment.second);
    }

    return segments;
}

std::vector<structs::Candidate>
ImgProcessor::findImageLogoCandidates(
    const cv::Mat& img,
    const std::vector<structs::Segment>& segments
)
const
{
    std::vector<structs::Candidate> candidates;

    for (const auto& segment: segments) {
        if (segment.getArea() < 225) {
            continue;
        }

        auto candidate = structs::Candidate(segment, 0);

        candidates.push_back(candidate);
    }

    return candidates;
}

cv::Mat
ImgProcessor::drawSegmentsBBoxes(
    const cv::Mat& img,
    const std::vector<pobr::imgProcessing::structs::Segment>& segments,
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
