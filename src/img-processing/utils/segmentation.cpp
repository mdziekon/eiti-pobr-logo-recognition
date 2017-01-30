#include "./segmentation.hpp"

#include <stack>
#include <unordered_map>
#include <unordered_set>

#include "../../utils/consts.hpp"
#include "./matrix-ops.hpp"

namespace consts = pobr::utils::consts;
namespace matrixOps = pobr::imgProcessing::utils::matrixOps;

namespace segmentation = pobr::imgProcessing::utils::segmentation;

std::vector<structs::Segment>
segmentation::getImageSegmentsScanMerge(const cv::Mat& img, const bool& useDiagonalDetection)
{
    auto segmentsIDs = cv::Mat(
        img.rows,
        img.cols,
        CV_64F,
        0.0
    );
    std::vector<structs::Segment> segmentsBoundaries;
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
                structs::Segment newSegment;

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
    std::vector<std::reference_wrapper<structs::Segment>> segmentsRefs;

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

    std::unordered_set<structs::Segment*> uniqSegments;
    for (const auto& segment: segmentsRefs) {
        structs::Segment* segmentPtr = &(segment.get());

        uniqSegments.insert(segmentPtr);
    }

    std::vector<structs::Segment> segments;
    for (const auto& segmentPtr: uniqSegments) {
        segments.push_back((*segmentPtr));
    }

    return segments;
}

std::vector<structs::Segment>
segmentation::getImageSegmentsFloodFill(const cv::Mat& img, const bool& diagDetection)
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
                        if (!diagDetection) {
                            if (adjacentY == -1 && adjacentX == -1) {
                                continue;
                            }
                            if (adjacentY == 1 && adjacentX == -1) {
                                continue;
                            }
                            if (adjacentY == -1 && adjacentX == 1) {
                                continue;
                            }
                            if (adjacentY == 1 && adjacentX == 1) {
                                continue;
                            }
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

    std::unordered_map<int, structs::Segment> segmentsMap;

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
                structs::Segment seg;

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

    std::vector<structs::Segment> segments;

    for (auto& segment: segmentsMap) {
        segments.push_back(segment.second);
    }

    return segments;
}
