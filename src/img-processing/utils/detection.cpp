#include "./detection.hpp"

#include <algorithm>

namespace detection = pobr::imgProcessing::utils::detection;

std::vector<structs::Segment> 
detection::groupLetters(
    const std::vector<structs::Segment>& segments
)
{
    std::vector<structs::Segment> boundingBoxes;

    std::vector<structs::Segment> lettersT;
    std::vector<structs::Segment> lettersE;
    std::vector<structs::Segment> lettersS;
    std::vector<structs::Segment> lettersC;
    std::vector<structs::Segment> lettersO;

    for (const auto& segment: segments) {
        const auto classification = segment.classify();

        if (classification == "LETTER_T") {
            lettersT.push_back(segment);
            continue;
        }
        if (classification == "LETTER_E") {
            lettersE.push_back(segment);
            continue;
        }
        if (classification == "LETTER_S") {
            lettersS.push_back(segment);
            continue;
        }
        if (classification == "LETTER_C") {
            lettersC.push_back(segment);
            continue;
        }
        if (classification == "LETTER_O") {
            lettersO.push_back(segment);
            continue;
        }
    }

    for (const auto& segmentT: lettersT) {
        for (const auto& segmentO: lettersO) {
            // Do not detect mirrored images
            const auto tGlobalCenter = segmentT.getGlobalCenter();
            const auto oGlobalCenter = segmentO.getGlobalCenter();

            if (tGlobalCenter.first - oGlobalCenter.first > 0) {
                continue;
            }

            const auto distance = structs::Segment::getDistance(segmentT, segmentO);
            const auto avgWidth = ((double) (segmentT.getWidth() + segmentO.getWidth())) / 2;
            const auto expectedDistance = avgWidth * 4;

            const auto ratio = distance / expectedDistance;

            // If distance seems too long or too short, skip it
            if (ratio < 0.8 || ratio > 1.2) {
                continue;
            }

            // Interpolate possible places for next letters
            structs::Segment segmentE;
            structs::Segment segmentS;
            structs::Segment segmentC;

            const auto xDiff = ((double) (oGlobalCenter.first - tGlobalCenter.first)) / 4;
            const auto yDiff = ((double) (oGlobalCenter.second - tGlobalCenter.second)) / 4;

            // - Find letter E
            {
                const auto expectedCenterXMin = tGlobalCenter.first + (xDiff * 0) + (xDiff * -1.5); 
                const auto expectedCenterXMax = tGlobalCenter.first + (xDiff * 0) + (xDiff * 1.5); 
                const auto expectedCenterYMin = tGlobalCenter.second + (yDiff * 0) + ((yDiff > 0 ? yDiff + 1 : yDiff - 1) * (yDiff > 0 ? -1.5 : 1.5)); 
                const auto expectedCenterYMax = tGlobalCenter.second + (yDiff * 0) + ((yDiff > 0 ? yDiff + 1 : yDiff - 1) * (yDiff > 0 ? 1.5 : -1.5)); 

                for (const auto& thisSegment: lettersE) {
                    const auto thisGlobalCenter = thisSegment.getGlobalCenter();

                    if (thisGlobalCenter.first < expectedCenterXMin) {
                        continue;
                    }
                    if (thisGlobalCenter.first > expectedCenterXMax) {
                        continue;
                    }
                    if (thisGlobalCenter.second < expectedCenterYMin) {
                        continue;
                    }
                    if (thisGlobalCenter.second > expectedCenterYMax) {
                        continue;
                    }

                    segmentE = thisSegment;

                    break;
                }

                if (!segmentE.isValid()) {
                    continue;
                }
            }

            // - Find letter S
            {
                const auto expectedCenterXMin = tGlobalCenter.first + (xDiff * 1) + (xDiff * -1.5); 
                const auto expectedCenterXMax = tGlobalCenter.first + (xDiff * 1) + (xDiff * 1.5); 
                const auto expectedCenterYMin = tGlobalCenter.second + (yDiff * 1) + ((yDiff > 0 ? yDiff + 1 : yDiff - 1) * (yDiff > 0 ? -1.5 : 1.5)); 
                const auto expectedCenterYMax = tGlobalCenter.second + (yDiff * 1) + ((yDiff > 0 ? yDiff + 1 : yDiff - 1) * (yDiff > 0 ? 1.5 : -1.5)); 

                for (const auto& thisSegment: lettersS) {
                    const auto thisGlobalCenter = thisSegment.getGlobalCenter();

                    if (thisGlobalCenter.first < expectedCenterXMin) {
                        continue;
                    }
                    if (thisGlobalCenter.first > expectedCenterXMax) {
                        continue;
                    }
                    if (thisGlobalCenter.second < expectedCenterYMin) {
                        continue;
                    }
                    if (thisGlobalCenter.second > expectedCenterYMax) {
                        continue;
                    }

                    segmentS = thisSegment;

                    break;
                }

                if (!segmentS.isValid()) {
                    continue;
                }
            }

            // - Find letter C
            {
                const auto expectedCenterXMin = tGlobalCenter.first + (xDiff * 2) + (xDiff * -1.5); 
                const auto expectedCenterXMax = tGlobalCenter.first + (xDiff * 2) + (xDiff * 1.5); 
                const auto expectedCenterYMin = tGlobalCenter.second + (yDiff * 2) + ((yDiff > 0 ? yDiff + 1 : yDiff - 1) * (yDiff > 0 ? -1.5 : 1.5)); 
                const auto expectedCenterYMax = tGlobalCenter.second + (yDiff * 2) + ((yDiff > 0 ? yDiff + 1 : yDiff - 1) * (yDiff > 0 ? 1.5 : -1.5)); 

                for (const auto& thisSegment: lettersC) {
                    const auto thisGlobalCenter = thisSegment.getGlobalCenter();

                    if (thisGlobalCenter.first < expectedCenterXMin) {
                        continue;
                    }
                    if (thisGlobalCenter.first > expectedCenterXMax) {
                        continue;
                    }
                    if (thisGlobalCenter.second < expectedCenterYMin) {
                        continue;
                    }
                    if (thisGlobalCenter.second > expectedCenterYMax) {
                        continue;
                    }

                    segmentC = thisSegment;

                    break;
                }

                if (!segmentC.isValid()) {
                    continue;
                }
            }

            // Found all letters, store bounding box and skip to next T letter
            structs::Segment bbox;

            bbox.xMin = segmentT.xMin;
            bbox.xMax = segmentO.xMax;

            if (tGlobalCenter.second < oGlobalCenter.second) {
                // Top to bottom
                bbox.yMin = segmentT.yMin;
                bbox.yMax = segmentO.yMax;
            } else {
                // Bottom to top
                bbox.yMin = segmentO.yMin;
                bbox.yMax = segmentT.yMax;
            }

            // Note: possible slight optimisation
            //       remove used letters from containers to prevent re-checking them

            boundingBoxes.push_back(bbox);
        }
    }

    return boundingBoxes;
}
