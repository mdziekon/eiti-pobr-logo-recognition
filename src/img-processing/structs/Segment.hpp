#ifndef POBR_IMGPROCESSING_STRUCTS_SEGMENT_HPP
#define POBR_IMGPROCESSING_STRUCTS_SEGMENT_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

#include "../../utils/consts.hpp"

namespace consts = pobr::utils::consts;

namespace pobr::imgProcessing::structs
{
    struct Segment
    {
    public:
        uint64_t xMin = 0;
        uint64_t xMax = 0;
        uint64_t yMin = 0;
        uint64_t yMax = 0;

        cv::Mat_<cv::Vec3b> pixels;

        const void updateBoundaries(const uint64_t& x, const uint64_t& y);
        const void updatePixels(const cv::Mat_<cv::Vec3i>& segmentedImg, const int& segmentID);

        const void merge(const Segment& other);

        const uint64_t getArea() const;
        const uint64_t getCircumference() const;
        const double getW3() const; // Malinowska
        const double getNormalMoment(const uint64_t& p, const uint64_t& q) const;
        const double getCentralMoment(const uint64_t& p, const uint64_t& q, const double& m00, const double& m10, const double& m01) const;
        const double getHuMomentInvariant(const uint8_t& no) const;

        const std::string classify() const;
        const bool isClassifiedAsLetter() const;
        const bool isLetterT() const;
        const bool isLetterE() const;
        const bool isLetterS() const;
        const bool isLetterC() const;
        const bool isLetterO() const;

    };
}

#endif
