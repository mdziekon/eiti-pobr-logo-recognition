#ifndef POBR_IMGPROCESSING_IMGPROCESSOR_HPP
#define POBR_IMGPROCESSING_IMGPROCESSOR_HPP

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

#include "./structs/Segment.hpp"

namespace structs = pobr::imgProcessing::structs;

namespace pobr::imgProcessing
{
    class ImgProcessor
    {
    public:
        const void loadImg(const std::string& imgPath);
        const cv::Mat& getImg() const;
        const cv::Mat getBinarizedImg() const;
        const std::vector<structs::Segment> process() const;

        cv::Mat drawSegmentsBBoxes(
            const cv::Mat& img,
            const std::vector<structs::Segment>& segments,
            const cv::Vec3b& borderColor = { 0, 0, 0 },
            const unsigned int& borderSize = 1
        ) const;

    protected:
        cv::Mat img;

        const bool isReady() const;
        const void assertIsReady() const;

        cv::Mat processPreEnhance(
            const cv::Mat& img,
            const bool& isProfiling = false
        ) const;
        cv::Mat processBinarize(
            const cv::Mat& img,
            const bool& isProfiling = false
        ) const;
        cv::Mat processBinaryEnhance(
            const cv::Mat& img,
            const bool& isProfiling = false
        ) const;
        std::vector<structs::Segment> processSegmentation(
            const cv::Mat& img,
            const bool& isProfiling = false
        ) const;
        std::vector<structs::Segment> processFilterCandidates(
            const std::vector<structs::Segment>& segments,
            const bool& isProfiling = false
        ) const;
        std::vector<structs::Segment> processDetection(
            const std::vector<structs::Segment>& segments,
            const bool& isProfiling = false
        ) const;
    };
}

#endif
