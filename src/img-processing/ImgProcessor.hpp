#ifndef POBR_IMGPROCESSING_IMGPROCESSOR_HPP
#define POBR_IMGPROCESSING_IMGPROCESSOR_HPP

#include <string>
#include <vector>
#include <functional>
#include <opencv2/core/core.hpp>

#include "./structs/Segment.hpp"
#include "./structs/Candidate.hpp"

namespace structs = pobr::imgProcessing::structs;

namespace pobr::imgProcessing
{
    class ImgProcessor
    {
    public:
        const void loadImg(const std::string& imgPath);
        const void process() const;

        cv::Mat detectEdges(const cv::Mat& img) const;

        std::vector<structs::Segment> getImageSegmentsScanMerge(const cv::Mat& img, const bool& useDiagonalDetection = true) const;
        std::vector<structs::Segment> getImageSegmentsFloodFill(const cv::Mat& img) const;

        std::vector<structs::Candidate> findImageLogoCandidates(const cv::Mat& img, const std::vector<structs::Segment>& segments) const;

        cv::Mat drawSegmentsBBoxes(
            const cv::Mat& img,
            const std::vector<pobr::imgProcessing::structs::Segment>& segments,
            const cv::Vec3b& borderColor = { 0, 0, 255 }
        ) const;

    protected:
        cv::Mat img;

        const bool isReady() const;
        const void assertIsReady() const;

        cv::Mat processPreEnhance(const cv::Mat& img) const;
        cv::Mat processBinarize(const cv::Mat& img) const;
        cv::Mat processBinaryEnhance(const cv::Mat& img) const;
        std::vector<structs::Segment> processSegmentation(const cv::Mat& img) const;
        std::vector<structs::Candidate> processFilterCandidates(const cv::Mat& img, const std::vector<structs::Segment>& segments) const;
        std::vector<structs::Segment> processDetection(const std::vector<structs::Candidate>& candidates) const;
    };
}

#endif
