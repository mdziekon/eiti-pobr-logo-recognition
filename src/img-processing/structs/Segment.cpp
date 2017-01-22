#include "Segment.hpp"

using Segment = pobr::imgProcessing::structs::Segment;

const void
Segment::updateBoundaries(const uint64_t& x, const uint64_t& y)
{
    if (x < this->xMin) {
        this->xMin = x;
    } else if (x > this->xMax) {
        this->xMax = x;
    }

    if (y < this->yMin) {
        this->yMin = y;
    } else if (y > this->yMax) {
        this->yMax = y;
    }
}

const void
Segment::updatePixels(const cv::Mat_<cv::Vec3i>& segmentedImg, const int& segmentID)
{
    this->pixels = cv::Mat_<cv::Vec3b>(
        (this->yMax - this->yMin + 1),
        (this->xMax - this->xMin + 1)
    );

    for (int x = 0; x < (this->xMax - this->xMin + 1); x++) {
        for (int y = 0; y < (this->yMax - this->yMin + 1); y++) {
            if (segmentedImg.at<cv::Vec3b>(this->yMin + y, this->xMin + x)[0] != segmentID) {
                continue;
            }

            this->pixels(y, x)[0] = consts::colors::white;
        }
    }
}

const void
Segment::merge(const Segment& other)
{
    this->updateBoundaries(other.xMin, other.yMin);
    this->updateBoundaries(other.xMax, other.yMax);
}
