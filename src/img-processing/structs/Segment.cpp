#include "Segment.hpp"

#include <cmath>

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

const uint64_t
Segment::getArea()
const
{
    uint64_t area = 0;

    for (int x = 0; x < this->pixels.cols; ++x) {
        for (int y = 0; y < this->pixels.rows; ++y) {
            auto& point = this->pixels(y, x);

            if (point[0] != consts::colors::white) {
                continue;
            }

            area++;
        }
    }

    return area;
}

const uint64_t
Segment::getCircumference()
const
{
    const auto& pixels = this->pixels;
    uint64_t circumference = 0;

    for (int x = 0; x < pixels.cols; ++x) {
        for (int y = 0; y < pixels.rows; ++y) {
            auto& point = pixels(y, x);

            if (point[0] != consts::colors::white) {
                continue;
            }

            if (x == 0 || x == this->pixels.cols - 1) {
                circumference++;
                continue;
            }
            if (y == 0 || y == this->pixels.rows - 1) {
                circumference++;
                continue;
            }

            if (
                pixels(y + 1, x)[0] != consts::colors::white ||
                pixels(y - 1, x)[0] != consts::colors::white ||
                pixels(y, x + 1)[0] != consts::colors::white ||
                pixels(y, x - 1)[0] != consts::colors::white
            ) {
                circumference++;
                continue;
            }
        }
    }

    return circumference;
}

const double
Segment::getW3()
const
{
    const double circumference = ((double) this->getCircumference());
    const double area = ((double) this->getArea());

    return (circumference / (2 * std::sqrt(area * 3.14))) - 1;
}

const double
Segment::getNormalMoment(const uint64_t& p, const uint64_t& q)
const
{
    double value = 0;

    for (int x = 0; x < this->pixels.cols; ++x) {
        for (int y = 0; y < this->pixels.rows; ++y) {
            auto& point = this->pixels.at<cv::Vec3b>(y, x);

            if (point[0] != consts::colors::white) {
                continue;
            }

            value += (std::pow(y, p) * std::pow(x, q));
        }
    }

    return value;
}

const double
Segment::getCentralMoment(const uint64_t& p, const uint64_t& q, const double& m00, const double& m10, const double& m01)
const
{
    double value = 0;

    const auto xTilde = (m01 / m00);
    const auto yTilde = (m10 / m00);

    for (int x = 0; x < this->pixels.cols; ++x) {
        for (int y = 0; y < this->pixels.rows; ++y) {
            auto& point = this->pixels.at<cv::Vec3b>(y, x);

            if (point[0] != consts::colors::white) {
                continue;
            }

            value += (std::pow(y - yTilde, p) * std::pow(x - xTilde, q));
        }
    }

    return value;
}

const double
Segment::getHuMomentInvariant(const uint8_t& no)
const
{
    const auto m00 = this->getNormalMoment(0, 0);
    const auto m01 = this->getNormalMoment(0, 1);
    const auto m10 = this->getNormalMoment(1, 0);

    switch (no) {
    case 1:
        return (
            (
                getCentralMoment(2, 0, m00, m10, m01) +
                getCentralMoment(0, 2, m00, m10, m01)
            )
            /
            std::pow(m00, 2)
        );
        break;
    case 2:
        return (
            (
                std::pow(
                    (
                        getCentralMoment(2, 0, m00, m10, m01) -
                        getCentralMoment(0, 2, m00, m10, m01)
                    ),
                    2
                )
                +
                (
                    4 *
                    std::pow(
                        getCentralMoment(1, 1, m00, m10, m01),
                        2
                    )
                )
            )
            /
            std::pow(m00, 4)
        );
        break;
    case 3:
        return (
            (
                std::pow(
                    (
                        (1 * getCentralMoment(3, 0, m00, m10, m01)) -
                        (3 * getCentralMoment(1, 2, m00, m10, m01))
                    ),
                    2
                )
                +
                std::pow(
                    (
                        (3 * getCentralMoment(2, 1, m00, m10, m01)) -
                        (1 * getCentralMoment(0, 3, m00, m10, m01))
                    ),
                    2
                )
            )
            /
            std::pow(m00, 5)
        );
        break;
    case 4:
        return (
            (
                std::pow(
                    (
                        getCentralMoment(3, 0, m00, m10, m01) +
                        getCentralMoment(1, 2, m00, m10, m01)
                    ),
                    2
                )
                +
                std::pow(
                    (
                        getCentralMoment(2, 1, m00, m10, m01) -
                        getCentralMoment(0, 3, m00, m10, m01)
                    ),
                    2
                )
            )
            /
            std::pow(m00, 5)
        );
        break;
    case 5:
        return (
            (
                (
                    (
                        (1 * getCentralMoment(3, 0, m00, m10, m01)) -
                        (3 * getCentralMoment(1, 2, m00, m10, m01))
                    )
                    *
                    (
                        getCentralMoment(3, 0, m00, m10, m01) +
                        getCentralMoment(1, 2, m00, m10, m01)
                    )
                    *
                    (
                        (
                            1 *
                            std::pow(
                                (
                                    getCentralMoment(3, 0, m00, m10, m01) +
                                    getCentralMoment(1, 2, m00, m10, m01)
                                ),
                                2
                            )
                        )
                        -
                        (
                            3 *
                            std::pow(
                                (
                                    getCentralMoment(2, 1, m00, m10, m01) +
                                    getCentralMoment(0, 3, m00, m10, m01)
                                ),
                                2
                            )
                        )
                    )
                )
                +
                (
                    (
                        (3 * getCentralMoment(2, 1, m00, m10, m01)) -
                        (1 * getCentralMoment(0, 3, m00, m10, m01))
                    )
                    *
                    (
                        getCentralMoment(2, 1, m00, m10, m01) +
                        getCentralMoment(0, 3, m00, m10, m01)
                    )
                    *
                    (
                        (
                            3 *
                            std::pow(
                                (
                                    getCentralMoment(3, 0, m00, m10, m01) +
                                    getCentralMoment(1, 2, m00, m10, m01)
                                ),
                                2
                            )
                        )
                        -
                        (
                            1 *
                            std::pow(
                                (
                                    getCentralMoment(2, 1, m00, m10, m01) +
                                    getCentralMoment(0, 3, m00, m10, m01)
                                ),
                                2
                            )
                        )
                    )
                )
            )
            /
            std::pow(m00, 10)
        );
        break;
    case 6:
        return (
            (
                (
                    (
                        getCentralMoment(2, 0, m00, m10, m01) -
                        getCentralMoment(0, 2, m00, m10, m01)
                    )
                    *
                    (
                        std::pow(
                            (
                                getCentralMoment(3, 0, m00, m10, m01) +
                                getCentralMoment(1, 2, m00, m10, m01)
                            ),
                            2
                        )
                        -
                        std::pow(
                            (
                                getCentralMoment(2, 1, m00, m10, m01) +
                                getCentralMoment(0, 3, m00, m10, m01)
                            ),
                            2
                        )
                    )
                )
                +
                (
                    (
                        4 *
                        getCentralMoment(1, 1, m00, m10, m01)
                    )
                    *
                    (
                        getCentralMoment(3, 0, m00, m10, m01) +
                        getCentralMoment(1, 2, m00, m10, m01)
                    )
                    *
                    (
                        getCentralMoment(2, 1, m00, m10, m01) +
                        getCentralMoment(0, 3, m00, m10, m01)
                    )
                )
            )
            /
            std::pow(m00, 7)
        );
        break;
    case 7:
        return (
            (
                (
                    getCentralMoment(2, 0, m00, m10, m01) *
                    getCentralMoment(0, 2, m00, m10, m01)
                )
                -
                std::pow(
                    getCentralMoment(1, 1, m00, m10, m01),
                    2
                )
            )
            /
            std::pow(m00, 4)
        );
        break;
    }

    return -1;
}
