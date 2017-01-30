#include "Segment.hpp"

#include <cmath>

using Segment = pobr::imgProcessing::structs::Segment;

const double
Segment::getDistance(const Segment& left, const Segment& right)
{
    const auto leftCenterPos = left.getGlobalCenter();
    const auto rightCenterPos = right.getGlobalCenter();

    return std::sqrt(
        std::pow(leftCenterPos.first - rightCenterPos.first, 2) +
        std::pow(leftCenterPos.second - rightCenterPos.second, 2)
    );
}

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
            auto& thisSegmentID = segmentedImg(this->yMin + y, this->xMin + x)[0];

            this->pixels(y, x)[1] = consts::colors::black;
            this->pixels(y, x)[2] = consts::colors::black;

            if (thisSegmentID != segmentID) {
                this->pixels(y, x)[0] = consts::colors::black;
            } else {
                this->pixels(y, x)[0] = consts::colors::white;
            }
        }
    }
}

const void
Segment::merge(const Segment& other)
{
    this->updateBoundaries(other.xMin, other.yMin);
    this->updateBoundaries(other.xMax, other.yMax);
}

const bool
Segment::isValid()
const
{
    return !(this->xMin == 0 && this->xMax == 0);
}


const uint64_t
Segment::getWidth()
const
{
    return this->xMax - this->xMin + 1;
}

const uint64_t
Segment::getHeight()
const
{
    return this->yMax - this->yMin + 1;
}

const std::pair<double, double>
Segment::getLocalCenter()
const
{
    const double xMiddle = ((double) this->getWidth()) / 2;
    const double yMiddle = ((double) this->getHeight()) / 2;

    return { xMiddle, yMiddle };
}

const std::pair<double, double>
Segment::getGlobalCenter()
const
{
    const auto localCenter = this->getLocalCenter();

    return {
        localCenter.first + this->xMin,
        localCenter.second + this->yMin
    };
}

const uint64_t
Segment::getBBoxArea()
const
{
    return (this->getWidth() * this->getHeight());
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
    switch (no) {
    case 1:
        return this->getHuMomentInvariantNo1();
    case 2:
        return this->getHuMomentInvariantNo2();
    case 3:
        return this->getHuMomentInvariantNo3();
    case 4:
        return this->getHuMomentInvariantNo4();
    case 5:
        return this->getHuMomentInvariantNo5();
    case 6:
        return this->getHuMomentInvariantNo6();
    case 7:
        return this->getHuMomentInvariantNo7();
    }

    return -1;
}

const std::string
Segment::classify()
const
{
    if (!this->isBigEnough()) {
        return "ERROR_TOOSMALL";
    }
    if (!this->isSmallEnough()) {
        return "ERROR_TOOBIG";
    }

    if (this->isLetterT()) {
        return "LETTER_T";
    }
    if (this->isLetterE()) {
        return "LETTER_E";
    }
    if (this->isLetterS()) {
        return "LETTER_S";
    }
    if (this->isLetterC()) {
        return "LETTER_C";
    }
    if (this->isLetterO()) {
        return "LETTER_O";
    }

    return "ERROR_UNKNOWN";
}

const bool
Segment::isSmallEnough()
const
{
    return (this->getArea() <= 2000);
}

const bool
Segment::isBigEnough()
const
{
    return (this->getArea() >= 60);
}

const bool
Segment::isClassifiedAsLetter()
const
{
    return this->classify().find("LETTER_") != std::string::npos;
}

const double
Segment::getHuMomentInvariantNo1()
const
{
    const auto m00 = this->getNormalMoment(0, 0);
    const auto m01 = this->getNormalMoment(0, 1);
    const auto m10 = this->getNormalMoment(1, 0);

    return (
        (
            getCentralMoment(2, 0, m00, m10, m01) +
            getCentralMoment(0, 2, m00, m10, m01)
        )
        /
        std::pow(m00, 2)
    );
}

const double
Segment::getHuMomentInvariantNo2()
const
{
    const auto m00 = this->getNormalMoment(0, 0);
    const auto m01 = this->getNormalMoment(0, 1);
    const auto m10 = this->getNormalMoment(1, 0);

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
}

const double
Segment::getHuMomentInvariantNo3()
const
{
    const auto m00 = this->getNormalMoment(0, 0);
    const auto m01 = this->getNormalMoment(0, 1);
    const auto m10 = this->getNormalMoment(1, 0);

    return (
        (
            std::pow(
                (
                    (1 * this->getCentralMoment(3, 0, m00, m10, m01)) -
                    (3 * this->getCentralMoment(1, 2, m00, m10, m01))
                ),
                2
            )
            +
            std::pow(
                (
                    (3 * this->getCentralMoment(2, 1, m00, m10, m01)) -
                    (1 * this->getCentralMoment(0, 3, m00, m10, m01))
                ),
                2
            )
        )
        /
        std::pow(m00, 5)
    );
}

const double
Segment::getHuMomentInvariantNo4()
const
{
    const auto m00 = this->getNormalMoment(0, 0);
    const auto m01 = this->getNormalMoment(0, 1);
    const auto m10 = this->getNormalMoment(1, 0);

    return (
        (
            std::pow(
                (
                    this->getCentralMoment(3, 0, m00, m10, m01) +
                    this->getCentralMoment(1, 2, m00, m10, m01)
                ),
                2
            )
            +
            std::pow(
                (
                    this->getCentralMoment(2, 1, m00, m10, m01) -
                    this->getCentralMoment(0, 3, m00, m10, m01)
                ),
                2
            )
        )
        /
        std::pow(m00, 5)
    );
}

const double
Segment::getHuMomentInvariantNo5()
const
{
    const auto m00 = this->getNormalMoment(0, 0);
    const auto m01 = this->getNormalMoment(0, 1);
    const auto m10 = this->getNormalMoment(1, 0);

    return (
        (
            (
                (
                    (1 * this->getCentralMoment(3, 0, m00, m10, m01)) -
                    (3 * this->getCentralMoment(1, 2, m00, m10, m01))
                )
                *
                (
                    this->getCentralMoment(3, 0, m00, m10, m01) +
                    this->getCentralMoment(1, 2, m00, m10, m01)
                )
                *
                (
                    (
                        1 *
                        std::pow(
                            (
                                this->getCentralMoment(3, 0, m00, m10, m01) +
                                this->getCentralMoment(1, 2, m00, m10, m01)
                            ),
                            2
                        )
                    )
                    -
                    (
                        3 *
                        std::pow(
                            (
                                this->getCentralMoment(2, 1, m00, m10, m01) +
                                this->getCentralMoment(0, 3, m00, m10, m01)
                            ),
                            2
                        )
                    )
                )
            )
            +
            (
                (
                    (3 * this->getCentralMoment(2, 1, m00, m10, m01)) -
                    (1 * this->getCentralMoment(0, 3, m00, m10, m01))
                )
                *
                (
                    this->getCentralMoment(2, 1, m00, m10, m01) +
                    this->getCentralMoment(0, 3, m00, m10, m01)
                )
                *
                (
                    (
                        3 *
                        std::pow(
                            (
                                this->getCentralMoment(3, 0, m00, m10, m01) +
                                this->getCentralMoment(1, 2, m00, m10, m01)
                            ),
                            2
                        )
                    )
                    -
                    (
                        1 *
                        std::pow(
                            (
                                this->getCentralMoment(2, 1, m00, m10, m01) +
                                this->getCentralMoment(0, 3, m00, m10, m01)
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
}

const double
Segment::getHuMomentInvariantNo6()
const
{
    const auto m00 = this->getNormalMoment(0, 0);
    const auto m01 = this->getNormalMoment(0, 1);
    const auto m10 = this->getNormalMoment(1, 0);

    return (
        (
            (
                (
                    this->getCentralMoment(2, 0, m00, m10, m01) -
                    this->getCentralMoment(0, 2, m00, m10, m01)
                )
                *
                (
                    std::pow(
                        (
                            this->getCentralMoment(3, 0, m00, m10, m01) +
                            this->getCentralMoment(1, 2, m00, m10, m01)
                        ),
                        2
                    )
                    -
                    std::pow(
                        (
                            this->getCentralMoment(2, 1, m00, m10, m01) +
                            this->getCentralMoment(0, 3, m00, m10, m01)
                        ),
                        2
                    )
                )
            )
            +
            (
                (
                    4 *
                    this->getCentralMoment(1, 1, m00, m10, m01)
                )
                *
                (
                    this->getCentralMoment(3, 0, m00, m10, m01) +
                    this->getCentralMoment(1, 2, m00, m10, m01)
                )
                *
                (
                    this->getCentralMoment(2, 1, m00, m10, m01) +
                    this->getCentralMoment(0, 3, m00, m10, m01)
                )
            )
        )
        /
        std::pow(m00, 7)
    );
}

const double
Segment::getHuMomentInvariantNo7()
const
{
    const auto m00 = this->getNormalMoment(0, 0);
    const auto m01 = this->getNormalMoment(0, 1);
    const auto m10 = this->getNormalMoment(1, 0);

    return (
        (
            (
                this->getCentralMoment(2, 0, m00, m10, m01) *
                this->getCentralMoment(0, 2, m00, m10, m01)
            )
            -
            std::pow(
                this->getCentralMoment(1, 1, m00, m10, m01),
                2
            )
        )
        /
        std::pow(m00, 4)
    );
}

const bool
Segment::isLetterT()
const
{
    const auto hu1 = this->getHuMomentInvariant(1);
    const auto hu2 = this->getHuMomentInvariant(2);
    const auto hu3 = this->getHuMomentInvariant(3);
    const auto hu4 = this->getHuMomentInvariant(4);
    const auto hu7 = this->getHuMomentInvariant(7);

    if (hu1 < 0.311719 * 0.75 || hu1 > 0.436134 * 1.25) {
        return false;
    }
    if (hu2 < 0.002169 * 0.75 || hu2 > 0.068402 * 1.25) {
        return false;
    }
    if (hu3 < 0.026323 * 0.75 || hu3 > 0.075005 * 1.25) {
        return false;
    }
    if (hu4 < 0.000072 * 0.75 || hu4 > 0.005648 * 1.25) {
        return false;
    }
    if (hu7 < 0.023207 * 0.75 || hu7 > 0.036627 * 1.25) {
        return false;
    }

    return true;
}

const bool
Segment::isLetterE()
const
{
    const auto hu1 = this->getHuMomentInvariant(1);
    const auto hu2 = this->getHuMomentInvariant(2);
    const auto hu3 = this->getHuMomentInvariant(3);
    const auto hu4 = this->getHuMomentInvariant(4);
    const auto hu7 = this->getHuMomentInvariant(7);

    if (hu1 < 0.28205 * 0.75 || hu1 > 0.414592 * 1.25) {
        return false;
    }
    if (hu2 < 0.003996 * 0.75 || hu2 > 0.074936 * 1.25) {
        return false;
    }
    if (hu3 < 0.000485 * 0.75 || hu3 > 0.004407 * 1.25) {
        return false;
    }
    if (hu4 < 0.000013 * 0.75 || hu4 > 0.000442 * 1.25) {
        return false;
    }
    if (hu7 < 0.018604 * 0.75 || hu7 > 0.025658 * 1.25) {
        return false;
    }

    return true;
}

const bool
Segment::isLetterS()
const
{
    const auto hu1 = this->getHuMomentInvariant(1);
    const auto hu2 = this->getHuMomentInvariant(2);
    const auto hu3 = this->getHuMomentInvariant(3);
    const auto hu4 = this->getHuMomentInvariant(4);
    const auto hu7 = this->getHuMomentInvariant(7);

    if (hu1 < 0.280296 * 0.75 || hu1 > 0.362073 * 1.25) {
        return false;
    }
    if (hu2 < 0.001189 * 0.75 || hu2 > 0.036808 * 1.25) {
        return false;
    }
    if (hu3 < 0.000033 * 0.75 || hu3 > 0.000296 * 1.25) {
        return false;
    }
    if (hu4 < 0.000002 * 0.75 || hu4 > 0.000033 * 1.25) {
        return false;
    }
    if (hu7 < 0.018977 * 0.75 || hu7 > 0.029561 * 1.25) {
        return false;
    }

    return true;
}

const bool
Segment::isLetterC()
const
{
    const auto hu1 = this->getHuMomentInvariant(1);
    const auto hu2 = this->getHuMomentInvariant(2);
    const auto hu3 = this->getHuMomentInvariant(3);
    const auto hu4 = this->getHuMomentInvariant(4);
    const auto hu7 = this->getHuMomentInvariant(7);

    if (hu1 < 0.352379 * 0.75 || hu1 > 0.47044 * 1.25) {
        return false;
    }
    if (hu2 < 0.000268 * 0.75 || hu2 > 0.066352 * 1.25) {
        return false;
    }
    if (hu3 < 0.005753 * 0.75 || hu3 > 0.027387 * 1.25) {
        return false;
    }
    if (hu4 < 0.000054 * 0.75 || hu4 > 0.005462 * 1.25) {
        return false;
    }
    if (hu7 < 0.030236 * 0.75 || hu7 > 0.048242 * 1.25) {
        return false;
    }

    return true;
}

const bool
Segment::isLetterO()
const
{
    const auto hu1 = this->getHuMomentInvariant(1);
    const auto hu2 = this->getHuMomentInvariant(2);
    const auto hu3 = this->getHuMomentInvariant(3);
    const auto hu4 = this->getHuMomentInvariant(4);
    const auto hu7 = this->getHuMomentInvariant(7);

    if (hu1 < 0.324674 * 0.75 || hu1 > 0.39543 * 1.25) {
        return false;
    }
    if (hu2 < 0.000849 * 0.75 || hu2 > 0.013153 * 1.25) {
        return false;
    }
    if (hu3 < 0.000003 * 0.75 || hu3 > 0.000035 * 1.25) {
        return false;
    }
    if (hu4 < 0.000001 * 0.75 || hu4 > 0.000005 * 1.25) {
        return false;
    }
    if (hu7 < 0.024815 * 0.75 || hu7 > 0.037956 * 1.25) {
        return false;
    }

    return true;
}
