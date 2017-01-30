#ifndef POBR_IMGPROCESSING_UTILS_DETECTION_HPP
#define POBR_IMGPROCESSING_UTILS_DETECTION_HPP

#include <vector>

#include "../structs/Segment.hpp"

namespace structs = pobr::imgProcessing::structs;

namespace pobr::imgProcessing::utils::detection
{
    std::vector<structs::Segment> groupLetters(
        const std::vector<structs::Segment>& segments
    );
}

#endif
