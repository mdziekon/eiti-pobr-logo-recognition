#ifndef POBR_IMGPROCESSING_UTILS_DETECTION_HPP
#define POBR_IMGPROCESSING_UTILS_DETECTION_HPP

#include <vector>

#include "../structs/Segment.hpp"
#include "../structs/Candidate.hpp"

namespace structs = pobr::imgProcessing::structs;

namespace pobr::imgProcessing::utils::detection
{
    std::vector<structs::Candidate> findImageLogoCandidates(
        const std::vector<structs::Segment>& segments
    );
}

#endif
