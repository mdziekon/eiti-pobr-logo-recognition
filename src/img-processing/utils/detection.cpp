#include "./detection.hpp"

namespace detection = pobr::imgProcessing::utils::detection;

std::vector<structs::Candidate>
detection::findImageLogoCandidates(
    const std::vector<structs::Segment>& segments
)
{
    std::vector<structs::Candidate> candidates;

    for (const auto& segment: segments) {
        auto candidate = structs::Candidate(segment, 0);

        candidates.push_back(candidate);
    }

    return candidates;
}
