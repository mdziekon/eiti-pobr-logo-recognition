#ifndef POBR_IMGPROCESSING_STRUCTS_CANDIDATE_HPP
#define POBR_IMGPROCESSING_STRUCTS_CANDIDATE_HPP

#include <cstdint>

#include "./Segment.hpp"

using Segment = pobr::imgProcessing::structs::Segment;

namespace pobr::imgProcessing::structs
{
    struct Candidate
    {
    public:
        const Segment segment;
        const char letter = 0;

        Candidate(const Segment& segment, const char& letter = 0):
        segment(segment), letter(letter)
        {}

        const bool isValid() const
        {
            return letter != 0;
        }
    };
}

#endif
