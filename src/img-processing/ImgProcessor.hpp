#ifndef POBR_IMGPROCESSING_IMGPROCESSOR_HPP
#define POBR_IMGPROCESSING_IMGPROCESSOR_HPP

#include <string>
#include <opencv2/core/core.hpp>

namespace pobr::imgProcessing
{
    class ImgProcessor
    {
    protected:
        cv::Mat img;

        const bool isReady();
        const void assertIsReady();
    public:
        const void loadImg(const std::string& imgPath);
        const void process();
    };
}

#endif
