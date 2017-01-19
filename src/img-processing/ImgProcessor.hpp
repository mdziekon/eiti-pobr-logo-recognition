#ifndef POBR_IMGPROCESSING_IMGPROCESSOR_HPP
#define POBR_IMGPROCESSING_IMGPROCESSOR_HPP

#include <string>

namespace pobr::imgProcessing
{
    class ImgProcessor
    {
    protected:
        void* img = nullptr;
    public:
        const void loadImg(const std::string& imgPath);
        const void process();
    };
}

#endif
