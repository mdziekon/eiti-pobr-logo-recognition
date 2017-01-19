#include "ImgProcessor.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../utils/error-handler/ErrorHandler.hpp"

using ErrorHandler = pobr::utils::ErrorHandler;
using ImgProcessor = pobr::imgProcessing::ImgProcessor;

const bool
ImgProcessor::isReady()
{
    return !(this->img.empty());
}

const void
ImgProcessor::assertIsReady()
{
    if (this->isReady()) {
        return;
    }

    ErrorHandler::error("ImgProcessor has no image loaded yet!");
}

const void
ImgProcessor::loadImg(const std::string& imgPath)
{
    this->img = cv::imread(imgPath);

    if (this->img.empty()) {
        ErrorHandler::warning("Could not properly load image \"" + imgPath + "\"...");
    }
}

const void
ImgProcessor::process()
{
    this->assertIsReady();

    cv::imshow("test", this->img);

    cv::waitKey(-1);
}
