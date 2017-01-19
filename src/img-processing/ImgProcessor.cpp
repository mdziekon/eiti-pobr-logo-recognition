#include "ImgProcessor.hpp"

#include "../utils/error-handler/ErrorHandler.hpp"

using ErrorHandler = pobr::utils::ErrorHandler;
using ImgProcessor = pobr::imgProcessing::ImgProcessor;

const void
ImgProcessor::loadImg(const std::string& imgPath)
{

}

const void
ImgProcessor::process()
{
    ErrorHandler::error("Nothing loaded yet");
}
