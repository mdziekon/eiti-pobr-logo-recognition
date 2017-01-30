#include "App.hpp"

#include <opencv2/highgui/highgui.hpp>

#include "../utils/logger/Logger.hpp"
#include "../img-processing/ImgProcessor.hpp"

using Logger = pobr::utils::Logger;
using ImgProcessor = pobr::imgProcessing::ImgProcessor;

using App = pobr::main::App;

App::App(const std::vector<std::string>& arguments)
{
    try
    {
        if (arguments.size() < 1)
        {
            Logger::error("No input file specified");
        }

        auto imgProcessor = ImgProcessor();

        imgProcessor.loadImg(arguments.at(0));

        auto letterSegments = imgProcessor.process();
        auto img = imgProcessor.drawSegmentsBBoxes(
            imgProcessor.getImg(),
            letterSegments,
            { 0, 0, 0 },
            3
        );

        cv::imshow(arguments.at(0), img);

        cv::waitKey(-1);

    }
    catch(Logger::Exception &e)
    {
        Logger::error("Terminating...", true);
    }
}
