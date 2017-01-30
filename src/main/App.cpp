#include "App.hpp"

#include <opencv2/highgui/highgui.hpp>

#include "../utils/logger/Logger.hpp"
#include "../utils/cmd-parser/CmdParser.hpp"
#include "../img-processing/ImgProcessor.hpp"

using Logger = pobr::utils::Logger;
using CmdParser = pobr::utils::CmdParser;
using ImgProcessor = pobr::imgProcessing::ImgProcessor;

using App = pobr::main::App;

App::App(const std::vector<std::string>& arguments)
{
    try
    {
        auto cmdParser = CmdParser(arguments);

        auto const filepath = cmdParser.getFlagValue("file");
        const bool showBinaryImg = cmdParser.hasFlag("binary");

        if (filepath.length() < 1)
        {
            Logger::error("No input file specified");
        }

        auto imgProcessor = ImgProcessor();

        imgProcessor.loadImg(filepath);

        auto letterSegments = imgProcessor.process();

        if (showBinaryImg) {
            auto img = imgProcessor.drawSegmentsBBoxes(
                imgProcessor.getBinarizedImg(),
                letterSegments,
                { 0, 0, 255 },
                3
            );

            cv::imshow(filepath, img);
        } else {
            auto img = imgProcessor.drawSegmentsBBoxes(
                imgProcessor.getImg(),
                letterSegments,
                { 0, 0, 0 },
                3
            );

            cv::imshow(filepath, img);
        }

        cv::waitKey(-1);

    }
    catch(Logger::Exception &e)
    {
        Logger::error("Terminating...", true);
    }
}
