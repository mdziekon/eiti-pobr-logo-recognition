#include "App.hpp"

#include "../utils/error-handler/ErrorHandler.hpp"
#include "../img-processing/ImgProcessor.hpp"

using ErrorHandler = pobr::utils::ErrorHandler;
using ImgProcessor = pobr::imgProcessing::ImgProcessor;

using App = pobr::main::App;

App::App(const std::vector<std::string>& arguments)
{
    try
    {
        if (arguments.size() < 1)
        {
            ErrorHandler::error("No input file specified");
        }

        auto imgProcessor = ImgProcessor();

        imgProcessor.loadImg(arguments.at(0));
        imgProcessor.process();

    }
    catch(ErrorHandler::Exception &e)
    {
        ErrorHandler::error("Terminating...", true);
    }
}
