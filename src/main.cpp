#include <vector>
#include <string>

#include "./modules/error-handler/ErrorHandler.hpp"

using ErrorHandler = pobr::modules::ErrorHandler;

int main(int argc, char** argv)
{
    std::vector<std::string> arguments(argv + 1, argv + argc);

    // TODO: Implement application
    //       pass args from "arguments" variable

    auto errHandler = ErrorHandler();

    errHandler.error("Nothing implemented yet", false);

    return 0;
}
