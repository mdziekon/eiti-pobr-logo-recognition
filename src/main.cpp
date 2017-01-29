#include <vector>
#include <string>

#include "./main/App.hpp"

using App = pobr::main::App;

int main(int argc, char** argv)
{
    std::vector<std::string> arguments(argv + 1, argv + argc);

    App myApp(arguments);

    return 0;
}
