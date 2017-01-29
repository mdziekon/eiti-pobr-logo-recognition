#ifndef POBR_MAIN_APP_HPP
#define POBR_MAIN_APP_HPP

#include <vector>
#include <string>

namespace pobr::main
{
    class App
    {
    public:
        App() = delete;
        explicit App(const std::vector<std::string>& arguments);
    };
}

#endif
