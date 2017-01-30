#ifndef POBR_UTILS_CMDPARSER_HPP
#define POBR_UTILS_CMDPARSER_HPP

#include <string>
#include <vector>

namespace pobr::utils
{
    class CmdParser
    {
    public:
        CmdParser() = delete;
        explicit CmdParser(const std::vector<std::string> args);

        const bool hasFlag(const std::string& flagName) const;
        const std::string getFlagValue(const std::string& flagName) const;

    protected:
        const std::vector<std::string> arguments;
    };
}

#endif
