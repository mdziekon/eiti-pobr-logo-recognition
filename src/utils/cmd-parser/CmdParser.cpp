#include "./CmdParser.hpp"

#include <algorithm>

using CmdParser = pobr::utils::CmdParser;

CmdParser::CmdParser(const std::vector<std::string> args):
arguments(args)
{}

const bool
CmdParser::hasFlag(const std::string& flagName)
const
{
    auto fullFlag = "--" + flagName;

    auto iter = std::find_if(
        this->arguments.begin(),
        this->arguments.end(),
        [&fullFlag](const std::string& argument) -> bool
        {
            return argument.find(fullFlag) != std::string::npos;
        }
    );

    return iter != this->arguments.end();
}

const std::string
CmdParser::getFlagValue(const std::string& flagName)
const
{
    auto fullFlagEq = "--" + flagName + "=";

    auto iter = std::find_if(
        this->arguments.begin(),
        this->arguments.end(),
        [&fullFlagEq](const std::string& argument) -> bool
        {
            return argument.find(fullFlagEq) != std::string::npos;
        }
    );

    if (iter == this->arguments.end()) {
        return "";
    }

    auto valIter = (*iter).find(fullFlagEq) + fullFlagEq.length();

    return (*iter).substr(valIter);
}
