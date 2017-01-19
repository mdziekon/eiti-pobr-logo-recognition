#include "TerminalPrinter.hpp"

#include <sstream>
#include <iostream>

using TerminalPrinter = pobr::utils::TerminalPrinter;

const std::map<std::string, const unsigned int>&
TerminalPrinter::getColors()
{
    static const std::map<std::string, const unsigned int> colors = {
        { "red", 31 },
        { "yellow", 33 },
        { "blue", 34 },
        { "cyan", 36 },
        { "magenta", 35 }
    };

    return colors;
}

const std::string
TerminalPrinter::colorize(const std::string& message, const std::string& color)
{
    if (POBR_CONFIG_TERMINALCOLORS && TerminalPrinter::getColors().count(color) == 1)
    {
        std::stringstream temp;

        temp << "\e[" << TerminalPrinter::getColors().at(color) << "m"
             << message
             << "\e[0m";

        return temp.str();
    }

    return message;
}

void
TerminalPrinter::printLabel(const std::string& message, const std::string& color)
{
    std::stringstream temp;

    temp << "[" << message << "]";

    std::cout << TerminalPrinter::colorize(temp.str(), color)
              << " ";
}

const unsigned int
TerminalPrinter::getLabelLength(const std::string& message)
{
    return message.length() + 3;
}
