#ifndef POBR_MODULES_TERMINALPRINTER_HPP
#define POBR_MODULES_TERMINALPRINTER_HPP

#ifndef POBR_CONFIG_TERMINALCOLORS
#define POBR_CONFIG_TERMINALCOLORS true
#endif

#include <string>
#include <map>

namespace pobr::modules
{
    class TerminalPrinter
    {
    protected:
        static const std::map<std::string, const unsigned int>& getColors();

        static const std::string colorize(const std::string& message, const std::string& color);
        static void printLabel(const std::string& message, const std::string& color = "");
        static const unsigned int getLabelLength(const std::string& message);
    };
}

#endif
