#ifndef POBR_UTILS_LOGGER_HPP
#define POBR_UTILS_LOGGER_HPP

#ifndef POBR_CONFIG_SILENTERRORS
#define POBR_CONFIG_SILENTERRORS false
#endif

#include <string>
#include <map>
#include <exception>

#include "../terminal-printer/TerminalPrinter.hpp"

namespace pobr::utils
{
    class Logger: public TerminalPrinter
    {
    public:
        class Exception: public std::exception
        {
            friend class Logger;

        public:
            virtual const char* what() const noexcept;

        private:
            Exception(const std::string& error) noexcept;

            const std::string error;
        };

        static void debugFatal(const std::string& message);
        static void error(const std::string& message, const bool& noThrow = false);
        static void warning(const std::string& message);
        static void notice(const std::string& message);

        static void print(const unsigned int& labelShift, const std::string& message);
    };
}

#endif
