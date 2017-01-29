#include "Logger.hpp"

#include <sstream>
#include <iostream>

using Logger = pobr::utils::Logger;
using Exception = pobr::utils::Logger::Exception;

// Exception class
Exception::Exception(const std::string& error) noexcept:
error(error)
{}

const char* Exception::what() const noexcept
{
    return this->error.c_str();
}


// Logger class
void Logger::debugFatal(const std::string& message)
{
    Logger::printLabel("Fatal", "magenta");
    Logger::print(Logger::getLabelLength("Fatal"), message);

    throw Logger::Exception(message);
}

void Logger::error(const std::string& message, const bool& noThrow)
{
    Logger::printLabel("Error", "red");
    Logger::print(Logger::getLabelLength("Error"), message);

    if (noThrow)
    {
        return;
    }

    throw Logger::Exception(message);
}

void Logger::warning(const std::string& message)
{
    Logger::printLabel("Warn", "yellow");
    Logger::print(Logger::getLabelLength("Warn"), message);
}

void Logger::notice(const std::string& message)
{
    Logger::printLabel("Note", "cyan");
    Logger::print(Logger::getLabelLength("Note"), message);
}

void Logger::print(const unsigned int& labelShift, const std::string& message)
{
    std::stringstream messageStream;
    messageStream << message;

    std::string line;

    std::getline(messageStream, line);
    std::cout << line << std::endl;

    while (std::getline(messageStream, line))
    {
        std::cout << std::string(labelShift, ' ') << line << std::endl;
    }
}
