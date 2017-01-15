#include "ErrorHandler.hpp"

#include <sstream>
#include <iostream>

using ErrorHandler = pobr::modules::ErrorHandler;
using Exception = pobr::modules::ErrorHandler::Exception;

// Exception class
Exception::Exception(const std::string& error) noexcept:
error(error)
{}

const char* Exception::what() const noexcept
{
    return this->error.c_str();
}


// ErrorHandler class
void ErrorHandler::debugFatal(const std::string& message)
{
    ErrorHandler::printLabel("Fatal", "magenta");
    ErrorHandler::print(ErrorHandler::getLabelLength("Fatal"), message);

    throw ErrorHandler::Exception(message);
}

void ErrorHandler::error(const std::string& message, const bool& noThrow)
{
    ErrorHandler::printLabel("Error", "red");
    ErrorHandler::print(ErrorHandler::getLabelLength("Error"), message);

    if (noThrow)
    {
        return;
    }

    throw ErrorHandler::Exception(message);
}

void ErrorHandler::warning(const std::string& message)
{
    ErrorHandler::printLabel("Warn", "yellow");
    ErrorHandler::print(ErrorHandler::getLabelLength("Warn"), message);
}

void ErrorHandler::notice(const std::string& message)
{
    ErrorHandler::printLabel("Note", "cyan");
    ErrorHandler::print(ErrorHandler::getLabelLength("Note"), message);
}

void ErrorHandler::print(const unsigned int& labelShift, const std::string& message)
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
