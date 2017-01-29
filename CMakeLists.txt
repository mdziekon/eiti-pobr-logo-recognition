cmake_minimum_required(VERSION 3.6)
project(eiti_pobr_logo_recognition)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")

set(SOURCE_FILES
        src/img-processing/structs/Candidate.hpp
        src/img-processing/structs/Segment.cpp
        src/img-processing/structs/Segment.hpp
        src/img-processing/utils/matrix-ops.hpp
        src/img-processing/utils/matrix-ops.impl.hpp
        src/img-processing/ImgProcessor.cpp
        src/img-processing/ImgProcessor.hpp
        src/main/App.cpp
        src/main/App.hpp
        src/utils/logger/Logger.cpp
        src/utils/logger/Logger.hpp
        src/utils/performance-timer/PerformanceTimer.cpp
        src/utils/performance-timer/PerformanceTimer.hpp
        src/utils/terminal-printer/TerminalPrinter.cpp
        src/utils/terminal-printer/TerminalPrinter.hpp
        src/utils/consts.hpp
        src/main.cpp
        utilities/calculate-ranges.js
        LICENSE
        README.md
        Sconstruct
        )

add_executable(eiti_pobr_logo_recognition ${SOURCE_FILES})
target_link_libraries(eiti_pobr_logo_recognition ${OpenCV_LIBS})
