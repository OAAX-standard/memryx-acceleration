// logging.h 
// Logger Utility to write necessary information to a file or to console

// logging.h

#ifndef LOGGING_H
#define LOGGING_H

#include <iostream>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <limits.h>

namespace MX {
namespace Utils {

enum class LogLevel {
    ERROR,
    WARNING,
    INFO,
    DEBUG
};

enum class LogDestination {
    FILE,
    CONSOLE,
    BOTH
};


class Logger {
public:
    Logger(const std::string& logFileName , LogLevel loggerLevel = LogLevel::INFO);
    Logger(LogDestination destination , LogLevel loggerLevel = LogLevel::INFO);
    Logger();
    ~Logger();

    void setLogDestination(LogDestination destination);
    void setLogLevel(LogLevel loggerLevel);
    LogDestination getLogDestination();
    LogLevel getCurrentLogLevel();
    void removeAllLogFiles();
    void log(const std::string& message, LogLevel level = LogLevel::INFO);

private:
    std::string logFileName;
    std::ofstream logFile;
    LogDestination outputDestination;
    LogLevel currentLogLevel;
    
    void initializeLogFile();
    bool createLogDirectory(const std::string& directoryPath);
    bool createLogFile();
    void logInitializationInfo();
    std::string getCurrentProcessName();
    std::string getCurrentTimeStamp();
    void logToConsole(const std::string& message, LogLevel level);
    void logToFile(const std::string& message, LogLevel level);
    void logMessage(const std::string& message, LogLevel level);
};

} // namespace Util
} // namespace MX

#endif // LOGGING_H
