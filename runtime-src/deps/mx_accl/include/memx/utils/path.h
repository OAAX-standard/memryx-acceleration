#ifndef PATH_UTILS_H
#define PATH_UTILS_H

#include <filesystem>
#include <string>
namespace MX
{
    namespace Utils
    {
        /**
         * Returns absolute path of home directory if env variable `MX_API_HOME`
         * is set else returns empty path
         */
        std::filesystem::path mx_get_home_dir();

        /**
         * Returns absolute path of accl directory
         */
        std::filesystem::path mx_get_accl_dir();

    } // namespace Utils
} // namespace MX

#endif