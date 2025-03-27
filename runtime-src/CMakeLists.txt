cmake_minimum_required(VERSION 3.10.2)

project(RuntimeLibrary)

# Set the build type
# add_definitions(-DDEBUG)

# Locate src, include, and deps folders
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DEPS_DIR ${CMAKE_CURRENT_LIST_DIR}/deps)
set(MEMX_DEPS ${DEPS_DIR}/memx)
set(MX_ACCL_DEPS ${DEPS_DIR}/mx_accl)

# add string option to specify the target platform
set(PLATFORM "NONE" CACHE STRING "The target platform")

######################### customize when cross-compiling ###############################################################
# set COMPILER_PREFIX, for example, "" for default compiler, arm-linux- , or aarch64-linux- etc for cross compilers
# set generic build target
if (PLATFORM STREQUAL "X86_64")
  set(GENERIC_BUILD_TARGET X86_64)
  set(CROSS_ROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0")
  set(COMPILER_PREFIX "x86_64-unknown-linux-gnu-")
  set(SYSROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/x86_64-unknown-linux-gnu/sysroot")
  set(CMAKE_BUILD_FLAGS " -fno-math-errno -fopenmp -std=c99 -march=haswell")

elseif (PLATFORM STREQUAL "AARCH64")
  set(GENERIC_BUILD_TARGET AARCH64)
  set(CROSS_ROOT "/opt/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu")
#   set(CROSS_ROOT "/opt/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu")
  set(COMPILER_PREFIX "aarch64-none-linux-gnu-")

else()
  message(FATAL_ERROR "Please specify the target platform: X86_64 or AARCH64")
endif()

set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
set(CMAKE_LINKER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Get all files in the src folder
file(GLOB_RECURSE SRC_FILES ${SRC_DIR}/*.cpp)
# add_library(RuntimeLibrary SHARED ${SRC_FILES})
add_library(RuntimeLibrary SHARED ${SRC_FILES} ${SRC_DIR}/yyjson.c)
# add_executable(RuntimeLibrary ${SRC_FILES} ${SRC_DIR}/yyjson.c)

# add build flags
target_compile_options(RuntimeLibrary PUBLIC -std=c++17 -O3)

# include
target_include_directories(RuntimeLibrary PUBLIC ${INCLUDE_DIR})

# add memx library
target_include_directories(RuntimeLibrary PUBLIC ${MEMX_DEPS}/include)
target_link_directories(RuntimeLibrary PUBLIC ${MEMX_DEPS}/${GENERIC_BUILD_TARGET})
target_include_directories(RuntimeLibrary PUBLIC ${MX_ACCL_DEPS}/include)
target_link_directories(RuntimeLibrary PUBLIC ${MX_ACCL_DEPS}/${GENERIC_BUILD_TARGET})
target_link_libraries(RuntimeLibrary PUBLIC memx)
target_link_libraries(RuntimeLibrary PUBLIC mx_accl)

# Other libraries
target_link_libraries(RuntimeLibrary PUBLIC 
    pthread
)