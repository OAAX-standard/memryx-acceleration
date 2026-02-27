cmake_minimum_required(VERSION 3.10)
project(RuntimeLibrary LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Paths
set(SRC_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# -------------------------
# OpenCV (needed by demo app)
# -------------------------
find_package(OpenCV REQUIRED)

 
# -------------------------
# Runtime shared library
# -------------------------
file(GLOB SRC_CPP ${SRC_DIR}/*.cpp)

add_library(RuntimeLibrary SHARED
  ${SRC_CPP}
  ${SRC_DIR}/yyjson.c
  ${SRC_DIR}/tensors_struct.c
)

target_include_directories(RuntimeLibrary
  PUBLIC
    ${INCLUDE_DIR}
)

target_compile_options(RuntimeLibrary
  PRIVATE
    -O3
    -Wall -Wextra -Wpedantic
)

target_link_libraries(RuntimeLibrary
  PUBLIC
    memx
    mx_accl
    pthread
)

# -------------------------
# Demo executable (YOLO camera)
# -------------------------
add_executable(yolo_cam
  ${SRC_DIR}/yolov8_nano_640_app.cpp   
)

target_compile_options(yolo_cam
  PRIVATE
    -O3
    -Wall -Wextra -Wpedantic
)

include_directories(${OpenCV_INCLUDE_DIRS})

target_include_directories(yolo_cam
  PRIVATE
    ${INCLUDE_DIR}
)
    # include_directories(${OpenCV_INCLUDE_DIRS})

target_link_libraries(yolo_cam
  PRIVATE
    RuntimeLibrary
    ${OpenCV_LIBS}
)

# -------------------------
# RPATH so the exe can find RuntimeLibrary in build tree
# -------------------------
# Put the directory containing libRuntimeLibrary.so into RUNPATH.
set_target_properties(yolo_cam PROPERTIES
  BUILD_RPATH "$ORIGIN;$ORIGIN/..;$ORIGIN/../lib;$<TARGET_FILE_DIR:RuntimeLibrary>"
  INSTALL_RPATH "$ORIGIN;$ORIGIN/..;$ORIGIN/../lib"
)