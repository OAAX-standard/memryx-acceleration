# MemryX OAAX Runtime Library + YOLOv8 Camera Demo

This folder provides:

-   **RuntimeLibrary** --- A shared library implementing an
    OAAX-compatible runtime wrapper backed by MemryX.
-   **YOLOv8 Camera Demo (`main`)** --- A sample application that:
    -   Captures frames from a camera
    -   Sends inputs asynchronously to the runtime
    -   Receives outputs asynchronously
    -   Parses YOLOv8 detections
    -   Displays bounding boxes and FPS using OpenCV

------------------------------------------------------------------------

# 1. Prerequisites

## 1.1 MemryX Runtime Libs

-   memx_drivers
-   memx_accl

Follow the official installation guide at: [MemryX Developer Hub](https://developer.memryx.com)

Ensure: - MemryX drivers are loaded - memx and mx_accl libraries are
installed. If these are not installed correctly, the runtime library will fail to build or link.

------------------------------------------------------------------------

## 1.2 OpenCV (Required for Demo App)

The YOLOv8 camera demo requires OpenCV with:

-   core
-   imgproc
-   highgui
-   dnn

Install OpenCV using your platform's package manager or build from
source.

Example (Ubuntu):

    sudo apt install libopencv-dev

------------------------------------------------------------------------

# 2. Build Instructions

From the runtime folder root:

    mkdir -p build
    cd build
    cmake ..
    cmake --build . -j

After successful build, the following artifacts will be generated:

-   libRuntimeLibrary.so --- Runtime wrapper shared library
-   yolo_cam --- YOLOv8 camera demo executable

------------------------------------------------------------------------

# 3. Running the YOLOv8 Demo

A sample YOLOv8 model archive is included:

    assets/yolov8.zip

Run the demo:

    cd build
    ./yolo_cam ../assets/yolov8.zip

------------------------------------------------------------------------

## Optional Arguments

    ./yolo_cam <model.zip> [camera_index] [conf_threshold] [nms_threshold]

Example:

    ./yolo_cam ../assets/yolov8.zip 0 0.25 0.45

Parameters:

-   model.zip --- Path to zipped model archive
-   camera_index --- Default: 0
-   conf_threshold --- Detection confidence threshold (default: 0.25)
-   nms_threshold --- NMS IoU threshold (default: 0.45)

------------------------------------------------------------------------

## Controls

-   Press `q` to exit
-   Press `Esc` to exit

------------------------------------------------------------------------

# 4. Runtime API Overview

The wrapper exposes OAAX-style runtime functions:

    int runtime_initialization();
    int runtime_model_loading(const char *file_path);
    int send_input(const tensors_struct *input_tensors);
    int receive_output(tensors_struct **output_tensors);
    int runtime_destruction();

Additionally:

    const io_info* runtime_get_io_info(void);

This function exposes model input/output tensor metadata for
application-side preprocessing.


------------------------------------------------------------------------

# 9. License

Provided as reference implementation for MemryX OAAX integration.
