// yolov8_nano_640_async.cpp
//
// Asynchronous 2-thread YOLOv8 camera sample using OAAX runtime wrapper.
//
// Threading model (simple + safe):
//  - Producer thread: capture + preprocess + send_input()
//    * owns the reusable input tensors_struct (no sharing across threads)
//    * pushes the original frame + pad metadata into a small queue keyed by frame_id
//  - Consumer thread: receive_output() + parse + NMS + draw + display
//    * owns all output tensors_struct allocations (returned from runtime) and frees them
//  - This example does NOT attempt multi-stream; it’s single cam, single stream_id.


#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "runtime_core.hpp"
#include "runtime_ioinfo.hpp"

// -----------------------------
// COCO class names
// -----------------------------
static std::vector<std::string> class_names = {
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "sofa","potted plant","bed","dining table","toilet","tv monitor","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

static inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

static size_t product_dims(const size_t* dims, size_t rank) {
    size_t p = 1;
    for (size_t i = 0; i < rank; ++i) p *= dims[i];
    return p;
}

// -----------------------------
// Input shape interpretation
// -----------------------------
static bool parse_input_hw_c(const io_info* io, int* H, int* W, int* C) {
    if (!io || io->num_inputs < 1) return false;
    if (!io->input_shapes || !io->input_shapes[0]) return false;

    const size_t r = io->input_ranks[0];
    const size_t* s = io->input_shapes[0];

    if (r == 4) { // [H, W, 1, C]
        if (s[2] != 1) return false;
        *H = (int)s[0];
        *W = (int)s[1];
        *C = (int)s[3];
        return true;
    } else if (r == 3) { // [H, W, C]
        *H = (int)s[0];
        *W = (int)s[1];
        *C = (int)s[2];
        return true;
    }
    return false;
}

// -----------------------------
// OAAX input tensor allocation (reused every frame)
// -----------------------------
static tensors_struct* make_reusable_input(const io_info* io) {
    if (!io || io->num_inputs != 1) return nullptr;

    tensors_struct* in = allocate_tensors_struct(1);
    if (!in) return nullptr;

    const char* nm = (io->input_names && io->input_names[0]) ? io->input_names[0] : "input0";
    in->names[0] = (char*)std::malloc(std::strlen(nm) + 1);
    if (!in->names[0]) { deep_free_tensors_struct(in); return nullptr; }
    std::memcpy(in->names[0], nm, std::strlen(nm) + 1);

    in->data_types[0] = DATA_TYPE_FLOAT;
    in->ranks[0] = io->input_ranks[0];

    in->shapes[0] = (size_t*)std::malloc(in->ranks[0] * sizeof(size_t));
    if (!in->shapes[0]) { deep_free_tensors_struct(in); return nullptr; }
    std::memcpy(in->shapes[0], io->input_shapes[0], in->ranks[0] * sizeof(size_t));

    const size_t elems = product_dims(in->shapes[0], in->ranks[0]);
    in->data[0] = std::malloc(elems * sizeof(float));
    if (!in->data[0]) { deep_free_tensors_struct(in); return nullptr; }

    return in;
}

// -----------------------------
// Preprocess
// -----------------------------
struct SquarePadInfo {
    int length = 0;  // square side length in original pixels
    int orig_w = 0;
    int orig_h = 0;
};

static cv::Mat square_pad_topleft_resize_rgb_f32(
    const cv::Mat& bgr,
    int model_w, int model_h,
    SquarePadInfo* info_out)
{
    const int orig_w = bgr.cols;
    const int orig_h = bgr.rows;
    const int length = std::max(orig_w, orig_h);

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat square = cv::Mat::zeros(cv::Size(length, length), rgb.type());
    rgb.copyTo(square(cv::Rect(0, 0, orig_w, orig_h))); // top-left pad

    cv::Mat resized;
    cv::resize(square, resized, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat f32;
    resized.convertTo(f32, CV_32FC3, 1.0 / 255.0);

    if (info_out) {
        info_out->length = length;
        info_out->orig_w = orig_w;
        info_out->orig_h = orig_h;
    }
    return f32; // RGB float32 HWC interleaved
}

static void pack_rgb_hwc_to_chw(const cv::Mat& rgb_f32_hwc, float* dst_chw, int H, int W) {
    for (int y = 0; y < H; ++y) {
        const cv::Vec3f* row = rgb_f32_hwc.ptr<cv::Vec3f>(y);
        for (int x = 0; x < W; ++x) {
            const cv::Vec3f& v = row[x]; // R,G,B
            dst_chw[0 * H * W + y * W + x] = v[0];
            dst_chw[1 * H * W + y * W + x] = v[1];
            dst_chw[2 * H * W + y * W + x] = v[2];
        }
    }
}

// -----------------------------
// Post output parsing
// Output: [1,84,8400] float; feature-major [84][8400]
// -----------------------------
struct Det {
    float x1, y1, x2, y2;
    float score;
    int cls;
};

static bool parse_yolo_output(
    const tensors_struct* out,
    std::vector<Det>& dets,
    float conf_thr,
    int model_w, int model_h,
    int square_length)
{
    dets.clear();
    if (!out || out->num_tensors < 1) return false;

    const int t = 0;
    if (out->data_types[t] != DATA_TYPE_FLOAT) return false;
    if (!out->data[t] || !out->shapes[t]) return false;
    if (out->ranks[t] != 3) return false;

    const size_t B = out->shapes[t][0];
    const size_t C = out->shapes[t][1];
    const size_t N = out->shapes[t][2];

    if (B != 1 || C != 84 || N == 0) return false;

    const float* p = reinterpret_cast<const float*>(out->data[t]);
    auto at = [&](size_t c, size_t i) -> float { return p[c * N + i]; };

    const float x_factor = (float)square_length / (float)model_w;
    const float y_factor = (float)square_length / (float)model_h;

    dets.reserve(2048);

    for (size_t i = 0; i < N; ++i) {
        float x0 = at(0, i);
        float y0 = at(1, i);
        float w  = at(2, i);
        float h  = at(3, i);

        x0 *= x_factor;
        y0 *= y_factor;
        w  *= x_factor;
        h  *= y_factor;

        const float x1 = x0 - 0.5f * w;
        const float y1 = y0 - 0.5f * h;
        const float x2 = x0 + 0.5f * w;
        const float y2 = y0 + 0.5f * h;

        if (x2 <= x1 || y2 <= y1) continue;

        for (int cls = 0; cls < 80; ++cls) {
            const float conf = at((size_t)(4 + cls), i);
            if (conf > conf_thr) {
                dets.push_back(Det{x1, y1, x2, y2, conf, cls});
            }
        }
    }
    return true;
}

static void clamp_box_to_original(Det& d, int orig_w, int orig_h) {
    d.x1 = clampf(d.x1, 0.f, (float)(orig_w - 1));
    d.x2 = clampf(d.x2, 0.f, (float)(orig_w - 1));
    d.y1 = clampf(d.y1, 0.f, (float)(orig_h - 1));
    d.y2 = clampf(d.y2, 0.f, (float)(orig_h - 1));
}

static std::vector<Det> nms_opencv(const std::vector<Det>& dets, float conf_thr, float nms_thr) {
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    boxes.reserve(dets.size());
    scores.reserve(dets.size());

    for (const auto& d : dets) {
        const int x = (int)std::round(d.x1);
        const int y = (int)std::round(d.y1);
        const int w = (int)std::round(std::max(0.f, d.x2 - d.x1));
        const int h = (int)std::round(std::max(0.f, d.y2 - d.y1));
        boxes.emplace_back(cv::Rect(x, y, w, h));
        scores.push_back(d.score);
    }

    std::vector<int> idxs;
    if (!boxes.empty()) cv::dnn::NMSBoxes(boxes, scores, conf_thr, nms_thr, idxs);

    std::vector<Det> out;
    out.reserve(idxs.size());
    for (int idx : idxs) out.push_back(dets[(size_t)idx]);
    return out;
}

static void draw_dets(cv::Mat& frame, const std::vector<Det>& dets) {
    for (const auto& d : dets) {
        const int x1 = (int)std::round(d.x1);
        const int y1 = (int)std::round(d.y1);
        const int x2 = (int)std::round(d.x2);
        const int y2 = (int)std::round(d.y2);

        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

        const char* cls_name = "unknown";
        if (d.cls >= 0 && (size_t)d.cls < class_names.size()) cls_name = class_names[(size_t)d.cls].c_str();

        char txt[128];
        std::snprintf(txt, sizeof(txt), "%s %.2f", cls_name, d.score);

        int base = 0;
        const auto ts = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
        const int bx = std::max(0, x1);
        const int by = std::max(0, y1 - ts.height - 6);

        cv::rectangle(frame, cv::Rect(bx, by, ts.width + 8, ts.height + 8), cv::Scalar(0, 255, 0), -1);
        cv::putText(frame, txt, cv::Point(bx + 4, by + ts.height + 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
}

// -----------------------------
// Async Inference
// -----------------------------
struct FrameItem {
    uint64_t id = 0;
    cv::Mat bgr;        // original (for display + drawing)
    SquarePadInfo sp;   // mapping from model coords -> original coords
    std::chrono::steady_clock::time_point ts;
};

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}

    void push_drop_oldest(T&& v) {
        std::unique_lock<std::mutex> lk(m_);
        if (closed_) return;
        if (q_.size() >= cap_) q_.pop_front(); // drop oldest to keep latency bounded
        q_.push_back(std::move(v));
        cv_.notify_one();
    }

    // blocks until item available or closed+empty
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return closed_ || !q_.empty(); });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    void close() {
        std::unique_lock<std::mutex> lk(m_);
        closed_ = true;
        cv_.notify_all();
    }

private:
    size_t cap_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<T> q_;
    bool closed_{false};
};

// -----------------------------
// Main
// -----------------------------
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.zip> [camera_index] [conf_thr] [nms_thr]\n";
        return 2;
    }

    const std::string zip_path = argv[1];
    const int cam_index = (argc >= 3) ? std::atoi(argv[2]) : 0;
    const float conf_thr = (argc >= 4) ? (float)std::atof(argv[3]) : 0.25f;
    const float nms_thr  = (argc >= 5) ? (float)std::atof(argv[4]) : 0.45f;

    std::cout << "Runtime: " << runtime_name() << " v" << runtime_version() << "\n";

    const char* keys[] = {"log_level"};
    const void* vals[] = {"2"};
    if (runtime_initialization_with_args(1, keys, vals) != 0) {
        std::cerr << "runtime_initialization_with_args failed: " << runtime_error_message() << "\n";
        return 1;
    }

    if (runtime_model_loading(zip_path.c_str()) != 0) {
        std::cerr << "runtime_model_loading failed: " << runtime_error_message() << "\n";
        runtime_destruction();
        return 1;
    }

    const io_info* io = runtime_get_io_info();
    if (!io) {
        std::cerr << "runtime_get_io_info returned null\n";
        runtime_destruction();
        return 1;
    }

    int H=0, W=0, C=0;
    if (!parse_input_hw_c(io, &H, &W, &C) || C != 3) {
        std::cerr << "Unsupported input shape. Expected [H,W,1,3] or [H,W,3].\n";
        runtime_destruction();
        return 1;
    }

    std::cout << "Input: " << H << "x" << W << "x" << C
              << " | Outputs: " << io->num_outputs << "\n";

    // Camera (owned by producer thread)
    cv::VideoCapture cap(cam_index);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera " << cam_index << "\n";
        runtime_destruction();
        return 1;
    }

    // Producer owns reusable input tensors_struct
    tensors_struct* in = make_reusable_input(io);
    if (!in) {
        std::cerr << "Failed to allocate input tensors_struct\n";
        runtime_destruction();
        return 1;
    }

    std::atomic<bool> stop{false};
    std::atomic<bool> send_failed{false};
    std::atomic<bool> recv_failed{false};

    // Queue of frames that have been sent (consumer pops one per received output)
    BoundedQueue<FrameItem> sent_frames(/*cap=*/8);

    // FPS calc (consumer measures display FPS)
    using clock = std::chrono::steady_clock;
    auto last = clock::now();
    double ema_fps = 0.0;
    const double a = 0.1;
    int frame_count = 0;

    // -------------------------
    // Producer thread
    // -------------------------
    std::thread producer([&]{
        uint64_t frame_id = 0;
        cv::Mat frame;

        while (!stop.load()) {
            if (!cap.read(frame)) {
                stop.store(true);
                break;
            }

            FrameItem item;
            item.id = frame_id++;
            item.bgr = frame.clone(); // keep a copy for display thread
            item.ts = clock::now();

            // preprocess
            cv::Mat rgb_f32 = square_pad_topleft_resize_rgb_f32(frame, W, H, &item.sp);

            float* in_f = reinterpret_cast<float*>(in->data[0]);
            pack_rgb_hwc_to_chw(rgb_f32, in_f, H, W);

            // send
            if (send_input(in) != 0) {
                std::cerr << "send_input failed: " << runtime_error_message() << "\n";
                send_failed.store(true);
                stop.store(true);
                break;
            }

            // enqueue metadata/frame after successful send
            sent_frames.push_drop_oldest(std::move(item));
        }

        // signal consumer to finish
        sent_frames.close();
    });

    // -------------------------
    // Consumer thread
    // -------------------------
    std::thread consumer([&]{
        while (!stop.load()) {
            // match one received output to one sent frame
            FrameItem item;
            if (!sent_frames.pop(item)) {
                // queue closed and empty
                break;
            }

            tensors_struct* out = nullptr;
            if (receive_output(&out) != 0) {
                std::cerr << "receive_output failed: " << runtime_error_message() << "\n";
                recv_failed.store(true);
                stop.store(true);
                break;
            }

            // parse + NMS + draw
            std::vector<Det> raw_dets;
            const bool ok = parse_yolo_output(out, raw_dets, conf_thr, W, H, item.sp.length);

            if (!ok) {
                static bool printed = false;
                if (!printed) {
                    printed = true;
                    std::cerr << "Could not parse output as [1,84,N]. Dumping tensor shapes:\n";
                    for (int t = 0; t < out->num_tensors; ++t) {
                        std::cerr << "out[" << t << "] rank=" << out->ranks[t] << " shape=[";
                        for (size_t d = 0; d < out->ranks[t]; ++d) {
                            std::cerr << out->shapes[t][d] << (d + 1 < out->ranks[t] ? "," : "");
                        }
                        std::cerr << "] dtype=" << out->data_types[t] << "\n";
                    }
                }
            } else {
                for (auto& d : raw_dets) clamp_box_to_original(d, item.sp.orig_w, item.sp.orig_h);
                std::vector<Det> kept = nms_opencv(raw_dets, conf_thr, nms_thr);
                draw_dets(item.bgr, kept);
            }

            deep_free_tensors_struct(out);

            // FPS overlay (display-side FPS)
            auto now = clock::now();
            const double dt = std::chrono::duration<double>(now - last).count();
            last = now;
            const double inst_fps = (dt > 0.0) ? (1.0 / dt) : 0.0;
            ema_fps = (frame_count == 0) ? inst_fps : (a * inst_fps + (1.0 - a) * ema_fps);
            frame_count++;

            char fps_txt[128];
            std::snprintf(fps_txt, sizeof(fps_txt), "FPS: %.1f  conf=%.2f  nms=%.2f",
                          ema_fps, conf_thr, nms_thr);
            cv::putText(item.bgr, fps_txt, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);

            cv::imshow("YOLOv8 Nano (async)", item.bgr);
            const int k = cv::waitKey(1) & 0xFF;
            if (k == 'q' || k == 27) {
                stop.store(true);
                break;
            }
        }

        stop.store(true);
    });

    // Wait for threads
    producer.join();
    consumer.join();

    deep_free_tensors_struct(in);
    runtime_destruction();

    if (send_failed.load() || recv_failed.load()) return 1;
    return 0;
}