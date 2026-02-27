// runtime_core.cpp
//
// MemryX OAAX runtime wrapper (v1)
// - Loads a model archive (zip) containing a single .dfp (+ optional pre/post onnx)
// - Pre-allocates input/output staging buffers sized from MxModelInfo
// - send_input(): copies OAAX input into preallocated input buffers, then calls MemryX send_input()
// - receive_output(): calls MemryX receive_output() into preallocated output staging buffers,
//   then allocates an OAAX-owned tensors_struct and copies data into it (hybrid approach)
//
// Buffer allocation rules (requested addition):
// - If a pre-processing model is connected, allocate INPUT buffers based on pre_model_info (pre model inputs).
// - Else allocate INPUT buffers based on main model_info.
// - If a post-processing model is connected, allocate OUTPUT buffers based on post_model_info (post model outputs).
// - Else allocate OUTPUT buffers based on main model_info.
//
// Notes on threading/shutdown:
// - OAAX may call send_input/receive_output from worker threads while runtime_destruction() is invoked.
// - We guard MemryX API calls and teardown using accl_call_mutex to prevent use-after-free of ctx->accl.
//

#include <atomic>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h> // getpid()
#include <vector>

#include "runtime_ioinfo.hpp"
#include "tensors_struct.h"

#include <memx/accl/MxAcclMT.h>

namespace fs = std::filesystem;

// -----------------------------
// Helpers
// -----------------------------

// Minimal single-quote escaping for POSIX shell.
// Used only for the unzip system() call.
static std::string shell_escape(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

// Extract a zip archive to an output directory using the system `unzip` command.
// Returns false and sets `err` if extraction fails.
static bool unzip_to_dir(const fs::path& zip_path, const fs::path& out_dir, std::string* err) {
    std::error_code ec;
    fs::create_directories(out_dir, ec);
    if (ec) {
        if (err) *err = "Failed to create output directory: " + out_dir.string() + " : " + ec.message();
        return false;
    }

    std::ostringstream cmd;
    cmd << "unzip -o " << shell_escape(zip_path.string())
        << " -d " << shell_escape(out_dir.string())
        << " >/dev/null 2>&1";

    const int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        if (err) *err = "Failed to unzip archive (is unzip installed? is zip valid?): " + zip_path.string();
        return false;
    }
    return true;
}

// Recursively find all files with the given extension (case-insensitive compare by normalizing found ext).
static std::vector<fs::path> find_all_files_with_ext(const fs::path& root, const std::string& ext_lower) {
    std::vector<fs::path> out;
    std::error_code ec;
    for (auto const& p : fs::recursive_directory_iterator(root, ec)) {
        if (ec) break;
        if (!p.is_regular_file()) continue;

        auto e = p.path().extension().string();
        for (auto& ch : e) ch = (char)std::tolower((unsigned char)ch);
        if (e == ext_lower) out.push_back(p.path());
    }
    return out;
}

// Recursively find a file named "chain.json" (optional).
static std::optional<fs::path> find_chain_json(const fs::path& root) {
    std::error_code ec;
    for (auto const& p : fs::recursive_directory_iterator(root, ec)) {
        if (ec) break;
        if (!p.is_regular_file()) continue;
        if (p.path().filename() == "chain.json") return p.path();
    }
    return std::nullopt;
}

// Create a per-process unique temporary workspace directory path.
static fs::path make_workspace_dir() {
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::ostringstream ss;
    ss << "/tmp/oaax_memryx_" << (int)getpid() << "_" << now;
    return fs::path(ss.str());
}

// -----------------------------
// Runtime config/context
// -----------------------------

// Wrapper configuration (minimal v1).
struct RuntimeConfig {
    std::vector<int> device_ids_to_use{0};
    std::array<bool, 2> use_model_shape{true, true};
    bool local_mode{false};
};

// RuntimeContext owns all state for the loaded model/runtime instance.
class RuntimeContext {
public:
    RuntimeConfig config;

    // Workspace + extracted artifacts
    fs::path workspace_dir;
    fs::path dfp_path;
    fs::path chain_json_path;
    std::vector<fs::path> pre_onnx_paths;
    std::vector<fs::path> post_onnx_paths;

    // OAAX metadata mirror
    io_info* io = nullptr;

    bool initialized{false};
    bool model_loaded{false};
    bool running{false};
    std::atomic<bool> running_atomic{false};

    // MemryX accelerator handle
    std::unique_ptr<MX::Runtime::MxAcclMT> accl;

    // Guards calls into `accl` and safe destruction (prevents teardown races with send/recv).
    std::mutex accl_call_mutex;

    // Input staging buffers (pre-allocated once at model load).
    // OAAX input is copied into these buffers, then pointers passed to MemryX.
    std::vector<std::vector<float>> in_bufs; // in_bufs[i].size() == in_featuremap_sizes[i]
    std::vector<float*> in_ptrs;             // in_ptrs[i] == in_bufs[i].data()

    // Output staging buffers (pre-allocated once at model load).
    // MemryX writes into these buffers. We then allocate OAAX output tensors and memcpy into them.
    std::vector<std::vector<float>> out_bufs; // out_bufs[i].size() == out_featuremap_sizes[i]
    std::vector<float*> out_ptrs;             // out_ptrs[i] == out_bufs[i].data()
    std::vector<size_t> out_elems;            // out_featuremap_sizes[i] (element count)

    // Cached model infos
    MX::Types::MxModelInfo model_info;
    bool model_info_valid{false};

    // Optional pre/post model infos (only valid if connected)
    MX::Types::MxModelInfo pre_model_info;
    MX::Types::MxModelInfo post_model_info;
    bool pre_model_connected{false};
    bool post_model_connected{false};

    // Buffer allocation reference infos (used for validation against OAAX tensors_struct)
    // - input_ref_info: expected input featuremap sizes/count for OAAX send_input
    // - output_ref_info: expected output featuremap sizes/count for OAAX receive_output
    MX::Types::MxModelInfo input_ref_info;
    MX::Types::MxModelInfo output_ref_info;

    int stream_id{0};
    int model_id{0};

    // Error state
    std::mutex state_mutex;
    std::string last_error;

    void set_error(const std::string& err) {
        std::lock_guard<std::mutex> lk(state_mutex);
        last_error = err;
    }

    std::string get_error_copy() {
        std::lock_guard<std::mutex> lk(state_mutex);
        return last_error;
    }

    // Reset state for a fresh model load. Caller must ensure no concurrent send/recv.
    void clear_partial_state_no_lock() {
        // Clear staging buffers and any derived state.
        in_bufs.clear();
        in_ptrs.clear();
        out_bufs.clear();
        out_ptrs.clear();
        out_elems.clear();

        pre_onnx_paths.clear();
        post_onnx_paths.clear();
        chain_json_path.clear();
        dfp_path.clear();

        input_ref_info = MX::Types::MxModelInfo{};
        output_ref_info = MX::Types::MxModelInfo{};
        pre_model_info = MX::Types::MxModelInfo{};
        post_model_info = MX::Types::MxModelInfo{};
        model_info = MX::Types::MxModelInfo{};

        pre_model_connected = false;
        post_model_connected = false;
        model_info_valid = false;

        if (io) {
            free_io_info(io);
            io = nullptr;
        }

        // Reset MemryX resources
        accl.reset();

        model_loaded = false;
        running = false;
        running_atomic.store(false);
    }
};

// -----------------------------
// Singleton runtime instance
// -----------------------------

static std::unique_ptr<RuntimeContext> g_ctx;
static std::mutex g_ctx_mutex;

// -----------------------------
// OAAX required exported funcs
// -----------------------------

extern "C" const char* runtime_name() { return "MemryX-OAAX-Runtime"; }
extern "C" const char* runtime_version() { return "1.0.0"; }

// Return the last error message recorded by this runtime on the calling thread.
// If no error, returns "OK".
extern "C" const char* runtime_error_message() {
    static thread_local std::string tls_err;
    RuntimeContext* ctx = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_ctx_mutex);
        ctx = g_ctx.get();
    }
    if (!ctx) {
        tls_err = "Runtime not initialized";
        return tls_err.c_str();
    }
    tls_err = ctx->get_error_copy();
    if (tls_err.empty()) tls_err = "OK";
    return tls_err.c_str();
}

// -----------------------------------------
// runtime_initialization(_with_args)
// -----------------------------------------

// Initialize the runtime singleton. v1 accepts keys but mostly ignores them.
// Returns 0 on success.
extern "C"
int runtime_initialization_with_args(int length, const char** keys, const void** values) {
    std::lock_guard<std::mutex> lk(g_ctx_mutex);

    if (g_ctx) {
        g_ctx->set_error("Runtime already initialized");
        return 1;
    }

    g_ctx = std::make_unique<RuntimeContext>();

    for (int i = 0; i < length; ++i) {
        const char* key = keys ? keys[i] : nullptr;
        const char* val = values ? static_cast<const char*>(values[i]) : nullptr;
        if (!key) continue;

        const std::string k(key);

        if (k == "log_level") {
            // Accepted for compatibility; not yet used in v1.
            (void)val;
            continue;
        }
        // Future keys:
        // device_ids, local_mode, scheduler options, etc.
    }

    g_ctx->initialized = true;
    g_ctx->set_error("OK");
    return 0;
}

// Initialize without args. Returns 0 on success.
extern "C"
int runtime_initialization(void) {
    std::lock_guard<std::mutex> lk(g_ctx_mutex);
    if (g_ctx) return 1;

    g_ctx = std::make_unique<RuntimeContext>();
    g_ctx->initialized = true;
    g_ctx->set_error("OK");
    return 0;
}

// -----------------------------------------
// runtime_model_loading (zip -> dfp + optional pre/post)
// -----------------------------------------

static bool validate_featuremap_sizes_vector(const MX::Types::MxModelInfo& mi, std::string* err) {
    // We only validate the size vectors against reported counts since those are used for allocations.
    if (mi.in_featuremap_sizes.size() != (size_t)mi.num_in_featuremaps) {
        if (err) *err = "model_info in_featuremap_sizes size mismatch";
        return false;
    }
    if (mi.out_featuremap_sizes.size() != (size_t)mi.num_out_featuremaps) {
        if (err) *err = "model_info out_featuremap_sizes size mismatch";
        return false;
    }
    return true;
}

// Load a MemryX OAAX model archive (.zip) by:
// - Extracting it to a temporary workspace directory
// - Finding a single .dfp (required)
// - Finding optional chain.json and optional *_pre.onnx / *_post.onnx models
// - Constructing MxAcclMT and connecting pre/post models
// - Querying model_info and building io_info
// - Pre-allocating input/output staging buffers based on:
//     * pre_model_info (inputs) if pre model is connected, else main model_info
//     * post_model_info (outputs) if post model is connected, else main model_info
extern "C"
int runtime_model_loading(const char* model_path) {
    RuntimeContext* ctx = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_ctx_mutex);
        ctx = g_ctx.get();
    }

    if (!ctx || !ctx->initialized) {
        if (ctx) ctx->set_error("runtime_model_loading: runtime not initialized");
        return 1;
    }
    if (!model_path || std::strlen(model_path) == 0) {
        ctx->set_error("runtime_model_loading: model_path is null/empty");
        return 1;
    }
    if (ctx->model_loaded) {
        ctx->set_error("runtime_model_loading: model already loaded");
        return 1;
    }

    // Ensure we're starting from a clean state.
    ctx->clear_partial_state_no_lock();

    try {
        const fs::path input_path(model_path);
        std::printf("[runtime_model_loading] zip: %s\n", input_path.c_str());

        if (!fs::exists(input_path)) {
            ctx->set_error("runtime_model_loading: path does not exist: " + input_path.string());
            return 1;
        }

        // Create workspace and extract zip.
        ctx->workspace_dir = make_workspace_dir();
        std::printf("[runtime_model_loading] workspace: %s\n", ctx->workspace_dir.c_str());

        std::string unzip_err;
        if (!unzip_to_dir(input_path, ctx->workspace_dir, &unzip_err)) {
            ctx->set_error("runtime_model_loading: " + unzip_err);
            return 1;
        }

        // Find a single DFP.
        auto dfps = find_all_files_with_ext(ctx->workspace_dir, ".dfp");
        if (dfps.empty()) {
            ctx->set_error("runtime_model_loading: no .dfp found in extracted archive");
            return 1;
        }
        if (dfps.size() > 1) {
            ctx->set_error("runtime_model_loading: multiple .dfp files found (ambiguous)");
            return 1;
        }
        ctx->dfp_path = dfps[0];
        std::printf("[runtime_model_loading] dfp: %s\n", ctx->dfp_path.c_str());

        // chain.json (optional).
        if (auto cj = find_chain_json(ctx->workspace_dir)) {
            ctx->chain_json_path = *cj;
        }

        // Detect optional pre/post ONNX models (heuristic by filename).
        auto onnx_files = find_all_files_with_ext(ctx->workspace_dir, ".onnx");
        for (auto const& p : onnx_files) {
            std::string lower = p.filename().string();
            for (auto& ch : lower) ch = (char)std::tolower((unsigned char)ch);

            if (lower.find("_pre") != std::string::npos) ctx->pre_onnx_paths.push_back(p);
            else if (lower.find("_post") != std::string::npos) ctx->post_onnx_paths.push_back(p);
        }

        std::printf("[runtime_model_loading] pre count: %zu, post count: %zu\n",
                    ctx->pre_onnx_paths.size(), ctx->post_onnx_paths.size());
        if (!ctx->pre_onnx_paths.empty()) {
            std::printf("[runtime_model_loading] pre model: %s\n", ctx->pre_onnx_paths[0].c_str());
        }
        if (!ctx->post_onnx_paths.empty()) {
            std::printf("[runtime_model_loading] post model: %s\n", ctx->post_onnx_paths[0].c_str());
        }

        // Construct accelerator (manual threading).
        ctx->accl = std::make_unique<MX::Runtime::MxAcclMT>(
            ctx->dfp_path,
            ctx->config.device_ids_to_use,
            ctx->config.use_model_shape,
            ctx->config.local_mode
        );
        std::printf("[runtime_model_loading] MxAcclMT constructed OK\n");

        // Model/stream IDs for v1.
        ctx->model_id = 0;
        ctx->stream_id = 0;

        // Connect optional pre model.
        if (!ctx->pre_onnx_paths.empty()) {
            if (ctx->pre_onnx_paths.size() > 1) {
                ctx->set_error("runtime_model_loading: multiple pre models found; expected at most one");
                return 1;
            }
            const fs::path pre_abs = fs::absolute(ctx->pre_onnx_paths[0]);
            ctx->accl->connect_pre_model(pre_abs, /*model_id=*/ctx->model_id);
            ctx->pre_model_connected = true;

            // Retrieve pre model info (used for INPUT staging buffers).
            ctx->pre_model_info = ctx->accl->get_pre_model_info(ctx->model_id);

            std::string verr;
            if (!validate_featuremap_sizes_vector(ctx->pre_model_info, &verr)) {
                ctx->set_error("runtime_model_loading: pre_model_info invalid: " + verr);
                return 1;
            }
        }

        // Connect optional post model.
        if (!ctx->post_onnx_paths.empty()) {
            if (ctx->post_onnx_paths.size() > 1) {
                ctx->set_error("runtime_model_loading: multiple post models found; expected at most one");
                return 1;
            }
            const fs::path post_abs = fs::absolute(ctx->post_onnx_paths[0]);
            const std::vector<size_t> post_size_list = {}; // v1: leave empty unless needed
            ctx->accl->connect_post_model(post_abs, /*model_id=*/ctx->model_id, post_size_list);
            ctx->post_model_connected = true;

            // Retrieve post model info (used for OUTPUT staging buffers).
            ctx->post_model_info = ctx->accl->get_post_model_info(ctx->model_id);

            std::string verr;
            if (!validate_featuremap_sizes_vector(ctx->post_model_info, &verr)) {
                ctx->set_error("runtime_model_loading: post_model_info invalid: " + verr);
                return 1;
            }
        }

        // Retrieve main model info (still needed for io_info + validation/logging).
        ctx->model_info = ctx->accl->get_model_info(ctx->model_id);
        ctx->model_info_valid = true;

        {
            std::string verr;
            if (!validate_featuremap_sizes_vector(ctx->model_info, &verr)) {
                ctx->set_error("runtime_model_loading: model_info invalid: " + verr);
                return 1;
            }
        }

        // Decide which model info defines external IO:
        // - Inputs: pre model if connected, else main model
        // - Outputs: post model if connected, else main model
        ctx->input_ref_info = ctx->pre_model_connected ? ctx->pre_model_info : ctx->model_info;
        ctx->output_ref_info = ctx->post_model_connected ? ctx->post_model_info : ctx->model_info;

        // Pre-allocate input staging buffers based on input_ref_info.
        {
            const size_t n_in = (size_t)ctx->input_ref_info.num_in_featuremaps;
            ctx->in_bufs.assign(n_in, {});
            ctx->in_ptrs.assign(n_in, nullptr);

            for (size_t i = 0; i < n_in; ++i) {
                const size_t elems = ctx->input_ref_info.in_featuremap_sizes[i];
                ctx->in_bufs[i].resize(elems);
                ctx->in_ptrs[i] = ctx->in_bufs[i].data();
            }
        }

        // Pre-allocate output staging buffers based on output_ref_info.
        {
            const size_t n_out = (size_t)ctx->output_ref_info.num_out_featuremaps;
            ctx->out_bufs.assign(n_out, {});
            ctx->out_ptrs.assign(n_out, nullptr);
            ctx->out_elems.assign(n_out, 0);

            for (size_t i = 0; i < n_out; ++i) {
                const size_t elems = ctx->output_ref_info.out_featuremap_sizes[i];
                ctx->out_elems[i] = elems;
                ctx->out_bufs[i].resize(elems);
                ctx->out_ptrs[i] = ctx->out_bufs[i].data();
            }
        }

        // Build io_info (names/ranks/shapes/dtypes) for the final externally-visible IO.
        // Important: initialize_io_info_from_model_info() must reflect the same IO definition
        // that OAAX expects. If pre/post are attached, the visible IO is pre-inputs and post-outputs.
        //
        // In v1, we approximate by using:
        // io_info should reflect the externally visible graph boundary:

        // No pre, no post → inputs/outputs from model_info

        // Pre only → inputs from pre_model_info, outputs from model_info

        // Post only → inputs from model_info, outputs from post_model_info

        // Pre + Post → inputs from pre_model_info, outputs from post_model_info

        const MX::Types::MxModelInfo& ext_in  = ctx->pre_model_connected  ? ctx->pre_model_info  : ctx->model_info;
        const MX::Types::MxModelInfo& ext_out = ctx->post_model_connected ? ctx->post_model_info : ctx->model_info;

        ctx->io = initialize_io_info_from_model_infos(ext_in, ext_out);
        if (!ctx->io) {
            ctx->set_error("runtime_model_loading: failed to initialize io_info (composed)");
            return 1;
        }

        std::printf("\nPrinting io info:\n");
        print_io_info(ctx->io);

        // Extra sanity: ensure io_info counts match the buffers we allocated.
        if ((size_t)ctx->io->num_inputs != ctx->in_bufs.size()) {
            ctx->set_error("runtime_model_loading: io_info inputs count does not match input staging buffers");
            return 1;
        }
        if ((size_t)ctx->io->num_outputs != ctx->out_bufs.size()) {
            ctx->set_error("runtime_model_loading: io_info outputs count does not match output staging buffers");
            return 1;
        }

        ctx->running_atomic.store(true);
        ctx->running = true;
        ctx->model_loaded = true;
        ctx->set_error("OK");
        return 0;

    } catch (const std::exception& e) {
        ctx->set_error(std::string("runtime_model_loading: exception: ") + e.what());
        return 1;
    }
}

// -----------------------------------------
// send_input / receive_output (OAAX)
// -----------------------------------------

// Send OAAX input tensors to MemryX.
// - Validates the OAAX tensor count, datatype, and element count matches input_ref_info.
// - Copies OAAX-provided input data into preallocated input staging buffers.
// - Calls MemryX send_input() with pointers to the staging buffers.
// Assumption: MemryX send_input() copies input data before returning.
extern "C"
int send_input(const tensors_struct* input) {
    RuntimeContext* ctx = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_ctx_mutex);
        ctx = g_ctx.get();
    }

    {
        std::lock_guard<std::mutex> lk(ctx->accl_call_mutex);
        if (!ctx->accl) { ctx->set_error("... accl is null (shutdown)"); return 1; }
    }
    if (!ctx || !ctx->initialized || !ctx->model_loaded || !ctx->running_atomic.load()) {
        if (ctx) ctx->set_error("send_input: runtime not ready");
        return 1;
    }
    if (!input) { ctx->set_error("send_input: input is null"); return 1; }
    if (!ctx->model_info_valid) { ctx->set_error("send_input: model_info missing"); return 1; }

    const size_t n = (size_t)ctx->input_ref_info.num_in_featuremaps;
    if ((size_t)input->num_tensors != n) {
        ctx->set_error("send_input: num_tensors mismatch vs expected input featuremaps");
        return 1;
    }
    if (ctx->in_bufs.size() != n || ctx->in_ptrs.size() != n) {
        ctx->set_error("send_input: cached input buffers not initialized");
        return 1;
    }

    // Validate and copy each input tensor into staging.
    for (size_t i = 0; i < n; ++i) {
        if (input->data_types[i] != DATA_TYPE_FLOAT) {
            ctx->set_error("send_input: only float inputs supported in v1");
            return 1;
        }
        if (!input->data[i] || !input->shapes[i] || input->ranks[i] == 0) {
            ctx->set_error("send_input: invalid tensor (data/shapes/rank)");
            return 1;
        }

        // Compute element count from OAAX shapes.
        size_t elems = 1;
        for (size_t d = 0; d < input->ranks[i]; ++d) {
            const size_t dim = input->shapes[i][d];
            if (dim == 0) { ctx->set_error("send_input: zero dimension in shape"); return 1; }
            elems *= dim;
        }

        const size_t expected = ctx->input_ref_info.in_featuremap_sizes[i];
        if (expected != elems) {
            std::ostringstream oss;
            oss << "send_input: input[" << i << "] elems mismatch got=" << elems
                << " expected=" << expected;
            ctx->set_error(oss.str());
            return 1;
        }

        if (ctx->in_bufs[i].size() != expected) {
            ctx->set_error("send_input: cached buffer size mismatch (internal)");
            return 1;
        }

        std::memcpy(ctx->in_bufs[i].data(), input->data[i], expected * sizeof(float));
    }

    // Call MemryX send_input() with a short timeout and retry loop,
    // so runtime_destruction() can stop a blocked send.
    const int32_t step_timeout_ms = 100;
    while (ctx->running_atomic.load()) {
        bool ok = false;
        {
            std::lock_guard<std::mutex> lk(ctx->accl_call_mutex);
            ok = ctx->accl->send_input(ctx->in_ptrs, ctx->model_id, ctx->stream_id, step_timeout_ms);
        }
        if (ok) {
            ctx->set_error("OK");
            return 0;
        }
        // timeout -> retry
    }

    ctx->set_error("send_input: runtime stopping");
    return 1;
}

// Receive MemryX outputs and return them to OAAX.
// Hybrid approach:
// - MemryX writes into preallocated output staging buffers (out_bufs).
// - We allocate a new OAAX tensors_struct each call (caller owns and frees it),
//   allocate per-output data buffers, and memcpy from staging to the OAAX-owned buffers.
extern "C"
int receive_output(tensors_struct** output) {
    if (!output) return 1;
    *output = nullptr;

    RuntimeContext* ctx = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_ctx_mutex);
        ctx = g_ctx.get();
    }

    {
        std::lock_guard<std::mutex> lk(ctx->accl_call_mutex);
        if (!ctx->accl) { ctx->set_error("... accl is null (shutdown)"); return 1; }
    }

    if (!ctx || !ctx->initialized || !ctx->model_loaded || !ctx->running_atomic.load()) {
        if (ctx) ctx->set_error("receive_output: runtime not ready");
        return 1;
    }
    if (!ctx->io || !ctx->model_info_valid) {
        ctx->set_error("receive_output: io_info or model_info missing");
        return 1;
    }

    const size_t n = (size_t)ctx->io->num_outputs;

    // v1: float outputs only
    for (size_t i = 0; i < n; ++i) {
        if (ctx->io->output_datatypes[i] != DATA_TYPE_FLOAT) {
            ctx->set_error("receive_output: only float outputs supported in v1");
            return 1;
        }
    }

    // Staging must be initialized and consistent with io_info.
    if (ctx->out_ptrs.size() != n || ctx->out_bufs.size() != n || ctx->out_elems.size() != n) {
        ctx->set_error("receive_output: staging out buffers not initialized");
        return 1;
    }

    // 1) Receive into staging buffers (no allocations).
    const int32_t step_timeout_ms = 100;
    bool got = false;
    while (ctx->running_atomic.load()) {
        bool ok = false;
        {
            std::lock_guard<std::mutex> lk(ctx->accl_call_mutex);
            ok = ctx->accl->receive_output(ctx->out_ptrs, ctx->model_id, ctx->stream_id, step_timeout_ms);
        }
        if (ok) { got = true; break; }
        // timeout -> retry
    }
    if (!got) {
        ctx->set_error("receive_output: runtime stopping");
        return 1;
    }

    // 2) Allocate OAAX tensors_struct for return (caller owns it).
    tensors_struct* out = allocate_tensors_struct((int)n);
    if (!out) {
        ctx->set_error("receive_output: allocate_tensors_struct failed");
        return 1;
    }

    // 3) Fill metadata + allocate per-output data + copy from staging.
    for (size_t i = 0; i < n; ++i) {
        // name
        const char* src_name = ctx->io->output_names[i];
        out->names[i] = (char*)std::malloc(std::strlen(src_name) + 1);
        if (!out->names[i]) {
            deep_free_tensors_struct(out);
            ctx->set_error("receive_output: malloc name failed");
            return 1;
        }
        std::memcpy(out->names[i], src_name, std::strlen(src_name) + 1);

        // dtype / rank
        out->data_types[i] = ctx->io->output_datatypes[i];
        out->ranks[i] = ctx->io->output_ranks[i];

        // shapes
        out->shapes[i] = (size_t*)std::malloc(out->ranks[i] * sizeof(size_t));
        if (!out->shapes[i]) {
            deep_free_tensors_struct(out);
            ctx->set_error("receive_output: malloc shapes failed");
            return 1;
        }
        std::memcpy(out->shapes[i], ctx->io->output_shapes[i], out->ranks[i] * sizeof(size_t));

        // data
        const size_t elems = ctx->out_elems[i];
        out->data[i] = std::malloc(elems * sizeof(float));
        if (!out->data[i]) {
            deep_free_tensors_struct(out);
            ctx->set_error("receive_output: malloc output data failed");
            return 1;
        }
        std::memcpy(out->data[i], ctx->out_bufs[i].data(), elems * sizeof(float));
    }

    *output = out;
    ctx->set_error("OK");
    return 0;
}

// -----------------------------------------
// runtime_destruction
// -----------------------------------------

// Destroy the runtime and free resources.
// - Sets running_atomic=false to stop send/recv loops.
// - Takes the accl_call_mutex to ensure no concurrent MemryX calls are in-flight
//   while we reset the accelerator.
extern "C"
int runtime_destruction(void) {
    std::unique_ptr<RuntimeContext> local;
    {
        std::lock_guard<std::mutex> lk(g_ctx_mutex);
        if (!g_ctx) return 1;
        local = std::move(g_ctx);
    }

    // Stop loops first.
    local->running_atomic.store(false);

    // Prevent teardown races with send_input/receive_output.
    {
        std::lock_guard<std::mutex> lk(local->accl_call_mutex);
        local->accl.reset();
    }

    // Free io_info.
    if (local->io) {
        free_io_info(local->io);
        local->io = nullptr;
    }

    // Remove workspace directory.
    if (!local->workspace_dir.empty()) {
        std::error_code ec;
        fs::remove_all(local->workspace_dir, ec);
    }

    return 0;
}


// Returns a pointer to runtime-owned io_info (read-only).
// Lifetime: valid until runtime_destruction() or next runtime_model_loading().
// Thread-safety: pointer itself is stable after model_load; do not call during model_loading/destroy.
extern "C"
const io_info* runtime_get_io_info(void) {
    RuntimeContext* ctx = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_ctx_mutex);
        ctx = g_ctx.get();
    }
    if (!ctx || !ctx->initialized || !ctx->model_loaded || !ctx->io) return nullptr;
    return ctx->io;
}