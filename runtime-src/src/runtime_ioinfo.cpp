#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "runtime_ioinfo.hpp"

// -----------------------------
// Helpers
// -----------------------------

static char* dup_cstr(const char* s) {
    if (!s) return nullptr;
    const size_t len = std::strlen(s);
    char* p = (char*)std::malloc(len + 1);
    if (!p) return nullptr;
    std::memcpy(p, s, len + 1); // includes '\0'
    return p;
}

static char* dup_string(const std::string& s) {
    char* p = (char*)std::malloc(s.size() + 1);
    if (!p) return nullptr;
    std::memcpy(p, s.c_str(), s.size() + 1); // includes '\0'
    return p;
}

// If ShapeVector methods are not const in MemryX headers, this avoids
// "discard qualifiers" issues by copying the ShapeVector first.
static std::vector<int64_t> shapevector_to_chlast_dims(const MX::Types::ShapeVector& sv) {
    MX::Types::ShapeVector tmp = sv; // copy (non-const)
    return tmp.chlast_shape();       // [H, W, Z, C] 
}

static bool alloc_io_arrays(io_info* info) {
    if (!info) return false;

    info->input_names      = (char**)std::calloc(info->num_inputs,  sizeof(char*));
    info->output_names     = (char**)std::calloc(info->num_outputs, sizeof(char*));
    info->input_shapes     = (size_t**)std::calloc(info->num_inputs,  sizeof(size_t*));
    info->output_shapes    = (size_t**)std::calloc(info->num_outputs, sizeof(size_t*));
    info->input_ranks      = (size_t*)std::calloc(info->num_inputs,  sizeof(size_t));
    info->output_ranks     = (size_t*)std::calloc(info->num_outputs, sizeof(size_t));
    info->input_datatypes  = (tensor_data_type*)std::calloc(info->num_inputs,  sizeof(tensor_data_type));
    info->output_datatypes = (tensor_data_type*)std::calloc(info->num_outputs, sizeof(tensor_data_type));

    return info->input_names && info->output_names &&
           info->input_shapes && info->output_shapes &&
           info->input_ranks && info->output_ranks &&
           info->input_datatypes && info->output_datatypes;
}

// -----------------------------
// JSON -> io_info
// -----------------------------

// io_info* initialize_io_info(const char *json) {
//     if (!json) {
//         std::printf("initialize_io_info: json is null\n");
//         return nullptr;
//     }

//     yyjson_doc* doc = yyjson_read(json, std::strlen(json), 0);
//     if (!doc) {
//         std::printf("initialize_io_info: yyjson_read failed\n");
//         return nullptr;
//     }

//     yyjson_val* root = yyjson_doc_get_root(doc);
//     if (!root || !yyjson_is_obj(root)) {
//         std::printf("initialize_io_info: root is not an object\n");
//         yyjson_doc_free(doc);
//         return nullptr;
//     }

//     yyjson_val* inputs  = yyjson_obj_get(root, "Inputs");
//     yyjson_val* outputs = yyjson_obj_get(root, "Outputs");
//     if (!inputs || !outputs || !yyjson_is_arr(inputs) || !yyjson_is_arr(outputs)) {
//         std::printf("initialize_io_info: couldn't find Inputs/Outputs arrays in JSON\n");
//         yyjson_doc_free(doc);
//         return nullptr;
//     }

//     io_info* info = (io_info*)std::calloc(1, sizeof(io_info));
//     if (!info) {
//         yyjson_doc_free(doc);
//         return nullptr;
//     }

//     info->num_inputs  = (size_t)yyjson_arr_size(inputs);
//     info->num_outputs = (size_t)yyjson_arr_size(outputs);

//     if (!alloc_io_arrays(info)) {
//         yyjson_doc_free(doc);
//         free_io_info(info);
//         return nullptr;
//     }

//     // Inputs
//     for (size_t i = 0; i < info->num_inputs; ++i) {
//         yyjson_val* input = yyjson_arr_get(inputs, (yyjson_arr_iter_idx)i);
//         if (!input || !yyjson_is_obj(input)) {
//             yyjson_doc_free(doc);
//             free_io_info(info);
//             std::printf("initialize_io_info: Inputs[%zu] invalid\n", i);
//             return nullptr;
//         }

//         yyjson_val* namev = yyjson_obj_get(input, "Name");
//         yyjson_val* shape = yyjson_obj_get(input, "Shape");
//         yyjson_val* dtype = yyjson_obj_get(input, "DataType");

//         const char* name_str = (namev && yyjson_is_str(namev)) ? yyjson_get_str(namev) : nullptr;
//         if (!name_str || !shape || !yyjson_is_arr(shape) || !dtype || !yyjson_is_num(dtype)) {
//             yyjson_doc_free(doc);
//             free_io_info(info);
//             std::printf("initialize_io_info: Inputs[%zu] missing Name/Shape/DataType\n", i);
//             return nullptr;
//         }

//         info->input_names[i] = dup_cstr(name_str);
//         if (!info->input_names[i]) {
//             yyjson_doc_free(doc);
//             free_io_info(info);
//             return nullptr;
//         }

//         info->input_ranks[i] = (size_t)yyjson_arr_size(shape);
//         if (info->input_ranks[i] == 0) {
//             yyjson_doc_free(doc);
//             free_io_info(info);
//             std::printf("initialize_io_info: Inputs[%zu] shape rank is 0\n", i);
//             return nullptr;
//         }

//         info->input_shapes[i] = (size_t*)std::calloc(info->input_ranks[i], sizeof(size_t));
//         if (!info->input_shapes[i]) {
//             yyjson_doc_free(doc);
//             free_io_info(info);
//             return nullptr;
//         }

//         for (size_t j = 0; j < info->input_ranks[i]; ++j) {
//             yyjson_val* dim = yyjson_arr_get(shape, (yyjson_arr_iter_idx)j);
//             if (!dim || !yyjson_is_num(dim)) {
//                 yyjson_doc_free(doc);
//                 free_io_info(info);
//                 std::printf("initialize_io_info: Inputs[%zu].Shape[%zu] invalid\n", i, j);
//                 return nullptr;
//             }
//             int64_t v = yyjson_get_sint(dim);
//             if (v <= 0) v = 1; // workaround for dynamic/unknown dims
//             info->input_shapes[i][j] = (size_t)v;
//         }

//         info->input_datatypes[i] = (tensor_data_type)yyjson_get_sint(dtype);
//     }

//     // Outputs
//     for (size_t i = 0; i < info->num_outputs; ++i) {
//         yyjson_val* output = yyjson_arr_get(outputs, (yyjson_arr_iter_idx)i);
//         if (!output || !yyjson_is_obj(output)) {
//             yyjson_doc_free(doc);
//             free_io_info(info);
//             std::printf("initialize_io_info: Outputs[%zu] invalid\n", i);
//             return nullptr;
//         }

//         yyjson_val* namev = yyjson_obj_get(output, "Name");
//         yyjson_val* shape = yyjson_obj_get(output, "Shape");
//         yyjson_val* dtype = yyjson_obj_get(output, "DataType");

//         const char* name_str = (namev && yyjson_is_str(namev)) ? yyjson_get_str(namev) : nullptr;
//         if (!name_str || !shape || !yyjson_is_arr(shape) || !dtype || !yyjson_is_num(dtype)) {
//             yyjson_doc_free(doc);
//             free_io_info(info);
//             std::printf("initialize_io_info: Outputs[%zu] missing Name/Shape/DataType\n", i);
//             return nullptr;
//         }

//         info->output_names[i] = dup_cstr(name_str);
//         if (!info->output_names[i]) {
//             yyjson_doc_free(doc);
//             free_io_info(info);
//             return nullptr;
//         }

//         info->output_ranks[i] = (size_t)yyjson_arr_size(shape);
//         if (info->output_ranks[i] == 0) {
//             yyjson_doc_free(doc);
//             free_io_info(info);
//             std::printf("initialize_io_info: Outputs[%zu] shape rank is 0\n", i);
//             return nullptr;
//         }

//         info->output_shapes[i] = (size_t*)std::calloc(info->output_ranks[i], sizeof(size_t));
//         if (!info->output_shapes[i]) {
//             yyjson_doc_free(doc);
//             free_io_info(info);
//             return nullptr;
//         }

//         for (size_t j = 0; j < info->output_ranks[i]; ++j) {
//             yyjson_val* dim = yyjson_arr_get(shape, (yyjson_arr_iter_idx)j);
//             if (!dim || !yyjson_is_num(dim)) {
//                 yyjson_doc_free(doc);
//                 free_io_info(info);
//                 std::printf("initialize_io_info: Outputs[%zu].Shape[%zu] invalid\n", i, j);
//                 return nullptr;
//             }
//             int64_t v = yyjson_get_sint(dim);
//             if (v <= 0) v = 1;
//             info->output_shapes[i][j] = (size_t)v;
//         }

//         info->output_datatypes[i] = (tensor_data_type)yyjson_get_sint(dtype);
//     }

//     yyjson_doc_free(doc);
//     return info;
// }

// -----------------------------
// MemryX model_info -> io_info
// -----------------------------

io_info* initialize_io_info_from_model_info(MX::Types::MxModelInfo &model_info) {
    io_info* info = (io_info*)std::calloc(1, sizeof(io_info));
    if (!info) return nullptr;

    // Defensive: clamp negative counts to 0
    const int nin  = (model_info.num_in_featuremaps  < 0) ? 0 : model_info.num_in_featuremaps;
    const int nout = (model_info.num_out_featuremaps < 0) ? 0 : model_info.num_out_featuremaps;

    info->num_inputs  = (size_t)nin;
    info->num_outputs = (size_t)nout;

    if (!alloc_io_arrays(info)) {
        free_io_info(info);
        return nullptr;
    }

    // Inputs
    for (size_t i = 0; i < info->num_inputs; ++i) {
        if (i >= model_info.input_layer_names.size() || i >= model_info.in_featuremap_shapes.size()) {
            free_io_info(info);
            return nullptr;
        }

        info->input_names[i] = dup_string(model_info.input_layer_names[i]);
        if (!info->input_names[i]) {
            free_io_info(info);
            return nullptr;
        }

        // Use ShapeVector -> vector dims, do NOT assume 4 or rely on operator[]
        std::vector<int64_t> dims = shapevector_to_chlast_dims(model_info.in_featuremap_shapes[i]);
        if (dims.empty()) {
            free_io_info(info);
            return nullptr;
        }

        info->input_ranks[i] = dims.size();
        info->input_shapes[i] = (size_t*)std::calloc(info->input_ranks[i], sizeof(size_t));
        if (!info->input_shapes[i]) {
            free_io_info(info);
            return nullptr;
        }

        for (size_t d = 0; d < dims.size(); ++d) {
            int64_t v = dims[d];
            if (v <= 0) v = 1;
            info->input_shapes[i][d] = (size_t)v;
        }

        // v1 assumption (float). If MemryX exposes dtype, wire it later.
        info->input_datatypes[i] = DATA_TYPE_FLOAT;
    }

    // Outputs
    for (size_t i = 0; i < info->num_outputs; ++i) {
        if (i >= model_info.output_layer_names.size() || i >= model_info.out_featuremap_shapes.size()) {
            free_io_info(info);
            return nullptr;
        }

        info->output_names[i] = dup_string(model_info.output_layer_names[i]);
        if (!info->output_names[i]) {
            free_io_info(info);
            return nullptr;
        }

        std::vector<int64_t> dims = shapevector_to_chlast_dims(model_info.out_featuremap_shapes[i]);
        if (dims.empty()) {
            free_io_info(info);
            return nullptr;
        }

        info->output_ranks[i] = dims.size();
        info->output_shapes[i] = (size_t*)std::calloc(info->output_ranks[i], sizeof(size_t));
        if (!info->output_shapes[i]) {
            free_io_info(info);
            return nullptr;
        }

        for (size_t d = 0; d < dims.size(); ++d) {
            int64_t v = dims[d];
            if (v <= 0) v = 1;
            info->output_shapes[i][d] = (size_t)v;
        }

        info->output_datatypes[i] = DATA_TYPE_FLOAT;
    }

    return info;
}


io_info* initialize_io_info_from_model_infos(const MX::Types::MxModelInfo& in_info,
                                             const MX::Types::MxModelInfo& out_info) {
    io_info* info = (io_info*)std::calloc(1, sizeof(io_info));
    if (!info) return nullptr;

    const int nin  = (in_info.num_in_featuremaps  < 0) ? 0 : in_info.num_in_featuremaps;
    const int nout = (out_info.num_out_featuremaps < 0) ? 0 : out_info.num_out_featuremaps;

    info->num_inputs  = (size_t)nin;
    info->num_outputs = (size_t)nout;

    if (!alloc_io_arrays(info)) {
        free_io_info(info);
        return nullptr;
    }

    // -----------------
    // Inputs (from in_info)
    // -----------------
    for (size_t i = 0; i < info->num_inputs; ++i) {
        if (i >= in_info.input_layer_names.size() || i >= in_info.in_featuremap_shapes.size()) {
            free_io_info(info);
            return nullptr;
        }

        info->input_names[i] = dup_string(in_info.input_layer_names[i]);
        if (!info->input_names[i]) {
            free_io_info(info);
            return nullptr;
        }

        std::vector<int64_t> dims = shapevector_to_chlast_dims(in_info.in_featuremap_shapes[i]);
        if (dims.empty()) {
            free_io_info(info);
            return nullptr;
        }

        info->input_ranks[i] = dims.size();
        info->input_shapes[i] = (size_t*)std::calloc(info->input_ranks[i], sizeof(size_t));
        if (!info->input_shapes[i]) {
            free_io_info(info);
            return nullptr;
        }

        for (size_t d = 0; d < dims.size(); ++d) {
            int64_t v = dims[d];
            if (v <= 0) v = 1;
            info->input_shapes[i][d] = (size_t)v;
        }

        info->input_datatypes[i] = DATA_TYPE_FLOAT; // v1 assumption
    }

    // -----------------
    // Outputs (from out_info)
    // -----------------
    for (size_t i = 0; i < info->num_outputs; ++i) {
        if (i >= out_info.output_layer_names.size() || i >= out_info.out_featuremap_shapes.size()) {
            free_io_info(info);
            return nullptr;
        }

        info->output_names[i] = dup_string(out_info.output_layer_names[i]);
        if (!info->output_names[i]) {
            free_io_info(info);
            return nullptr;
        }

        std::vector<int64_t> dims = shapevector_to_chlast_dims(out_info.out_featuremap_shapes[i]);
        if (dims.empty()) {
            free_io_info(info);
            return nullptr;
        }

        info->output_ranks[i] = dims.size();
        info->output_shapes[i] = (size_t*)std::calloc(info->output_ranks[i], sizeof(size_t));
        if (!info->output_shapes[i]) {
            free_io_info(info);
            return nullptr;
        }

        for (size_t d = 0; d < dims.size(); ++d) {
            int64_t v = dims[d];
            if (v <= 0) v = 1;
            info->output_shapes[i][d] = (size_t)v;
        }

        info->output_datatypes[i] = DATA_TYPE_FLOAT; // v1 assumption
    }

    return info;
}

// -----------------------------
// Print Model Info
// -----------------------------

void print_io_info(io_info *info) {
    if (!info) {
        std::printf("IO Information: <null>\n");
        return;
    }
    std::printf("IO Information:\n");
    std::printf("Number of inputs: %zu\n", info->num_inputs);
    std::printf("Number of outputs: %zu\n", info->num_outputs);

    std::printf("Inputs:\n");
    for (size_t i = 0; i < info->num_inputs; i++) {
        const char* nm = info->input_names ? info->input_names[i] : nullptr;
        std::printf("Name: %s, Rank: %zu, Shape: [", nm ? nm : "<null>", info->input_ranks[i]);
        for (size_t j = 0; j < info->input_ranks[i]; j++) {
            std::printf("%zu", info->input_shapes[i][j]);
            if (j + 1 < info->input_ranks[i]) std::printf(", ");
        }
        std::printf("], Data type: %d\n", (int)info->input_datatypes[i]);
    }

    std::printf("Outputs:\n");
    for (size_t i = 0; i < info->num_outputs; i++) {
        const char* nm = info->output_names ? info->output_names[i] : nullptr;
        std::printf("Name: %s, Rank: %zu, Shape: [", nm ? nm : "<null>", info->output_ranks[i]);
        for (size_t j = 0; j < info->output_ranks[i]; j++) {
            std::printf("%zu", info->output_shapes[i][j]);
            if (j + 1 < info->output_ranks[i]) std::printf(", ");
        }
        std::printf("], Data type: %d\n", (int)info->output_datatypes[i]);
    }
}



// -----------------------------
// Free Model Info
// -----------------------------

void free_io_info(io_info *info) {
    if (!info) return;

    if (info->input_names) {
        for (size_t i = 0; i < info->num_inputs; i++) {
            std::free(info->input_names[i]);
        }
        std::free(info->input_names);
    }

    if (info->input_shapes) {
        for (size_t i = 0; i < info->num_inputs; i++) {
            std::free(info->input_shapes[i]);
        }
        std::free(info->input_shapes);
    }

    if (info->output_names) {
        for (size_t i = 0; i < info->num_outputs; i++) {
            std::free(info->output_names[i]);
        }
        std::free(info->output_names);
    }

    if (info->output_shapes) {
        for (size_t i = 0; i < info->num_outputs; i++) {
            std::free(info->output_shapes[i]);
        }
        std::free(info->output_shapes);
    }

    std::free(info->input_ranks);
    std::free(info->output_ranks);
    std::free(info->input_datatypes);
    std::free(info->output_datatypes);

    std::free(info);
}