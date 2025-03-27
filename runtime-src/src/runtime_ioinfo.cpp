#include "runtime_ioinfo.hpp"

io_info* initialize_io_info(const char *json){
    io_info *info = (io_info *) malloc(sizeof(io_info));
    yyjson_doc *doc = yyjson_read(json, strlen(json), 0);
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *inputs = yyjson_obj_get(root, "Inputs");
    yyjson_val *outputs = yyjson_obj_get(root, "Outputs");
    if (!inputs || !outputs){
        printf("Error: couldn't find the Inputs and Ouputs in the JSON\n");
        yyjson_doc_free(doc);
        free(info);
        return NULL;
    }
    
    // Allocate memory for the io_info structure
    info->num_inputs = yyjson_arr_size(inputs);
    info->num_outputs = yyjson_arr_size(outputs);
    info->input_names = (char **)malloc(info->num_inputs * sizeof(char *));
    info->output_names = (char **)malloc(info->num_outputs * sizeof(char *));
    info->input_shapes = (size_t **)malloc(info->num_inputs * sizeof(size_t *));
    info->output_shapes = (size_t **)malloc(info->num_outputs * sizeof(size_t *));
    info->input_ranks = (size_t *)malloc(info->num_inputs * sizeof(size_t));
    info->output_ranks = (size_t *)malloc(info->num_outputs * sizeof(size_t));
    info->input_datatypes = (tensor_data_type *)malloc(info->num_inputs * sizeof(tensor_data_type));
    info->output_datatypes = (tensor_data_type *)malloc(info->num_outputs * sizeof(tensor_data_type));

    // Inputs
    for (size_t i = 0; i < info->num_inputs; i++){
        yyjson_val *input = yyjson_arr_get(inputs, i);
        yyjson_val *name = yyjson_obj_get(input, "Name");
        yyjson_val *shape = yyjson_obj_get(input, "Shape");
        yyjson_val *datatype = yyjson_obj_get(input, "DataType");
        // Name
        const char *name_str = yyjson_get_str(name);
        size_t name_len = strlen(name_str);
        info->input_names[i] = (char *)malloc((name_len + 1) * sizeof(char));
        memcpy(info->input_names[i], name_str, name_len);
        info->input_names[i][name_len] = '\0';
        // Rank
        info->input_ranks[i] = yyjson_arr_size(shape);
        // Shape
        info->input_shapes[i] = (size_t *)malloc(info->input_ranks[i] * sizeof(size_t));
        for (size_t j = 0; j < info->input_ranks[i]; j++){
            yyjson_val *dim = yyjson_arr_get(shape, j);
            info->input_shapes[i][j] = yyjson_get_int(dim);
            // TODO: workaround for dynamic shape
            if (info->input_shapes[i][j] == 0){
                info->input_shapes[i][j] = 1;
            }
        }
        // Data type
        info->input_datatypes[i] = (tensor_data_type) yyjson_get_int(datatype);
    }

    // Outputs
    for (size_t i = 0; i < info->num_outputs; i++){
        yyjson_val *output = yyjson_arr_get(outputs, i);
        yyjson_val *name = yyjson_obj_get(output, "Name");
        yyjson_val *shape = yyjson_obj_get(output, "Shape");
        yyjson_val *datatype = yyjson_obj_get(output, "DataType");
        // Name
        const char *name_str = yyjson_get_str(name);
        size_t name_len = strlen(name_str);
        info->output_names[i] = (char *)malloc((name_len + 1) * sizeof(char));
        memcpy(info->output_names[i], name_str, name_len);
        info->output_names[i][name_len] = '\0';
        // Rank
        info->output_ranks[i] = yyjson_arr_size(shape);
        // Shape
        info->output_shapes[i] = (size_t *)malloc(info->output_ranks[i] * sizeof(size_t));
        for (size_t j = 0; j < info->output_ranks[i]; j++){
            yyjson_val *dim = yyjson_arr_get(shape, j);
            info->output_shapes[i][j] = yyjson_get_int(dim);
            // TODO: workaround for dynamic shape
            if (info->output_shapes[i][j] == 0){
                info->output_shapes[i][j] = 1;
            }
        }
        // Data type
        info->output_datatypes[i] = (tensor_data_type) yyjson_get_int(datatype);
    }


    yyjson_doc_free(doc);
    return info;
}

io_info* initialize_io_info_from_model_info(MX::Types::MxModelInfo &model_info){
    io_info *info = (io_info *) malloc(sizeof(io_info));
    
    // Allocate memory for the io_info structure
    info->num_inputs = model_info.num_in_featuremaps;
    info->num_outputs = model_info.num_out_featuremaps;
    info->input_names = (char **)malloc(info->num_inputs * sizeof(char *));
    info->output_names = (char **)malloc(info->num_outputs * sizeof(char *));
    info->input_shapes = (size_t **)malloc(info->num_inputs * sizeof(size_t *));
    info->output_shapes = (size_t **)malloc(info->num_outputs * sizeof(size_t *));
    info->input_ranks = (size_t *)malloc(info->num_inputs * sizeof(size_t));
    info->output_ranks = (size_t *)malloc(info->num_outputs * sizeof(size_t));
    info->input_datatypes = (tensor_data_type *)malloc(info->num_inputs * sizeof(tensor_data_type));
    info->output_datatypes = (tensor_data_type *)malloc(info->num_outputs * sizeof(tensor_data_type));

    // Inputs
    for (size_t i = 0; i < info->num_inputs; i++){
        info->input_names[i] = (char *)malloc(strlen(model_info.input_layer_names[i]) * sizeof(char));
        memcpy(info->input_names[i], model_info.input_layer_names[i], strlen(model_info.input_layer_names[i]));
        info->input_ranks[i] = 4;
        info->input_shapes[i] = (size_t *)malloc(info->input_ranks[i] * sizeof(size_t));
        info->input_shapes[i][0] = model_info.in_featuremap_shapes[i][0];
        info->input_shapes[i][1] = model_info.in_featuremap_shapes[i][1];
        info->input_shapes[i][2] = model_info.in_featuremap_shapes[i][2];
        info->input_shapes[i][3] = model_info.in_featuremap_shapes[i][3];
        info->input_datatypes[i] = DATA_TYPE_FLOAT;
    }
    // Outputs
    for (size_t i = 0; i < info->num_outputs; i++){
        info->output_names[i] = (char *) malloc(strlen(model_info.output_layer_names[i]) * sizeof(char));
        memcpy(info->output_names[i], model_info.output_layer_names[i], strlen(model_info.output_layer_names[i]));
        info->output_ranks[i] = 4;
        info->output_shapes[i] = (size_t *) malloc(info->output_ranks[i] * sizeof(size_t));
        info->output_shapes[i][0] = model_info.out_featuremap_shapes[i][0];
        info->output_shapes[i][1] = model_info.out_featuremap_shapes[i][1];
        info->output_shapes[i][2] = model_info.out_featuremap_shapes[i][2];
        info->output_shapes[i][3] = model_info.out_featuremap_shapes[i][3];
        info->output_datatypes[i] = DATA_TYPE_FLOAT;
    }
    
    return info;
}


void print_io_info(io_info *info){
    printf("IO Information:\n");
    printf("Number of inputs: %zu\n", info->num_inputs);
    printf("Number of outputs: %zu\n", info->num_outputs);
    printf("Inputs:\n");
    for (size_t i = 0; i < info->num_inputs; i++){
        printf("Name: %s, Rank: %zu, Shape: [", info->input_names[i], info->input_ranks[i]);
        for (size_t j = 0; j < info->input_ranks[i]; j++){
            printf("%zu", info->input_shapes[i][j]);
            if (j < info->input_ranks[i] - 1){
                printf(", ");
            }
        }
        printf("], Data type: %d\n", info->input_datatypes[i]);
    }
    printf("Outputs:\n");
    for (size_t i = 0; i < info->num_outputs; i++){
        printf("Name: %s, Rank: %zu, Shape: [", info->output_names[i], info->output_ranks[i]);
        for (size_t j = 0; j < info->output_ranks[i]; j++){
            printf("%zu", info->output_shapes[i][j]);
            if (j < info->output_ranks[i] - 1){
                printf(", ");
            }
        }
        printf("], Data type: %d\n", info->output_datatypes[i]);
    }

}

void free_io_info(io_info *info){
    for (size_t i = 0; i < info->num_inputs; i++){
        free(info->input_names[i]);
        free(info->input_shapes[i]);
    }
    for (size_t i = 0; i < info->num_outputs; i++){
        free(info->output_names[i]);
        free(info->output_shapes[i]);
    }
    free(info->input_names);
    free(info->output_names);
    free(info->input_shapes);
    free(info->output_shapes);
    free(info->input_ranks);
    free(info->output_ranks);
    free(info->input_datatypes);
    free(info->output_datatypes);
    free(info);
}