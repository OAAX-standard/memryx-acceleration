#include "runtime_utils.hpp"
#include <iostream>

void print_model_info(MX::Types::MxModelInfo &model_info){
    std::cout << "\n******** Model Index : " << model_info.model_index << " ********\n";
    std::cout << "\nNum of in featuremaps : " << model_info.num_in_featuremaps << "\n";
    
    std::cout << "\nIn featureMap Shapes \n";
    for(int i; i<model_info.num_in_featuremaps ; ++i){
        std::cout << "Shape of featureMap : " << i+1 << "\n";
        std::cout << "Layer Name : " << model_info.input_layer_names[i] << "\n";
        std::cout << "H = " << model_info.in_featuremap_shapes[i][0] << "\n";
        std::cout << "W = " << model_info.in_featuremap_shapes[i][1] << "\n";
        std::cout << "Z = " << model_info.in_featuremap_shapes[i][2] << "\n";
        std::cout << "C = " << model_info.in_featuremap_shapes[i][3] << "\n";
    }

    std::cout << "\n\nNum of out featuremaps : " << model_info.num_out_featuremaps << "\n";
    std::cout << "\nOut featureMap Shapes \n";
    for(int i; i<model_info.num_out_featuremaps ; ++i){
        std::cout << "Shape of featureMap : " << i+1 << "\n";
        std::cout << "Layer Name : " << model_info.output_layer_names[i] << "\n";
        std::cout << "H = " << model_info.out_featuremap_shapes[i][0] << "\n";
        std::cout << "W = " << model_info.out_featuremap_shapes[i][1] << "\n";
        std::cout << "Z = " << model_info.out_featuremap_shapes[i][2] << "\n";
        std::cout << "C = " << model_info.out_featuremap_shapes[i][3] << "\n";
    }
        std::cout << "Done printing model info \n";
}

bool input_needs_transpose(int input_index, MX::Types::MxModelInfo &model_info, tensors_struct *input_tensors){
    // Make sure the input_index is within bounds
    if (input_index >= model_info.num_in_featuremaps){
        printf("Input index out of bounds 1\n");
        return false;
    }
    if (input_index >= input_tensors->num_tensors){
        printf("Input index out of bounds 2\n");
        return false;
    }
    // Check if the input tensor shape is the same as the model input shape or at least with ones in the beginning
    for(int i=input_tensors->ranks[input_index]-1; i >=0 ; i--){
        if(input_tensors->shapes[input_index][i] != model_info.in_featuremap_shapes[input_index][i]){
            if(input_tensors->shapes[input_index][i] != 1){
                return true;
            }
        }
    }
    return false;
}

float *transpose_input_data(int input_index, tensors_struct *input_tensors){
    // TODO: are these conditions really needed?
    // Make sure the tensor rank is 4
    if (input_tensors->ranks[input_index] != 4){
        printf("Input rank is not 4\n");
        return NULL;
    }
    // Make sure the tensor data type is float
    if (input_tensors->data_types[input_index] != DATA_TYPE_FLOAT){
        printf("Input data type is not float\n");
        return NULL;
    }
    // Make sure the batch size is 1
    if (input_tensors->shapes[input_index][0] != 1){
        printf("Batch size is not 1\n");
        return NULL;
    }
    // Compute tensor size
    size_t size = 1;
    for (size_t i = 0; i < input_tensors->ranks[input_index]; i++)
        size *= input_tensors->shapes[input_index][i];
    // Allocate memory for the transposed data
    float *transposed_data = (float *)malloc(size * sizeof(float));
    float *data = (float *)input_tensors->data[input_index];
    // Move tensor format from NCHW to NHWC
    size_t C = input_tensors->shapes[input_index][1];
    size_t H = input_tensors->shapes[input_index][2];
    size_t W = input_tensors->shapes[input_index][3];
    for (size_t c = 0; c < C; c++){
        for (size_t h = 0; h < H; h++){
            for (size_t w = 0; w < W; w++){
                transposed_data[h * W * C + w * C + c] = data[c * H * W + h * W + w];
            }
        }
    }
    return transposed_data;
}

bool output_needs_transpose(int output_index, MX::Types::MxModelInfo &model_info, tensors_struct *output_tensors){
    // Make sure the output_index is within bounds
    if (output_index >= model_info.num_out_featuremaps){
        printf("Output index out of bounds 1\n");
        return false;
    }
    if (output_index >= output_tensors->num_tensors){
        printf("Output index out of bounds 2\n");
        return false;
    }
    // Check if the output tensor shape is the same as the model output shape or at least with ones in the beginning
    for(int i=output_tensors->ranks[output_index]-1; i >=0 ; i--){
        if(output_tensors->shapes[output_index][i] != model_info.out_featuremap_shapes[output_index][i]){
            if(output_tensors->shapes[output_index][i] != 1){
                return true;
            }
        }
    }
    return false;
}

float *transpose_output_data(int output_index, tensors_struct *output_tensors){
    // TODO: are these conditions really needed?
    // Make sure the tensor rank is 4
    if (output_tensors->ranks[output_index] != 4){
        printf("Output rank is not 4\n");
        return NULL;
    }
    // Make sure the tensor data type is float
    if (output_tensors->data_types[output_index] != DATA_TYPE_FLOAT){
        printf("Output data type is not float\n");
        return NULL;
    }
    // Make sure the batch size is 1
    if (output_tensors->shapes[output_index][0] != 1){
        printf("Batch size is not 1\n");
        return NULL;
    }
    // Compute tensor size
    size_t size = 1;
    for (size_t i = 0; i < output_tensors->ranks[output_index]; i++)
        size *= output_tensors->shapes[output_index][i];
    // Allocate memory for the transposed data
    float *transposed_data = (float *)malloc(size * sizeof(float));
    float *data = (float *)output_tensors->data[output_index];
    // Move tensor format from NHWC to NCHW 
    size_t C = output_tensors->shapes[output_index][1];
    size_t H = output_tensors->shapes[output_index][2];
    size_t W = output_tensors->shapes[output_index][3];
    for (size_t c = 0; c < C; c++){
        for (size_t h = 0; h < H; h++){
            for (size_t w = 0; w < W; w++){
                transposed_data[c * H * W + h * W + w] = data[h * W * C + w * C + c];
            }
        }
    }

    return transposed_data;
}

void allocate_output_tensors(tensors_struct *output_tensors, io_info *info){
    printf("Allocating output tensors\n");

    output_tensors->num_tensors = info->num_outputs;
    output_tensors->data = (void **)malloc(output_tensors->num_tensors * sizeof(void *));
    output_tensors->data_types = (tensor_data_type *)malloc(output_tensors->num_tensors * sizeof(tensor_data_type));
    output_tensors->ranks = (size_t *)malloc(output_tensors->num_tensors * sizeof(size_t));
    output_tensors->shapes = (size_t **)malloc(output_tensors->num_tensors * sizeof(size_t *));
    output_tensors->names = (char **)malloc(output_tensors->num_tensors * sizeof(char *));

    // Copy names, data types, ranks, and shapes
    // Names
    for (size_t i = 0; i < output_tensors->num_tensors; i++){
        size_t name_len = strlen(info->output_names[i]);
        output_tensors->names[i] = (char *)malloc((name_len + 1) * sizeof(char));
        memcpy(output_tensors->names[i], info->output_names[i], name_len);
        output_tensors->names[i][name_len] = '\0';
    }
    // Data types
    memcpy(output_tensors->data_types, info->output_datatypes, output_tensors->num_tensors * sizeof(tensor_data_type));
    // Ranks
    memcpy(output_tensors->ranks, info->output_ranks, output_tensors->num_tensors * sizeof(size_t));
    // Shapes
    for (size_t i = 0; i < output_tensors->num_tensors; i++){
        output_tensors->shapes[i] = (size_t *)malloc(output_tensors->ranks[i] * sizeof(size_t));
        memcpy(output_tensors->shapes[i], info->output_shapes[i], output_tensors->ranks[i] * sizeof(size_t));
    }
    // Compute sizes
    size_t *sizes = (size_t *)malloc(output_tensors->num_tensors * sizeof(size_t));
    for (size_t i = 0; i < output_tensors->num_tensors; i++){
        sizes[i] = 1;
        for (size_t j = 0; j < output_tensors->ranks[i]; j++){
            sizes[i] *= output_tensors->shapes[i][j];
        }
    }
    // Allocate memory for data
    for (size_t i = 0; i < output_tensors->num_tensors; i++){
        output_tensors->data[i] = malloc(sizes[i] * sizeof(float)); // TODO: support other data types
        for (size_t j = 0; j < sizes[i]; j++){
            ((float *)output_tensors->data[i])[j] = -1.0f;
        }
    }

    free(sizes);
}

void free_tensors_struct(tensors_struct *tensors) {
    printf("Freeing tensors\n");
    if (tensors->data_types != NULL) {
        free(tensors->data_types);
        tensors->data_types = NULL;
    }

    if (tensors->ranks != NULL) {
        free(tensors->ranks);
        tensors->ranks = NULL;
    }

    if (tensors->data != NULL) {
        for (size_t i = 0; i < tensors->num_tensors; i++) {
            if (tensors->data[i] != NULL)
                free(tensors->data[i]);
        }
        free(tensors->data);
        tensors->data = NULL;
    }

    if (tensors->shapes != NULL) {
        for (size_t i = 0; i < tensors->num_tensors; i++) {
            if (tensors->shapes[i] != NULL)
                free(tensors->shapes[i]);
        }
        free(tensors->shapes);
        tensors->shapes = NULL;
    }

    if (tensors->names != NULL) {
        for (size_t i = 0; i < tensors->num_tensors; i++) {
            if (tensors->names[i] != NULL)
                free(tensors->names[i]);
        }
        free(tensors->names);
        tensors->names = NULL;
    }
    printf("Tensors freed\n");
}