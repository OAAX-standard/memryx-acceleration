#include "runtime_core.hpp"
#include "runtime_utils.hpp"
#include "runtime_ioinfo.hpp"
#include "memx/MxAccl.h"


static MX::Runtime::MxAccl *accl = NULL;
static MX::Types::MxModelInfo model_info;
static std::vector<float*> input_data;
static std::vector<float*> output_data;

static vector<bool> input_transposed;

static int model_id = 0;    // TODO: make it configurable
static int stream_id = 0;   // TODO: make it configurable

static tensors_struct local_output_tensors;
static io_info *info = NULL;

#ifdef __cplusplus
extern "C" {
#endif

int runtime_initialization_with_args(int length, const char **keys, const void **values){
    printf("Initialization with %d arguments.\n", length);
    
    // Look for an argument with the key "json"
    for (int i = 0; i < length; i++){
        if (strcmp(keys[i], "json") == 0){
            const char *json = (const char *)values[i];
            info = initialize_io_info(json);
            if (info == NULL){
                printf("Error: cannot initialize the io_info structure\n");
                return 1;
            }
            break;
        }
    }

    if (info == NULL){
        printf("Error: cannot find the JSON argument\n");
        return 1;
    }

#ifdef DEBUG
    print_io_info(info);
#endif

    return 0;
}

int runtime_initialization(){
    return 0;
}

int runtime_model_loading(const char *file_path){
    printf("Loading model: `%s`\n", file_path);

    accl = new MX::Runtime::MxAccl(file_path);
    model_info = accl->get_model_info(model_id);

    // Allocate output tensors
    if(info == NULL)
        info = initialize_io_info_from_model_info(model_info);

    allocate_output_tensors(&local_output_tensors, info);

    // Debug IO information
#ifdef DEBUG
    print_model_info(model_info);
#endif

    accl->start(true);

    return 0;
}

int runtime_inference_execution(tensors_struct *input_tensors, tensors_struct *output_tensors){
    printf("Inference\n");
    // Check if all inputs are FLOATS
    for (size_t i = 0; i < input_tensors->num_tensors; i++){
        if (input_tensors->data_types[i] != DATA_TYPE_FLOAT){
            printf("Error: input tensor data type is not FLOAT\n");
            return 3;
        }
    }
    for (size_t i = 0; i < input_tensors->num_tensors; i++){
        if(input_needs_transpose(i, model_info, input_tensors)){
            float *transposed_data = transpose_input_data(i, input_tensors);
            if (transposed_data == NULL){
                printf("Error: cannot transpose the input data\n");
                return 1;
            }
            input_data.push_back(transposed_data);
            input_transposed.push_back(true);
        } else {
            input_data.push_back((float *) input_tensors->data[i]);
            input_transposed.push_back(false);
        }
    }

    for (size_t i = 0; i < local_output_tensors.num_tensors; i++)
        output_data.push_back((float *)local_output_tensors.data[i]);

    // Perform the inference on the accelerator
    accl->send_input(input_data, model_id, stream_id, false);
    accl->receive_output(output_data, model_id, stream_id, false);

    // Free the input data
    for (size_t i = 0; i < input_data.size(); i++){
        if (input_transposed[i])
            free(input_data[i]);
    }

    // Transpose the output data if needed
    for (size_t i = 0; i < output_data.size(); i++){
        if(output_needs_transpose(i, model_info, &local_output_tensors)){
            float *transposed_data = transpose_output_data(i, &local_output_tensors );
            if (transposed_data == NULL){
                printf("Error: cannot transpose the output data\n");
                return 1;
            }
            // Free the original output data
            free(local_output_tensors.data[i]);
            // Point to the transposed data
            local_output_tensors.data[i] = (void *)transposed_data;
        }
    }

    input_transposed.clear();
    input_data.clear();

    *output_tensors = local_output_tensors;

    return 0;
}

int runtime_inference_cleanup(){
    printf("Cleanup\n");

    output_data.clear();

    return 0;
}

int runtime_destruction(){
    printf("Destruction\n");

    free_tensors_struct(&local_output_tensors);
    runtime_inference_cleanup();
    free_io_info(info);
    delete accl;

    return 0;
}

const char *runtime_error_message() {
    return "Check the stdout for the error message.";
}

const char *runtime_version() {
    return MEMX_LIBRARY_VERSION;
}

const char *runtime_name() {
    return "MXA";
}

#ifdef __cplusplus
}
#endif