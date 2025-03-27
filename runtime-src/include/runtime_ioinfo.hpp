#ifndef RUNTIME_UTILS_HEADER
#define RUNTIME_UTILS_HEADER

#include "runtime_core.hpp"
#include "yyjson.h"
#include "memx/MxAccl.h"

#include <stddef.h>
#include <stdlib.h>

typedef struct io_info {
    // number of inputs and outputs
    size_t num_inputs;
    size_t num_outputs;
    // names
    char **input_names;
    char **output_names;
    // shapes
    size_t** input_shapes;
    size_t** output_shapes;
    // ranks
    size_t* input_ranks;
    size_t* output_ranks;
    // datatypes
    tensor_data_type* input_datatypes;
    tensor_data_type* output_datatypes;
} io_info;

/**
 * @brief Initialize the io_info structure from the JSON string.
 * 
 * @param json The JSON string containing the input and output information.
 * The JSON string should have the following format:
 * {
 *  "Inputs": [
 *    {"Name": "input_1", "Shape": [1, 28, 28, 1], "DataType": 1},
 *     ...
 *  ], 
 * "Outputs": [
 *     {"Name": "output_1", "Shape": [1, 10], "DataType": 1},
 *     ...
 *  ]
 * }
 * 
 * @return The io_info structure.
*/
io_info *initialize_io_info(const char *json);


/**
 * @brief Initialize the io_info structure from the model_info structure.
 * 
 * @param model_info The model_info structure.
 * 
 * @return The io_info structure.
 */
io_info* initialize_io_info_from_model_info(MX::Types::MxModelInfo &model_info);

/**
 * @brief Print the io_info structure.
 * 
 * @param info The io_info structure to print.
 */
void print_io_info(io_info *info);

/**
 * @brief Free the io_info structure.
 *
 * @param info The io_info structure to free.
 */
void free_io_info(io_info *info);

#endif //RUNTIME_UTILS_HEADER
