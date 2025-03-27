#ifndef RUNTIME_UTILS_HPP
#define RUNTIME_UTILS_HPP

#include "runtime_ioinfo.hpp"
#include "memx/MxAccl.h"


/**
 * @brief Print the model information.
 * 
 * @param model_info The model information to print.
*/
void print_model_info(MX::Types::MxModelInfo &model_info);

/**
 * @brief Check if the input needs to be transposed from NCHW to NHWC.
 * 
 * @param input_index The index of the input tensor.
 * @param model_info The model information.
 * @param input_tensors The input tensors.
 * 
 * @return True if the input needs to be transposed, false otherwise.
*/
bool input_needs_transpose(int input_index, MX::Types::MxModelInfo &model_info, tensors_struct *input_tensors);

/**
 * @brief Transpose the input data from NCHW to NHWC.
 * 
 * @param input_index The index of the input tensor.
 * @param input_tensors The input tensors.
 * 
 * @warning The returned data must be freed by the caller.
 * @warning The returned data can be NULL for invalid input tensors. 
 * 
 * @return The transposed input data.
*/
float *transpose_input_data(int input_index, tensors_struct *input_tensors);

/**
 * @brief Check if the output needs to be transposed from NHWC to NCHW.
 * 
 * @param output_index The index of the output tensor.
 * @param model_info The model information.
 * @param output_tensors The output tensors.
 * 
 * @return True if the output needs to be transposed, false otherwise.
*/
bool output_needs_transpose(int output_index, MX::Types::MxModelInfo &model_info, tensors_struct *output_tensors);

/**
 * @brief Transpose the output data from NHWC to NCHW.
 * 
 * @param output_index The index of the output tensor.
 * @param output_tensors The output tensors.
 * 
 * @warning The returned data must be freed by the caller.
 * @warning The returned data can be NULL for invalid output tensors. 
 * 
 * @return The transposed output data.
*/
float *transpose_output_data(int output_index, tensors_struct *output_tensors);

/**
 * @brief Allocate the output tensors.
 * It sets all fields of the output_tensors structure except the data field, which is just allocated.
 * 
 * @param output_tensors The output tensors to allocate.
 * @param info The io_info structure.
 */
void allocate_output_tensors(tensors_struct *output_tensors, io_info *info);


/**
 * @brief Free the tensors_struct structure.
 * @note This function does not free the variable itself.
 * 
 * @param tensors The tensors_struct structure to free.
 */
void free_tensors_struct(tensors_struct *tensors);



#endif