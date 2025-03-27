/***************************************************************************//**
 * @note
 * Copyright (C) 2019-2024 MemryX Limited. All rights reserved.
 *
 ******************************************************************************/
#ifndef MEMX_H_
#define MEMX_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
 * version
 ******************************************************************************/

/**
 * @brief Library version can be obtained using command:
 *  $ strings libmemx.so | grep "libmemx version".
 * Library version 'x.y.z' represents and should be modified in case:
 *  x: new memryx device support,
 *  y: top level memx api changed,
 *  z: minor bug fix.
 */
#define MEMX_LIBRARY_VERSION "2.2.20"

/***************************************************************************//**
 * common
 ******************************************************************************/
#ifndef MEMX_COMMON_H_
#define MEMX_COMMON_H_
#if defined(__MSC_VER) || defined(WIN_EXPORTS)
  #define MEMX_API_EXPORT __declspec(dllexport)
  #define MEMX_API_IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
  #define MEMX_API_EXPORT __attribute__((visibility("default")))
  #define MEMX_API_IMPORT
#else
  #define MEMX_API_EXPORT
  #define MEMX_API_IMPORT
#endif
#ifndef unused
#define unused(x) (void)(x)
#endif
#endif /* MEMX_COMMON_H_ */

/***************************************************************************//**
 * status
 ******************************************************************************/
#ifndef MEMX_STATUS_H_
#define MEMX_STATUS_H_
/**
 * @brief Helper macro to check if there is no error occurs.
 */
#define memx_status_no_error(_status_) ((_status_) == MEMX_STATUS_OK)

/**
 * @brief Helper macro to check if there is any error occurs.
 */
#define memx_status_error(_status_) ((_status_) != MEMX_STATUS_OK)

/**
 * @brief Driver internal error code. '0' should always be used to indicate no
 * error occurs.
 */
typedef enum _memx_status {
  MEMX_STATUS_OK = 0,
  MEMX_STATUS_OTHERS = 1,
} memx_status;

#endif /* MEMX_STATUS_H_ */
/******************************************************************************/

/**
 * @brief MemryX device MX3(pre-prod):Cascade. Constant value which should be referenced
 * only and not be modified manually.
 */
#define MEMX_DEVICE_CASCADE (30)

/**
 * @brief MemryX device MX3:Cascade+. Constant value which should be referenced
 * only and not be modified manually.
 */
#define MEMX_DEVICE_CASCADE_PLUS (31)

/**
 * @brief Maximum number of model contexts can be stored within driver. Constant
 * value which should be referenced only and not be modified manually.
 */
#define MEMX_MODEL_MAX_NUMBER (32)

/**
 * @brief Maximum number of MPU device group contexts can be stored within
 * driver. Constant value which should be referenced only and not be modified
 * manually.
 */
#define MEMX_DEVICE_GROUP_MAX_NUMBER (4)

/**
 * @brief Option of `memx_set_ifmap_size()` and `memx_set_ofmap_size()` to
 * configure model input or output format to 32-bits floating point. By default,
 * input and output feature map should be configured using this option if model
 * input and output are using floating-point.
 */
#define MEMX_FMAP_FORMAT_FLOAT32 (5)

/**
 * @brief Option of `memx_set_ifmap_size()` and `memx_set_ofmap_size()` to
 * configure model input or output format to bfloat16. By default,
 * input and output feature map should be configured using this option if model
 * input and output are using floating-point.
 */
#define MEMX_FMAP_FORMAT_BF16 (4)

/**
 * @brief Option of `memx_set_ifmap_size()` and `memx_set_ofmap_size()` to
 * configure model input or output feature map format to raw byte array. Input
 * feature map should use this option for format such as RGB888, while output
 * feature map in general case should never use this option.
 */
#define MEMX_FMAP_FORMAT_RAW (2)

/**
 * @brief Option of `memx_set_ifmap_size()` and `memx_set_ofmap_size()` to
 * configure model input or output feature map format to MemryX proprietary
 * format group-bfloat-80. This option is only used for internal debug purpose
 * only.
 */
#define MEMX_FMAP_FORMAT_GBF80 (0)
#define MEMX_FMAP_FORMAT_GBF80_ROW_PAD (6)

/**
 * @brief Option of `memx_download_model()` to download weight memory only to
 * device. Can be used together with `MEMX_DOWNLOAD_TYPE_MODEL`.
 */
#define MEMX_DOWNLOAD_TYPE_WTMEM (1)

/**
 * @brief Option of `memx_download_model()` to download model only to device.
 * Can be used together with `MEMX_DOWNLOAD_TYPE_WTMEM`.
 */
#define MEMX_DOWNLOAD_TYPE_MODEL (2)

/**
 * @brief Option of `memx_download_model()` to download both weight memory and
 * model to device. The same effect as using `MEMX_DOWNLOAD_TYPE_WTMEM` and
 * `MEMX_DOWNLOAD_TYPE_MODEL` together.
 */
#define MEMX_DOWNLOAD_TYPE_WTMEM_AND_MODEL (3)

/**
 * @brief Option of `memx_config_mpu_group()` to set different MPU group
 */
#define MEMX_MPU_GROUP_CONFIG_ONE_GROUP_FOUR_MPUS  (0)
#define MEMX_MPU_GROUP_CONFIG_TWO_GROUP_TWO_MPUS   (1)
#define MEMX_MPU_GROUP_CONFIG_ONE_GROUP_ONE_MPU    (2)
#define MEMX_MPU_GROUP_CONFIG_ONE_GROUP_THREE_MPUS (3)
#define MEMX_MPU_GROUP_CONFIG_ONE_GROUP_TWO_MPUS   (4)

/**
 * @brief Acquires mutex lock from driver. To be noticed, this function will
 * not actually lock any operation but only acquire the unique mutex within
 * driver. Lock is only required in multi-threading or multi-processing case to
 * avoid driver access conflict when multiple processes trying to re-configure
 * the same hardware device. Also, this function will block until mutex is
 * acquired successfully. See `memx_trylock()` for non-blocking mutex lock.
 *
 * ~~~~~
 * // lock MPU device group 0
 * memx_lock(0);
 * printf("lock success.\n");
 * // always remember to release lock finally
 * memx_unlock(0);
 * ~~~~~
 *
 * @see Advanced Driver Usage: Example 3: Runtime Model Swap
 *
 * @param group_id            MPU device group ID
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_lock(uint8_t group_id);

/**
 * @brief Acquires mutex lock from driver. Apart from `memx_lock()`, this
 * function returns immediately if fail to acquire lock.
 *
 * ~~~~~
 * // do something only if lock MPU device group 0 successfully
 * if (memx_trylock(0) == 0) {
 *   printf("lock success.\n");
 *   // always remember to release lock finally
 *   memx_unlock(0);
 * } else {
 *   printf("lock failure.\n");
 * }
 * ~~~~~
 *
 * @param group_id            MPU device group ID
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_trylock(uint8_t group_id);

/**
 * @brief Releases the mutex lock acquired from driver. Be careful and not to
 * unexpectedly release mutex which is not locked by current process. Uses
 * `memx_lock()` and `memx_trylock()` to acquire lock first.
 *
 * @param group_id            MPU device group ID
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_unlock(uint8_t group_id);

/**
 * @brief Open new model context with specified MPU device driver and MPU device
 * group binding to it. Internal reference count increases if the same model
 * ID is re-used and opened multiple times. All operations later using this
 * model ID will use binded MPU device driver and MPU device group. MPU device
 * group binded can later be re-configured using `memx_reconfigure()` while MPU
 * device driver may not be re-configured once model context is created.
 * Currently, at most 32 model contexts (model ID = 0 ~ 31) can be stored
 * within driver.
 *
 * ~~~~~
 * // bind MPU device group 0 as MX3 to model 0
 * memx_status status = memx_open(0, 0, MEMX_DEVICE_CASCADE_PLUS);
 * if (status == 0) {
 *   printf("open success.\n");
 * } else {
 *   printf("open failure.\n");
 * }
 * // always remember to close device finally
 * memx_close(0);
 * ~~~~~
 *
 * @see Classification (Driver APIs): Basic Inference
 *
 * @param model_id            model ID
 * @param group_id            MPU device group ID
 * @param chip_gen            Deprecated, user don't care. device chip_gen was detected by driver
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_open(uint8_t model_id, uint8_t group_id, float chip_gen);

/**
 * @brief Close current connected MPU device group and destory model context
 * stored in driver. If model context is opened multiple times, calling close
 * will decrease internal reference count one each time. Only after reference
 * count goes down to zero will model context finally be released. So make sure
 * to use equivalent times of open and close to clean-up driver resources.
 * Otherwise, it is safe to call close model ID which is not opened or is closed
 * already. See `memx_open()` for more information.
 *
 * @param model_id            model ID
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_close(uint8_t model_id);

/**
 * @brief Runtime re-configure MPU device group binding to specified model ID.
 * Be sure to finish all on-going data transfer first before calling this
 * function.
 *
 * ~~~~~
 * // bind MPU device group 0 as MX3 to model 0
 * memx_status status = memx_open(0, 0, MEMX_DEVICE_CASCADE_PLUS);
 * if (status == 0) {
 *   // re-configure to bind MPU device group 1 to model 0
 *   status = memx_reconfigure(0, 1);
 * }
 * // always remember to close device finally
 * memx_close(0);
 * ~~~~~
 *
 * @param model_id            model ID
 * @param group_id            MPU device group ID
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_reconfigure(uint8_t model_id, uint8_t group_id);

/**
 * @brief General purpose operation. Reserved for future.
 *
 * @param model_id            model ID
 * @param cmd_id              command ID
 * @param data                command data
 * @param size                command data size
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_operation(uint8_t model_id, uint32_t cmd_id, void* data, uint32_t size);

/**
 * @brief Download DFP (Data Flow Program) weight memory to device. DFP file is
 * generated by MIX and should by default be named with extension `.dfp`. This
 * function download only weight memory with no model setting to device. Use
 * `memx_download_model()` with option instead to download both weight memory
 * and model to device.
 *
 * @param model_id            model ID
 * @param file_path           DFP file path
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_download_model_wtmem(uint8_t model_id, const char* file_path);

/**
 * @brief Download DFP (Data Flow Program) model to device. DFP file is
 * generated by MIX and should by default be named with extension `.dfp`. This
 * function download only model setting with no weight memory to device. Use
 * `memx_download_model()` with option instead to download both weight memory
 * and model to device.
 *
 * @param model_id            model ID
 * @param file_path           DFP file path
 * @param model_idx           model index within DFP file, give '0' if there is only one model compiled in DFP file
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_download_model_config(uint8_t model_id, const char* file_path, uint8_t model_idx);

/**
 * @brief Download weight memory and model to device based on given download
 * type selection. If both weight memory and model download are required, weight
 * memory download will be performed before model download. After model download
 * is completed,  input and output feature map shape will be configured to
 * driver to allocate resources automatically. After model download and before
 * data can be transferred to device, `memx_set_stream_enable()` must be called
 * to change driver internal state from idle to data-transfer.
 *
 * ~~~~~
 * // bind MPU device group 0 as MX3 to model 0
 * memx_status status = memx_open(0, 0, MEMX_DEVICE_CASCADE_PLUS);
 * if (status == 0) {
 *   // download weight memory and 1st model within DFP file to device
 *   status = memx_download_model(0, "model.dfp", 0,
 *     MEMX_DOWNLOAD_TYPE_WTMEM_AND_MODEL);
 * }
 * // always remember to close device finally
 * memx_close(0);
 * ~~~~~
 *
 * ~~~~~
 * // download weight memory and 1st model within DFP file to device separately
 * // same effect as example using type MEMX_DOWNLOAD_TYPE_WTMEM_AND_MODEL
 *
 * // bind MPU device group 0 as MX3 to model 0
 * memx_status status = memx_open(0, 0, MEMX_DEVICE_CASCADE_PLUS);
 * if (status == 0) {
 *   // download weight memory to device, model index within DFP will be ignored
 *   status = memx_download_model(0, "model.dfp", 0, MEMX_DOWNLOAD_TYPE_WTMEM);
 * }
 * if (status == 0) {
 *   // download 1st model within DFP file to device
 *   status = memx_download_model(0, "model.dfp", 0, MEMX_DOWNLOAD_TYPE_MODEL);
 * }
 * // always remember to close device finally
 * memx_close(0);
 * ~~~~~
 *
 * @param model_id            model ID
 * @param file_path           DFP file path
 * @param model_idx           model index within DFP file, give '0' if there is only one model compiled in DFP file.
 *                            model index will be ignored if download type selection is weight memory only
 * @param type                0: ignored, 1: weight memory only, 2: model only, 3: both weight memory and model
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_download_model(uint8_t model_id, const char* file_path, uint8_t model_idx, int32_t type);

/**
 * @brief Download firmware to device flash.
 *
 *
 * @param model_id            model ID
 * @param file_path           firmware file path
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_download_firmware(uint8_t model_id, const char *file_path);

/**
 * @brief Configure driver background encoding and decoding worker (thread)
 * number. By default, there is one worker for encoding and one worker for
 * decoding. Setting worker number will adjust the number of workers for both
 * encoding and decoding in the same time. It is recommend to keep it as default
 * value, otherwise too many workers with frequently context switch might slow
 * down overall system performance.
 *
 * @param model_id            model ID
 * @param worker_number       worker number
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_set_worker_number(uint8_t model_id, int32_t worker_number);

/**
 * @brief Enable all input and output data to be transferred to device of
 * specified model. When model context is first initialized, all data transfers
 * are blocked (but model download is allowed) to avoid data being transferred
 * to device before model download is finished.
 *
 * ~~~~~
 * // bind MPU device group 0 as MX3 to model 0
 * memx_status status = memx_open(0, 0, MEMX_DEVICE_CASCADE_PLUS);
 * if (status == 0) {
 *   // download weight memory and 1st model within DFP file to device
 *   status = memx_download_model(0, "model.dfp", 0,
 *     MEMX_DOWNLOAD_TYPE_WTMEM_AND_MODEL);
 * }
 * if (status == 0) {
 *   // enable all data being transferred to device
 *   status = memx_set_stream_enable(0, 0);
 * }
 * // always remember to close device finally
 * memx_close(0);
 * ~~~~~
 *
 * @param model_id            model ID
 * @param wait                wait until driver state change from idle to data-transfer complete
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_set_stream_enable(uint8_t model_id, int32_t wait);

/**
 * @brief Disable all input and output data to be transferred to device of
 * specified model. This function is used to guarantee no data from specified
 * model will be transferred to device in case multiple models are sharing the
 * the same hardware resource and want to transfer data in the same time. Use
 * `memx_set_stream_enable()` to resume data transfer of specified model.
 *
 * @param model_id            model ID
 * @param wait                wait until driver state change from data-transfer to idle complete
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_set_stream_disable(uint8_t model_id, int32_t wait);

/**
 * @brief Configure driver input feature map internal queue size. No more input
 * feature map will be allowed to write to driver if driver internal queue full.
 * An error will be reported in case driver internal queue full and data is not
 * copied to driver. All input flows share the same queue size configuration
 * with a default queue size to '4'.
 *
 * @param model_id            model ID
 * @param size                input feature map queue size
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_set_ifmap_queue_size(uint8_t model_id, int32_t size);

/**
 * @brief Configure input feature map shape for driver to allocate internal
 * buffer resources. Input feature map shape is the input shape of the first
 * layer of model. The shape configuration will also propagate to device
 * (firmware on device) to allocate hardware resources, so make sure there is no
 * other model are preforming inference before configuration. See
 * `memx_download_model()` for input feature map shape auto configuraion.
 *
 * ~~~~~
 * memx_status status = MEMX_STATUS_OK;
 * if (status == 0) {
 *   // set model 0 flow 0 (port 0) input feature map shape as RGB888(224,224,1,3)
 *   status = memx_set_ifmap_size(0, 0, 224, 224, 1, 3, MEMX_FMAP_FORMAT_RAW);
 * }
 * ~~~~~
 *
 * @param model_id            model ID
 * @param flow_id             input flow (port) ID
 * @param height              input feature map height
 * @param width               input feature map width
 * @param z                   input feature map z
 * @param channel_number      input feature map channel number
 * @param format              input feature map format
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_set_ifmap_size(uint8_t model_id, uint8_t flow_id, int32_t height, int32_t width, int32_t z, int32_t channel_number, int32_t format);

/**
 * @brief Configure input feature map using floating-point to RGB value range
 * conversion. This function will reshape internal buffer based on original
 * input feature map shape configuration, and must be called after
 * `memx_set_ifmap_size()`. Also, calling `memx_set_ifmap_size()` again will
 * clear current range conversion setting.
 *
 * @note Range conversion formula = (uint8)(y) = ( (float)(x) + shift ) * scale
 *
 * @param model_id            model ID
 * @param flow_id             input flow (port) ID
 * @param enable              input feature map data range conversion is to be enabled
 * @param shift               amount to shift before scale
 * @param scale               amount to scale before integer cast
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_set_ifmap_range_convert(uint8_t model_id, uint8_t flow_id, int32_t enable, float shift, float scale);

/**
 * @brief Configure driver output feature map internal queue size. No more
 * output feature map will be allowed to read from device to driver if driver
 * internal queue full. In case driver internal queue full which causes driver
 * does not asking data transfer from device, data flow might be blocked and
 * back pressure to input feature map causing no more input feature map will be
 * allowed to write to device. All output flows share the same queue size
 * configuration with a default queue size to '4'.
 *
 * @param model_id            model ID
 * @param size                output feature map queue size
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_set_ofmap_queue_size(uint8_t model_id, int32_t size);

/**
 * @brief Configure output feature map shape for driver to allocate internal
 * buffer resources. Output feature map shape is the output shape of the last
 * layer of model. The shape configuration will also propagate to device
 * (firmware on device) to allocate hardware resources, so make sure there is
 * no other model are performing intference before configuration. See
 * `memx_download_model()` for output feature map shape auto configuation.
 *
 * ~~~~~
 * memx_status status = MEMX_STATUS_OK;
 * if (status == 0) {
 *   // set model 0 flow 0 (port 0) output feature map shape as float32(19,19,1,12)
 *   status = memx_set_ofmap_size(0, 0, 19, 19, 1, 12, MEMX_FMAP_FORMAT_FLOAT32);
 * }
 * if (status == 0) {
 *   // set model 0 flow 1 (port 1) output feature map shape as float32(19,19,1,273)
 *   status = memx_set_ofmap_size(0, 1, 19, 19, 1, 273, MEMX_FMAP_FORMAT_FLOAT32);
 * }
 * ~~~~~
 *
 * @param model_id            model ID
 * @param flow_id             output flow (port) ID
 * @param height              output feature map height
 * @param width               output feature map width
 * @param z                   output feature map z
 * @param channel_number      output feature map channel number
 * @param format              output feature map format
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_set_ofmap_size(uint8_t model_id, uint8_t flow_id, int32_t height, int32_t width, int32_t z, int32_t channel_number, int32_t format);

/**
 * @brief Configure output feature map HPOC (High-Precision-Output-Channel)
 * setting. This setting must be used along with model compiled with option HPOC
 * enabled using MIX compiler. By enabling this option, driver will re-calculate
 * output feature map size with dummy channels inserted and remove these dummy
 * dummy channels when read back data from device.
 *
 * @note Performance might suffer since driver needs to walk through each
 * channel to remove all dummy channels
 *
 * @param model_id            model ID
 * @param flow_id             output flow (port) ID
 * @param hpoc_size           high-precision-output-channel number
 * @param hpoc_indexes        high-precision-output-channel index array
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_set_ofmap_hpoc(uint8_t model_id, uint8_t flow_id, int32_t hpoc_size, int32_t* hpoc_indexes);

/**
 * @brief Apply input feature map and output feature map shape configureaion
 * to device. This function should be called after `memx_set_ifmap_size()` and
 * `memx_set_ofmap_size()` for driver to calcaulte hardware resource allocation.
 *
 * ~~~~~
 * memx_status status = MEMX_STATUS_OK;
 * if (status == 0) {
 *   // set model 0 flow 0 (port 0) input feature map shape as RGB888(224,224,3)
 *   status = memx_set_ofmap_size(0, 0, 224, 224, 3, MEMX_FMAP_FORMAT_FLOAT32);
 * }
 * if (status == 0) {
 *   // set model 0 flow 0 (port 0) output feature map shape as float32(19,19,273)
 *   status = memx_set_ofmap_size(0, 1, 19, 19, 273, MEMX_FMAP_FORMAT_FLOAT32);
 * }
 * if (status == 0) {
 *   // calcaulate hardware resource allocation and configure to device
 *   status = memx_update_fmap_size(0, 1, 1);
 * }
 * ~~~~~
 *
 * @param model_id            model ID
 * @param in_flow_count       input flow numbers
 * @param out_flow_count      output flow numbers
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_update_fmap_size(uint8_t model_id, uint8_t in_flow_count, uint8_t out_flow_count);

/**
 * @brief Read back input feature map shape configured of specific flow (port).
 * Returns failure with variables set to all zeros in case given flow is not
 * configured.
 *
 * ~~~~~
 * // bind MPU device group 0 as MX3 to model 0
 * memx_status status = memx_open(0, 0, MEMX_DEVICE_CASCADE_PLUS);
 * if (status == 0) {
 *   // download weight memory and 1st model within DFP file to device
 *   status = memx_download_model(0, "model.dfp", 0,
 *     MEMX_DOWNLOAD_TYPE_WTMEM_AND_MODEL);
 * }
 * if (status == 0) {
 *   // read back flow 0 (port 0) input feature map shape for debug information
 *   int32_t height, width, z, channel_number, format;
 *   memx_get_ifmap_size(0, 0, &height, &width, &z, &channel_number, &format);
 *   // for example: shape = (224, 224, 1, 3), format = 1
 *   printf("shape = (%d, %d, %d, %d), format = %d\n",
 *     height, width, z, channel_number, format);
 * }
 * // always remember to close device finally
 * memx_close(0);
 * ~~~~~
 *
 * @param model_id            model ID
 * @param flow_id             input flow (port) ID
 * @param height              input feature map height
 * @param width               input feature map width
 * @param z                   input feature map z
 * @param channel_number      input feature map channel number
 * @param format              input feature map format
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_get_ifmap_size(uint8_t model_id, uint8_t flow_id, int32_t* height, int32_t* width, int32_t* z, int32_t* channel_number, int32_t* format);

/**
 * @brief Read back input feature map range conversion configured of specific
 * flow (port). Returns failure with variables set to all zeros in case flow is
 * not configured.
 *
 * @param model_id            model ID
 * @param flow_id             input flow (port) ID
 * @param enable              input feature map data range conversion is enabled
 * @param shift               amount to shift before scale
 * @param scale               amount to scale before integer cast
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_get_ifmap_range_convert(uint8_t model_id, uint8_t flow_id, int32_t* enable, float* shift, float* scale);

/**
 * @brief Read back output feature map shape configured of specific flow (port).
 * Returns failure with variables set to all zeros in case given flow is not
 * configured.
 *
 * ~~~~~
 * // bind MPU device group 0 as MX3 to model 0
 * memx_status status = memx_open(0, 0, MEMX_DEVICE_CASCADE_PLUS);
 * if (status == 0) {
 *   // download weight memory and 1st model within DFP file to device
 *   status = memx_download_model(0, "model.dfp", 0,
 *     MEMX_DOWNLOAD_TYPE_WTMEM_AND_MODEL);
 * }
 * if (status == 0) {
 *   // read back flow 0 (port 0) input feature map shape for debug information
 *   int32_t height, width, z, channel_number, format;
 *   memx_get_ofmap_size(0, 0, &height, &width, &z, &channel_number, &format);
 *   // for example: shape = (224, 224, 1, 3), format = 1
 *   printf("shape = (%d, %d, %d, %d), format = %d\n",
 *     height, width, z, channel_number, format);
 * }
 * // always remember to close device finally
 * memx_close(0);
 * ~~~~~
 *
 * @param model_id            model ID
 * @param flow_id             output flow (port) ID
 * @param height              output feature map height
 * @param width               output feature map width
 * @param z                   output feature map z
 * @param channel_number      output feature map channel number
 * @param format              output feature map format
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_get_ofmap_size(uint8_t model_id, uint8_t flow_id, int32_t* height, int32_t* width, int32_t* z, int32_t* channel_number, int32_t* format);

/**
 * @brief Read back output feature map HPOC (High-Precision-Output-Channel)
 * setting. The 'hpoc_indexes' returned is owned by driver in case 'hpoc_size'
 * larger than zero, and should not be modified by user.
 *
 * @param model_id            model ID
 * @param flow_id             output flow (port) ID
 * @param hpoc_size           high-precision-output-channel number
 * @param hpoc_indexes        high-precision-output-channel index array (read-only, do not modify)
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_get_ofmap_hpoc(uint8_t model_id, uint8_t flow_id, int32_t* hpoc_size, int32_t** hpoc_indexes);

/**
 * @brief Get device count
 *
 * @param model_id Model index
 * @param pData    Buffer from host to get device count
 *
 * @return memx_status Return status from driver
 */
memx_status memx_operation_get_device_count(void *pData);

/**
 * @brief Get group count by model_id
 *
 * @param model_id Model index
 * @param pData    Buffer from host to get group count
 *
 * @return memx_status Return status from driver
 */
memx_status memx_operation_get_mpu_group_count(uint8_t group_id, void *pData);

/**
 * @brief Control channel write multiple bytes of data to device. The most
 * common usage is to write control register setting for debugging purpose.
 * Based on hardware design, data length given should always be aligned to
 * device's bus width which is one word equals to 4 bytes.
 *
 * @param model_id            model ID
 * @param address             target address
 * @param data                byte buffer to write
 * @param length              byte length to write
 * @param increment           post-write-address-increase-4-bytes enable
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_control_write(uint8_t model_id, uint32_t address, uint8_t* data, size_t length, int32_t increment);

/**
 * @brief Control channel read multiple bytes of data from device. The most
 * common usage is to read debug register or dump setting for debugging
 * purpose. Based on hardware design, data length given should always be
 * aligned to device's bus width which is one word equals to 4 bytes.
 *
 * @param model_id            model ID
 * @param address             source address
 * @param data                byte buffer to store data read
 * @param length              byte length to read
 * @param increment           post-read-address-increase-4-bytes enable
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_control_read(uint8_t model_id, uint32_t address, uint8_t* data, size_t length, int32_t increment);

/**
 * @brief Data channel write multiple bytes of data to device. This function
 * provides raw byte stream data access without internal queue and encoding.
 * Also, will not be blocked by driver internal state. Use `memx_stream_ifmap()`
 * instead for normal inference.
 *
 * @param model_id            model ID
 * @param flow_id             input flow (port) ID
 * @param data                byte buffer to write
 * @param length              byte length to write
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_stream_write(uint8_t model_id, uint8_t flow_id, uint8_t* data, size_t length);

/**
 * @brief Data channel read multiple bytes of data from device. This function
 * provides raw byte stream data access without internal queue and decoding.
 * Also, will not be blocked by driver internal state. Use `memx_stream_ofmap()`
 * instead for normal inference.
 *
 * @param model_id            model ID
 * @param flow_id             output flow (port) ID
 * @param data                byte buffer to store data read
 * @param length              byte length to read
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_stream_read(uint8_t model_id, uint8_t flow_id, uint8_t* data, size_t length);

/**
 * @brief Data channel write input feature map of specified flow (port) to
 * device. This function must be called after `memx_download_model()`.
 *
 * ~~~~~
 * memx_status status = MEMX_STATUS_OK;
 * if (status == 0) {
 *   // write data to model 0 flow 0 (port 0) to run inference
 *   status = memx_stream_ifmap(0, 0, ifmap, 200);
 * }
 * ~~~~~
 *
 * @param model_id            model ID
 * @param flow_id             input flow (port) ID
 * @param ifmap               input feature map (frame)
 * @param timeout             milliseconds timeout, '0' indicates infinite
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_stream_ifmap(uint8_t model_id, uint8_t flow_id, void* ifmap, int32_t timeout);

/**
 * @brief Data channel read output feature map of specified flow (port) from
 * device. This function must be called after `memx_download_model()`.
 *
 * ~~~~~
 * memx_status status = MEMX_STATUS_OK;
 * if (status == 0) {
 *   // read data from model 0 flow 0 (port 0) as inference result
 *   status = memx_stream_ofmap(0, 0, ofmap, 200);
 * }
 * ~~~~~
 *
 * @param model_id            model ID
 * @param flow_id             output flow (port) ID
 * @param ofmap               output feature map (frame)
 * @param timeout             milliseconds timeout, '0' indicates infinite
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_stream_ofmap(uint8_t model_id, uint8_t flow_id, void* ofmap, int32_t timeout);

/**
 * @brief Config MPU group on the device that specified MPU model used, support MEMX_MPU_GROUP_CONFIG_ONE_GROUP_FOUR_MPUS and MEMX_MPU_GROUP_CONFIG_TWO_GROUP_TWO_MPUS
 *
 * @param model_id            model ID
 * @param mpu_group_config    MPU group config
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_config_mpu_group(uint8_t model_id, uint8_t mpu_group_config);

/**
 * @brief Read back the chip gen for the currently active model.
 *
 * @param model_id            model ID
 * @param chip_gen            chip gen of device (MEMX_MPU_CHIP_GEN_CASCADE, MEMX_MPU_CHIP_GEN_CASCADE_PLUS)
 *
 * @return 0 on success, otherwise error code
 */
MEMX_API_EXPORT memx_status memx_get_chip_gen(uint8_t model_id, uint8_t* chip_gen);


#ifdef __cplusplus
}
#endif

#endif /* MEMX_H_ */
