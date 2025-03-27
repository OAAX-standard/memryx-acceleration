#ifndef MX_ACCL
#define MX_ACCL

#include <string>
#include <stdint.h>
#include <atomic>
#include <thread>

#include <memx/MxModel.h>
#include <memx/dfp.h>
#include <memx/utils/general.h>
#include <memx/utils/featureMap.h>
#include <memx/utils/path.h>

using namespace std;

namespace MX
{
  namespace Runtime
  {
    class MxAccl
    {
    private:
      typedef std::function<bool(vector<const MX::Types::FeatureMap<uint8_t> *>, int stream_id)> int_callback_t;
      typedef std::function<bool(vector<const MX::Types::FeatureMap<float> *>, int stream_id)> float_callback_t;
    public:
      /**
       * @brief Construct a new MxAccl object
       *
       * @param file_path Absolute path of DFP file. char* and String types can also be passed.
       * @param group_id GroupId of MPU this application is intended to use.
       * group_id is defaulted to 0, but needs to be provided if using
       * any other group
       */
      MxAccl(const std::filesystem::path file_path, int group_id=0);

      //Destructor
      ~MxAccl();

      /**
       * @brief Start running inference.
       * All streams must be connected before calling this function
       *
       * @param manual_threading set this flag true to use the accl in userThreading mode by sending and receiving input and output through functions send_input() and receive_output()
       */
      void start(bool manual_threading=false);

      /**
       * @brief Stop running inference.
       * Shouldn't be called before calling start.
       *
       */
      void stop();

      /**
       * @brief Wait for all the streams to be done streaming. This function waits
       * till all the started input callbacks have returned false.
       * Shouldn't be called before calling start.
       *
       */
      void wait();

      /**
       * @brief Get number of models in the compiled DFP
       *
       * @return Number of models
       */
      int get_num_models();

      /**
       * @brief Get number the number of streams connected to the object
       *
       * @return Number of streams
       */
      int get_num_streams();

      /**
       * @brief Connect a stream to a model
       * - float_callback_t is a function pointer of type, bool foo(vector<const MX::Types::FeatureMap<float>*>, int).
       * - When this input callback function returns false, the corresponding stream is stopped and when all the streams stop,
       * wait() is executed.
       * - connect_stream should be called before calling start() or after calling stop().
       * @param in_cb -> input callback function used by this stream
       * @param out_cb -> output callback function used by this stream
       * @param stream_id -> Unique id given to this stream which can later
       *              be used in the corresponding callback functions
       * @param model_id -> Index of model this stream is intended to be connected
      */
      void connect_stream(float_callback_t in_cb, float_callback_t out_cb, int stream_id, int model_id=0);
      /**
       * @brief Connect a stream to a model
       * - float_callback_t is a function pointer of type, bool foo(vector<const MX::Types::FeatureMap<float>*>, int).
       * - int_callback_t is a function pointer of type, bool foo(vector<const MX::Types::FeatureMap<int>*>, int).
       * - When this input callback function returns false, the corresponding stream is stopped and when all the streams stop,
       * wait() is executed.
       * - connect_stream should be called before calling start() or after calling stop().
       * @param in_cb -> input callback function used by this stream
       * @param out_cb -> output callback function used by this stream
       * @param stream_id -> Unique id given to this stream which can later
       *              be used in the corresponding callback functions
       * @param model_id -> Index of model this stream is intended to be connected
      */
      void connect_stream(int_callback_t in_cb, float_callback_t out_cb, int stream_id, int model_id=0);

      /**
       * @brief get information of a particular model such as number of in out featureMaps and in out layer names
       * @param model_id model ID or the index for the required information
       * @return if valid model_id then MxModelInfo model_info with necessary information else throw runtime error invalid model_id
      */
      MX::Types::MxModelInfo get_model_info(int model_id) const;

      // User threading functions - No doxygen comments as we are releasing this for internal use
      // send input - float
      /**
       * @brief Send input to the accelerator in userThreading mode. Do not use this function when using auto-threading
       *
       * @param in_data -> vector of input data to the model
       * @param model_id -> Index of the model the data is targetted to.
       * @param stream_id -> Index of stream the input data belongs to.
       * @param channel_first -> boolean variable that indicates the copied data is in channel first or channle last format. default is false expecting data in channel last format
      */
      void send_input(std::vector<float*> in_data, int model_id, int stream_id, bool channel_first = false);

      /**
       * @brief Send input to the accelerator in userThreading mode. Do not use this function when using auto-threading
       *
       * @param in_data -> vector of input data to the model
       * @param model_id -> Index of the model the data is targetted to.
       * @param stream_id -> Index of stream the input data belongs to.
       * @param channel_first -> boolean variable that indicates the copied data is in channel first or channle last format. default is false expecting data in channel last format
      */
      void send_input(std::vector<uint8_t*> in_data, int model_id, int stream_id, bool channel_first = false);

      /**
       * @brief Receive output from the accelerator in userThreading mode. Do not use this function when using auto-threading
       *
       * @param in_data -> vector of output data from the model
       * @param model_id -> Index of the model the data is intended to come from.
       * @param stream_id -> Index of stream the output data belongs to.
       * @param channel_first -> boolean variable that indicates the copied data is in channel first or channle last format. default is false expecting data in channel last format
      */
      void receive_output(std::vector<float*> &out_data, int model_id, int& stream_id, bool channel_first = false);

      /**
       * @brief Set the number of workers for input and output streams. The default is the number of streams
       * for both number of input and output streams as that provides the maximum performance. If this method is
       * not called before calling start(), the accl will run in default mode. This method should be called
       * after connecting all the required streams.
       *
       * @param input_num_workers Number of input workers
       * @param output_num_workers Number of output workers
       * @param model_idx Index of model to which the workers are intended to be assigned to. The default is set to 0
      */
      void set_num_workers(int input_num_workers, int output_num_workers,int model_idx=0);

    private:
      //Initiates the accl
      void init();
      //Open and lock the MPU
      bool setup_mxa();
      //Gracefully close the MPU
      bool close_mxa();
      //CascadePlus specific function that consfigures the num chips
      //config based on dfp
      void configure_group();
      //Helper to throw exception incase of chip, dfp mismatch
      void throw_chip_exception(int chips);

      bool we_opened_dfp;
      Dfp::DfpObject *dfp;

      bool valid;

      int group; //Group of the chip connected

      int context_id; // usedfor internal chip access

      int num_chips;//Number of chips DFP is compiled

      float mxa_gen;//Generation of chip DFP is compiled

      bool async_mode;//Use MxAccl in async mode

      int num_models;//Number of models DFP is compiled

      std::atomic_bool run;//Flag to know status of the Accl

      std::atomic_bool manual_run; // Flag to mark manual threading option;

      unordered_set<int> stream_id_set;//Set of stream_ids provided by users during connect stream

      std::vector<ModelBase *> models;//Vector of all model objects
    };
  } // namespace Runtime
} // namespace MX

#endif
