#ifndef MX_MODEL
#define MX_MODEL

#include <iostream>
#include <vector>
#include <memx/dfp.h>
#include <memx/utils/general.h>
#include <memx/utils/featureMap.h>
#include <memx/utils/errors.h>
#include <memx/utils/mxTypes.h>
#include <stdint.h>
#include <cstdio>
#include <memx/memx.h>
#include <thread>
#include <functional>
#include <atomic>
#include <cstring>
#include <unordered_set>
#include <memx/utils/thread_pool.hpp>

using namespace std;

namespace MX
{
    namespace Runtime
    {
        /**
         * Base Model class that needs to be inherited by template Model class
         * This class provides virtual funtions required by user that are
         * supposed to be overriden by child classes
        */
        class ModelBase{
            public:
            //function refernce for callback functions
            typedef std::function<bool(vector<const MX::Types::FeatureMap<uint8_t> *>, int stream_id)> int_callback_t;
            typedef std::function<bool(vector<const MX::Types::FeatureMap<float> *>, int stream_id)> float_callback_t;

            //connect_stream to this Model
            virtual void connect_stream(float_callback_t in_cb, float_callback_t out_cb, int stream_id)
                                            {
                                                throw runtime_error("base connect_stream float is called");
                                            }

            //connect_stream to this Model
            virtual void connect_stream(int_callback_t in_cb, float_callback_t out_cb, int stream_id)
                                            {
                                                throw runtime_error("base connect_stream int is called");
                                            }

            //Set number of workers
            virtual void set_num_workers(int input_workers, int output_workers){
                throw runtime_error("base set_num_workers called");
            }
            //Start the model
            virtual void model_start(){throw runtime_error("base model_start is called");};

            //Stop the model
            virtual void model_stop(){throw runtime_error("base model_stop is called");};

            // manual threading model start function to init model and featureMaps
            virtual void model_manual_start(){throw runtime_error("base model_manual_start is called");};

            // // manual threading model stop function to init model and featureMaps
            virtual void model_manual_stop(){throw runtime_error("base model_manual_stop is called");};

            // manual threading model send for float
            virtual void model_manual_send(std::vector<float*> in_data, int stream_id, bool channel_first=false){
                throw runtime_error("base model manual send for float is called");
            };

            // manual threading send for int
            virtual void model_manual_send(std::vector<uint8_t*> in_data, int stream_id, bool channel_first=false){
                throw runtime_error("base model manual send for int is called");
            };

            // manual threadin send for int
            virtual void model_manual_receive(std::vector<uint8_t*> &out_data, int& stream_id, bool channel_first=false){
                throw runtime_error("base model manual receive is called");
            };
            // manual threadin send for float
            virtual void model_manual_receive(std::vector<float*> &out_data, int& stream_id, bool channel_first=false){
                throw runtime_error("base model manual receive is called");
            };
            //Get num streams in this model
            virtual int get_num_streams(){throw runtime_error("base get_num_streams is called");};

            //Wait for model to finish
            virtual void model_wait(){throw runtime_error("base model_wait is called");};

            virtual void log_model_info(){throw runtime_error("base print info is called");};

            virtual MX::Types::MxModelInfo return_model_info() {throw runtime_error("base model info requested");};

            virtual ~ModelBase(){};
        };

        template <typename T>
        class MxModel : public ModelBase
        {
        private:
            int model_id_; // unique id of the model on MXA
            int group_id_; // unique id of MXA
            int num_streams_;// num streams connected to this model
            Dfp::DfpObject *dfp_; // Dfp object
            MX::Utils::fifo_queue<int> stream_queue; //queue to store stream ids for ifmaps
            MX::Utils::fifo_queue<int> out_queue; // queue to store stream ids for ofmaps
            vector<uint8_t> in_ports_; // input port information
            vector<uint8_t> out_ports_; // output port information
            int input_num_workers_;
            int output_num_workers_;
            void set_num_workers(int input_workers, int output_workers) override;
            thread_pool* input_pool;
            thread_pool* output_pool;
            void model_send_fun(); //send thread function to perform ifmap
            void model_recv_fun(); //recv thread function to perform ofmap
            thread *model_send_thread; //send thread for model
            thread *model_recv_thread; //recv thread for model
            atomic_bool model_run; //flag to specify if model is running
            atomic_bool model_recv_run;// flag to specify if model recv thread is running
            atomic_bool model_manual_run; //flag to specify manual threadin is opted out
            typedef std::function<bool(vector<const MX::Types::FeatureMap<T> *>, int stream_id)> combined_input_callback_t;
            typedef std::function<bool(vector<const MX::Types::FeatureMap<float> *>, int stream_id)> combined_output_callback_t;
            std::vector<std::mutex*>input_mutex;
            bool inputTask(combined_input_callback_t in_cb, vector<const MX::Types::FeatureMap<T>* >inputs,int stream, int stream_idx);
            //vector of input callback functions
            vector<combined_input_callback_t> comb_in_call;
            //vector of output callback functions
            vector<combined_output_callback_t> comb_out_call;
            //set of stream ids connected to the whole accl
            unordered_set<int>* stream_set_;
            //list of streamids connected to the model
            vector<int> stream_id_list;

            //Vector of featureMaps of size num_streams that holds inputs for models
            vector<vector<MX::Types::FeatureMap<T> *>> in_featuremaps_;
            //Vector of featureMaps of size num_streams that holds outputs for models
            vector<vector<MX::Types::FeatureMap<float> *>> out_featuremaps_;

            //model information
            MX::Types::MxModelInfo model_info;
            float mxa_gen;//Generation of chip DFP is compiled

            vector<MX::Types::FeatureMap<T> *> single_input_featuremap_;
            vector<MX::Types::FeatureMap<float> *> single_output_featuremap_;
            std::condition_variable model_manual_cv;
            std::mutex manual_mutex;
            bool model_manual_in_done;

            //Context ID
            int context_id_;

            //input task
            bool input_task_flag;
            std::mutex input_task_mutex;
            std::condition_variable input_task_cv;
        public:
            MxModel(int model_id, Dfp::DfpObject *dfp_object, std::unordered_set<int>* stream_set, int context_id); // Construct model for Inference
            void model_start() override;
            void model_stop() override;
            void model_wait() override;

            void model_manual_start() override;
            void model_manual_stop() override;
            ~MxModel();
            void connect_stream(combined_input_callback_t in_cb, combined_output_callback_t out_cb, int stream_id) override;
            int get_num_streams() override;
            void log_model_info() override;
            MX::Types::MxModelInfo return_model_info() override;

            void model_manual_send(std::vector<T*> in_data, int stream_id, bool channel_first=false) override;

            void model_manual_receive(std::vector<float*> &out_data, int& stream_id, bool channel_first=false) override;
        };
    } // namespace Runtime
} // namespace MX

#endif
