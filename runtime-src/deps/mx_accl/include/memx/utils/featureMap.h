#ifndef FEATUREMAP_H
#define FEATUREMAP_H

#include <vector>
#include <stdint.h>
#include <stdexcept>
#include <memx/utils/general.h>
#include <memx/utils/errors.h>
#include <memx/utils/mxTypes.h>
#include <atomic>
namespace MX
{
    namespace Types
    {
        //                          0             1              2              3              4            5              6
        enum MX_data_format { MX_FMT_GBF80, MX_FMT_RGB565, MX_FMT_RGB888, MX_FMT_YUV422, MX_FMT_BF16, MX_FMT_FP32, MX_FMT_GBF80_ROW};

        /**
         * @brief The FeatureMap class
         *
         * FeatureMaps are entities that are internally used by MxAccl to hold
         * and manipulate data. It provides required methods such as set_data() and
         * get_data() for the users to safely send and receive data from MxAccl.
         *
         */
        template <typename T>
        class FeatureMap
        {
        public:
            /**
             * @brief Constructor to featureMap - creates a data block of given size and sets all necessary dimension of the featureMap
             *
             * @param size      featureMap size
             * @param format    featureMap format
             * @param dim_h     Height of the featureMap
             * @param num_chan  Width of the featureMap
             * @param dim_z     Z dimension of featureMap
             * @param num_chan  number of channels
             */
            FeatureMap(size_t size, MX_data_format format = MX_FMT_FP32, uint16_t dim_h = 0, uint16_t dim_w = 0, uint16_t dim_z = 0, size_t num_chan = 0);

            /**
             * @brief Additional Constructor to featureMap - creates a data block and copies the input data to the block and sets all necessary dimensions of the featureMap
             *
             * @param input     POinter to input data to featureMap
             * @param size      featureMap size
             * @param format    featureMap format
             * @param dim_h     Height of the featureMap
             * @param num_chan  Width of the featureMap
             * @param dim_z     Z dimension of featureMap
             * @param num_chan  number of channels
             */
            FeatureMap(T *input, size_t size, MX_data_format format = MX_FMT_FP32, uint16_t dim_h = 0, uint16_t dim_w = 0, uint16_t dim_z = 0, size_t num_chan = 0);

            /**
             * @brief Copy constructor of the featureMap. Copies all necessary information from another featureMap
             *
             * @param rhs featureMap object
             */
            FeatureMap(const FeatureMap& rhs);

            /**
             * @brief Function get output from Accelarator. Copies output data from featureMap to passed pointer
             *
             * @param out_data pointer to destination where output data from accelrator to be copied
             * @param channel_first boolean variable based on which output data is copied in channel first or channel last format. default is false to return channel last format
             * @return MX_Status Success if the copy is successfull
             */
            MX::Utils::MX_status get_data(T *out_data , bool channel_first=false) const;
            /**
             * @brief Function to set input data to Accelarator. Copies data from provided input pointer to featureMap
             *
             * @param in_data pointer to source from where input data is to be copied from
             * @param channel_first boolean variable that indicates the copied data is in channel first or channle last format. default is false expecting data in channel last format
             * @return MX_Status Success if the copy is successfull
             */
            MX::Utils::MX_status set_data(T *in_data , bool channel_first=false) const;
            //Returns the data pointer of featureMap after
            const T * const get_raw_data();

            //Returns the pointer of formatted data of featureMap
            uint8_t *get_formatted_data();

            virtual ~FeatureMap();
            //sets the the in_ready flag to user input
            void set_in_ready(bool flag);
            //sets the the out_ready flag to user input
            void set_out_ready(bool flag);
            //return in_ready flag
            bool get_in_ready();
            //returns out_ready flag
            bool get_out_ready();
            // Transpose function to use if channel first is required
            void transpose_hwc_chw(T* input, T* output) const;
            //transpose function to use if channel last is required
            void transpose_chw_hwc(T* input, T* output) const;

            //copy assignment operator
            FeatureMap& operator=(const FeatureMap& rhs);

            //returns size of featureMap
            size_t get_size() const;

            size_t get_formatted_size();
            void notify();
            void wait();
            std::vector<int64_t> shape(bool channel_first=false) const;
        private:
            T *fmap_data; // data in user-facing format (float or uint8_t)
            size_t featureMap_size; // size, in terms of user-facing format
            MX_data_format fmt; // data format
            uint8_t *formatted_data;
            size_t formatted_featuremap_size; // how many bytes the converted data is
            void convert_data() const; // converts *data -> *formatted_data
            void unconvert_data() const; // converts *formatted_data -> *data
            void calc_convert_size_and_new(); // calculates size for and allocates formatted bytes
            mutable std::atomic_bool out_ready; //flag that is used by MxModel
            mutable std::atomic_bool in_ready; //flag that is used by MxModel
            std::mutex wait_m; //
            bool wait_flag;
            std::condition_variable wait_cv;

            /* dimension variables used for GBF calculations, shape and transofrms*/
            uint16_t dim_h;      // shape dimension x (height)
            uint16_t dim_w;      // shape dimension y (width)
            uint16_t dim_z;      // shape dimension z
            size_t num_ch;      // number of channels -- used for GBF calculations, shape and transforms

        };
    } // namespace Types
} // namespace MX

#endif
