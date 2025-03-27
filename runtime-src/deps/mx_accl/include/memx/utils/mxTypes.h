#ifndef MXTYPES_H
#define MXTYPES_H
#include <vector>
#include <stdint.h>
#include <stdexcept>
#include <iostream>
namespace MX
{
    namespace Types
    {

        class ShapeVector {
            private:
               std::vector<int64_t> shape;
               int64_t h=0, w=0, z=0, c=0;
            public:   
                // Constructor
                ShapeVector();// Initialize shape with 4 elements, all initialized to 0
                // Constructor
                ShapeVector(int64_t h, int64_t w, int64_t z, int64_t c) ;
                // Overload [] operator const
                const int64_t& operator[](int64_t index) const ;
                // Overload [] operator 
                int64_t& operator[](int64_t index) ;
                // get shape in channel first format
                std::vector<int64_t> chfirst_shape();
                // get shape in channel last format
                std::vector<int64_t> chlast_shape();
                // data to return pointer
                int64_t* data();
                // size of the vector
                int64_t size() const ;
               
        };
        //struct with necessary model information collated for internal and external purposes 
        struct MxModelInfo{
            int model_index;
            int num_in_featuremaps;
            int num_out_featuremaps;
            std::vector<const char*> input_layer_names;
            std::vector<const char*> output_layer_names; 
            std::vector<MX::Types::ShapeVector> in_featuremap_shapes; // <<H:W:Z:C>>
            std::vector<MX::Types::ShapeVector> out_featuremap_shapes; // <<H:W:Z:C>>
            std::vector<size_t> in_featuremap_sizes;
            std::vector<size_t> out_featuremap_sizes;

        };

    } // Namespace Types
} // Namespace MX


#endif