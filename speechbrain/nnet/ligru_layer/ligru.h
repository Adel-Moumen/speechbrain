#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace ligru {

template<typename T>
class ForwardPass {

    public:
        ForwardPass(
            const bool training,
            const int batch_size,
            const int input_size,
            const int hidden_size,
            const cublasHandle_t& blas_handle,
            const cudaStream_t& stream = 0);

        ~ForwardPass();

    private:

        struct private_data;
        private_data* data_;

};

}  // namespace ligru