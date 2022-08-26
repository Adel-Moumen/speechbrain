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

        void Iterate(
        const T* w,
        const T* u,
        const T* x,
        const T* h,
        T* h_out,
        T* v,
        T* tmp_Wx,
        T* tmp_Rh,
        const T* drop_mask);


    private:

        struct private_data;
        private_data* data_;

};

}  // namespace ligru