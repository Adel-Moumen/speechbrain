
#pragma once

#include <torch/extension.h>

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

template<typename U>
struct native_type {
  using T = U;
};

template<>
struct native_type<c10::Half> {
  using T = __half;
};

template<typename U>
typename native_type<U>::T* ptr(torch::Tensor t) {
  return reinterpret_cast<typename native_type<U>::T*>(t.data_ptr<U>());
}
