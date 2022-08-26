// Copyright 2020 LMNT, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "ligru.h"
#include "support.h"

namespace {

using torch::Tensor;

int ligru_forward(
    Tensor w, 
    Tensor u, 
    Tensor x, 
    Tensor h,
    Tensor drop_mask
) {

  CHECK_INPUT(w);
  CHECK_INPUT(u);
  CHECK_INPUT(x);
  CHECK_INPUT(h);
  CHECK_INPUT(drop_mask);

  const auto batch_size = x.size(0);
  const auto time_steps = x.size(1);
  const auto input_size = x.size(2);
  const auto hidden_size = u.size(0);
  
  
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "ligru_forward", ([&] {

  }));

  return 0;
}

}  // anonymous namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ligru_forward, "LiGRU forward");
}