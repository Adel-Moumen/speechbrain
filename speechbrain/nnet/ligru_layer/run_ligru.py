import torch
from torch.utils.cpp_extension import load

ligru_model = load(
    "ligru",
    [
        "src/nnet/ligru_layer_v2/ligru.cc",
        "src/nnet/ligru_layer_v2/ligru_forward.cu.cc",
        "src/nnet/ligru_layer_v2/ligru_backward.cu.cc",
    ],
    verbose=True,
)

if __name__ == "__main__":
    print(ligru_model.backward())
