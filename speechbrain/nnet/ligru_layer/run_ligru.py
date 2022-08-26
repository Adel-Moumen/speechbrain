import torch
from torch.utils.cpp_extension import load

ligru_model = load(
    "ligru",
    [
        "speechbrain/nnet/ligru_layer/ligru.cc",
        "speechbrain/nnet/ligru_layer/ligru_forward.cu.cc",
        "speechbrain/nnet/ligru_layer/ligru_backward.cu.cc",
    ],
    verbose=True,
)

if __name__ == "__main__":
    print(ligru_model.backward())
