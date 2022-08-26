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

    input_size, hidden_size, batch_size, seq_length = 2, 2, 2, 2

    w = torch.randn((input_size, hidden_size * 2), device="cuda")
    u = torch.randn((hidden_size, hidden_size * 2), device="cuda")
    x = torch.randn((batch_size, seq_length, input_size), device="cuda")
    h = torch.randn((batch_size, hidden_size), device="cuda")
    drop_mask = torch.randn((batch_size, hidden_size), device="cuda")

    print(ligru_model.forward(
        w,
        u,
        x,
        h,
        drop_mask
    ))
