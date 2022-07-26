import torch
from torch.utils.cpp_extension import load

ligru_cell = load("ligru_cell", sources=["speechbrain/nnet/ligru_layer/ligru_layer.cpp"], verbose=True)


class LayerNormCustom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, wx, ht, u, drop_mask):


        gates = ligru_cell.forward(wx, ht, u, drop_mask)

        ctx.save_for_backward(
            wx, ht, u
        )

        return gates[0]

    @staticmethod
    def backward(ctx, grad_out):
        wx, ht, u = ctx.saved_tensors

        dx, dh, du = ligru_cell.backward(
            grad_out.contiguous(),
            ht,
            ht,
            u,
            wx)

        return None, dh, du, None


if __name__ == "__main__":
    B, T, H = 1, 2, 2

    torch.manual_seed(42)
    wx = torch.randn(B, T, H * 2, device="cuda", dtype=torch.double)
    ht = torch.randn(B, H, device="cuda", dtype=torch.double)
    u = torch.randn(H * 2, H, device="cuda", requires_grad=True, dtype=torch.double)
    drop_mask = torch.randn(B, H, device="cuda", dtype=torch.double)
    eps = 1e-5

    # print(wx)

    # u = torch.nn.Linear(H, 2 * H, bias=False)
    # print(u.weight.shape)

    out = LayerNormCustom.apply(wx, ht, u, drop_mask)
    out.sum().backward()
    # out = ligru_cell.forward(wx, ht, u, drop_mask, eps)
    # out.sum().backward()
    print(torch.autograd.gradcheck(LayerNormCustom.apply, [wx, ht, u, drop_mask]))

    # torch.manual_seed(42)
    # wx = torch.randn(B, T, H * 2, device="cuda")
    # ht = torch.randn(B, H, device="cuda")
    # u = torch.randn(H * 2, H, device="cuda", requires_grad=True)
    # drop_mask = torch.randn(B, H, device="cuda")
    # norm = torch.nn.LayerNorm(H * 2, elementwise_affine=False)
    # out2 = norm(ht @ u.T)
    # out2.sum().backward()
    # print(u.grad)
    # normalized, mean, rstd = out
