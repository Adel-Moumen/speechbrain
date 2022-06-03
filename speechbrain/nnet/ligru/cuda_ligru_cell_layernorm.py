"""This library implements a LiGRU cell needed for the LiGRU.

LiGRU is a Gated Recurrent Unit introduced by Assoc. Prof Mirco
Ravanelli in 2018 (see: https://arxiv.org/pdf/1803.10225.pdf).

Authors
 * Adel Moumen 2022
"""

from numpy import var
import torch
import torch.autograd as autograd

try:
    import cupy as cp
except ImportError:
    err_msg = "The optional dependency CuPy is needed to use LiGRU on CuPy\n"
    err_msg += "Cannot import CuPy.\n"
    err_msg += "Please follow the instructions below\n"
    err_msg += "=============================\n"
    err_msg += "If you use your localhost:\n"
    err_msg += "$ python -m pip install -U setuptools pip\n"
    err_msg += "$ pip install cupy-cudaXXX (XXX is your Cuda Toolkit version)\n"
    err_msg += "If you use conda:\n"
    err_msg += "$ conda install -c conda-forge cupy"
    err_msg += "for more details: https://docs.cupy.dev/en/stable/install.html"
    err_msg += "=============================\n"
    raise ImportError(err_msg)

_preamble_gradient_relu = """
template <typename T> __device__ T gradient_activation_hcand(T x) { return (x > 0. ? 1. : 0.); }
"""

_preamble_gradient_leaky_relu = """
template <typename T> __device__ T gradient_activation_hcand(T x) {
    T negative_slope = 1e-2;
    return (x > 0. ? 1. : negative_slope);
}
"""

_preamble_gradient_tanh = """
template <typename T> __device__ T gradient_activation_hcand(T x) { return 1 - x * x; }
"""


def transform_tensor_to_cupy(x):
    """Transform a PyTorch Tensor located on device="cuda" to a CuPy array.

    Argument
    --------
        x : torch.Tensor
    """
    return cp.ascontiguousarray(
        cp.from_dlpack(torch.utils.dlpack.to_dlpack(x.detach()))
    )

def forward_layernorm(ctx, ht, u, gamma, beta):
    x = ht @ u.T
    shape = x.shape
    dim = shape[-1]
    x = x.view(-1, dim)
    eps = 1e-5
    mean = x.mean(-1, keepdim = True)
    var = x.var(-1, keepdim = True, unbiased=False)
    x_hat = (x-mean)/torch.sqrt(var+eps)
    output  = gamma * x_hat + beta
    ctx.save_for_backward(gamma, x_hat, ht,u)
    ctx.eps = eps
    return output 

def backward_layernorm(grad_out, x_hat, ht, eps, u, gamma):
    x = ht @ u.T
    var = x.var(-1, keepdim = True, unbiased=False)
    shape = grad_out.shape
    dim = shape[-1]
    grad_out = grad_out.view(-1, dim)
    save_grad = grad_out
    grad_out = grad_out * gamma
    dx = (((dim * grad_out) - (torch.sum(grad_out, axis=-1, keepdim = True)) -  x_hat * (torch.sum(grad_out * x_hat, axis=-1, keepdim = True))) / (dim * torch.sqrt(var+eps)))
    return transform_tensor_to_cupy((dx @ u)) , transform_tensor_to_cupy((dx.T @ ht)) , (torch.sum(x_hat * save_grad, axis=0)), (torch.sum(save_grad, axis=0))


class Sin(torch.nn.Module):
    def __init__(self):
        super(Sin, self).__init__()
        
    def forward(self, x):
        return torch.sin(x)

class _ligru_cell_jit(torch.nn.Module):
    """This class redefines the forward of a LiGRU cell.
    """

    def __init__(self, act, norm):
        super(_ligru_cell_jit, self).__init__()
        if act == "sin":
            self.act = Sin()
        elif act == "leaky_relu":
            self.act = torch.nn.LeakyReLU()
        elif act == "tanh":
            self.act = torch.nn.Tanh()
        else:
            self.act = torch.nn.ReLU()

        self.norm = norm

    def forward(self, wx, u, ht, drop_mask):
        """Compute the forward pass of a LiGRU cell.

        Arguments
        ---------
        wx : torch.Tensor
            Linearly transformed input.
        u  : torch.Tensor
            Recurrent weight.
        ht : torch.Tensor
            Hidden state.
        drop_mask : torch.Tensor
            Dropout mask.
        act : nn.Module
            Activation Function.
        """
        # save values for backward
        hiddens = []
        candidate_gate = []
        update_gate = []
        save_at = []
        save_x_hat = []
        h_init = ht
        eps = 1e-5
        # iterate over each timesteps
        for k in range(wx.shape[1]):

            x_hat = self.norm(ht @ u.T)
            gates = wx[:, k] + x_hat
            at, zt = gates.chunk(2, 1)
            zt = torch.sigmoid(zt)
            hcand = self.act(at) * drop_mask
            ht = ht * zt + (1 - zt) * hcand

            hiddens.append(ht)
            candidate_gate.append(hcand)
            update_gate.append(zt)
            save_at.append(at)
            save_x_hat.append(x_hat)

        # stacks values
        ht = torch.stack(hiddens, dim=1)
        zt = torch.stack(update_gate, dim=1)
        at = torch.stack(save_at, dim=1)
        hcand = torch.stack(candidate_gate, dim=1)

        return ht, zt, at, hcand,  eps, save_x_hat


class _ligru_cell_cupy(autograd.Function):
    """This class uses the jitted forward of a ligru cell and implement the backward using CuPy.
    By doing so, we speed up the training by a factor of ~4.7x in comparison to the original implementation
    and ~3x in comparison to the jitted version.
    """
    save_norm = 0

    @staticmethod
    def forward(ctx, cell_jit, wx, u, ht, drop_mask, gamma, beta):
        """Compute the hidden states over each timestep and save the intermediate results for the backward.

        The utilisation of Tanh in the LiGRU cell is not recommended because it increases the instability and can lead to
        nan values in the gradients.

        The forward has not been implemented with CuPy because it was leading to numerical instabilities.

        Arguments
        ---------
        cell_jit : _ligru_cell_jit object
            Jitted nn.Module
        wx : torch.Tensor
            Linearly transformed input.
        u  : torch.Tensor
            Recurrent weight.
        ht : torch.Tensor
            Hidden state.
        drop_mask : torch.Tensor
            Dropout mask.
        act : nn.Module
            Activation Function.
            Only three possibilities for now:
                1) ReLU,
                2) Leaky ReLU with slope_parameter of 1e-2 (default parameter of PyTorch),
                3) Tanh.
        """
        h_init = ht

        ht, zt, at, hcand,  eps, xhat = cell_jit(wx, u, h_init, drop_mask)
        ctx.save_for_backward(h_init, u, wx, zt, at, ht, hcand, drop_mask, gamma, beta)
        ctx.activation_function = cell_jit.act
        ctx.eps = eps
        ctx.xhat = xhat
        return ht

    @staticmethod
    def backward(ctx, grad_out):
        """Run the backward phase of the forward call defined above. This implementation
        is dependant of CuPy.

        Arguments
        ---------
        ctx : context variable
        grad_out : torch.Tensor
        """
        h_init, u, wx, zt, at, ht, hcand, drop_mask, gamma, beta, = ctx.saved_tensors

        activation_function = ctx.activation_function
        eps = ctx.eps
        xhat = ctx.xhat
        # we need to reshape our h_init tensor if h_init doesnt match the shape.
        if h_init.shape[0] != ht[:, 0].shape[0]:
            h_init = h_init.repeat(ht[:, 0].shape[0], 1)

        # Tensor -> Cuda CuPy Array
        h_init, u, wx, zt, at, ht, hcand, drop_mask, grad_out = (
            transform_tensor_to_cupy(x)
            for x in [h_init, u, wx, zt, at, ht, hcand, drop_mask, grad_out]
        )

        # allocate memory space
        dwx = cp.empty_like(wx)
        du = cp.zeros_like(u)
        dh = cp.empty_like(h_init)
        dh_prev = cp.zeros_like(h_init)
        idx = dwx.shape[2] // 2

        # find the activation function and load the appropriate preamble
        activation_function_name = activation_function.__class__.__name__
        if activation_function_name == "LeakyReLU":
            preamble_gradient = _preamble_gradient_leaky_relu
        elif activation_function_name == "Tanh":
            preamble_gradient = _preamble_gradient_tanh
        else:
            preamble_gradient = _preamble_gradient_relu

        _ligru_cell_backward_kernel = cp.ElementwiseKernel(
            "T grad_out, T dh_prev, T zt, T at, T drop_mask, T ht, T hcand",
            "T dh, T dat, T dzt, T grad_dh_prev",
            """
            dh = grad_out + dh_prev;
            T temp = (1 - zt) * dh;
            dat = gradient_activation_hcand(at) * drop_mask * temp;
            dzt = (ht - hcand) * temp * zt;
            grad_dh_prev = dh * zt;
            """,
            "_ligru_cell_backward_kernel",
            preamble=preamble_gradient,
        )

        save_dh = cp.zeros_like(dh)
        dgamma = torch.zeros_like(gamma)
        dbeta = torch.zeros_like(beta)

        # iterate over each timesteps in reversed
        for t in reversed(range(dwx.shape[1])):

            ht_ = h_init if t - 1 < 0 else ht[:, t - 1]

            _ligru_cell_backward_kernel(
                grad_out[:, t],
                dh_prev,
                zt[:, t],
                at[:, t],
                drop_mask,
                ht_,
                hcand[:, t],
                dh,
                dwx[:, t, :idx],
                dwx[:, t, idx:],
                dh_prev,
            )

            dh_act, du_act, _dgamma, _dbeta = backward_layernorm(torch.from_dlpack(dwx[:, t]), xhat[t], torch.from_dlpack(ht_), eps, torch.from_dlpack(u), gamma)
            dh_prev = dh_prev + dh_act
            save_dh += dh_prev
            du += du_act
            dgamma += _dgamma
            dbeta += _dbeta

        return (
            None,
            torch.from_dlpack(dwx),
            torch.from_dlpack(du),
            torch.from_dlpack(dh),
            None,
            dgamma,
            dbeta
        )