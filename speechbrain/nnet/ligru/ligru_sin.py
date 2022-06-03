import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from core_rnn.cuda_ligru_cell import _ligru_cell_cupy

FLAG = False

class LiGRU(torch.nn.Module):
    """ This function implements a Light GRU (liGRU).
    LiGRU is single-gate GRU model based on batch-norm + relu
    activations + recurrent dropout. For more info see:
    "M. Ravanelli, P. Brakel, M. Omologo, Y. Bengio,
    Light Gated Recurrent Units for Speech Recognition,
    in IEEE Transactions on Emerging Topics in Computational Intelligence,
    2018" (https://arxiv.org/abs/1803.10225)
    It accepts in input tensors formatted as (batch, time, fea).
    In the case of 4d inputs like (batch, time, fea, channel) the tensor is
    flattened as (batch, time, fea*channel).
    Arguments
    ---------
    hidden_size : int
        Number of output neurons (i.e, the dimensionality of the output).
        values (i.e, time and frequency kernel sizes respectively).
    input_shape : tuple
        The shape of an example input.
    nonlinearity : str
        Type of nonlinearity (tanh, relu).
    normalization : str
        Type of normalization for the ligru model (batchnorm, layernorm).
        Every string different from batchnorm and layernorm will result
        in no normalization.
    num_layers : int
        Number of layers to employ in the RNN architecture.
    bias : bool
        If True, the additive bias b is adopted.
    dropout : float
        It is the dropout factor (must be between 0 and 1).
    re_init : bool
        If True, orthogonal initialization is used for the recurrent weights.
        Xavier initialization is used for the input connection weights.
    bidirectional : bool
        If True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used.
    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = LiGRU(input_shape=inp_tensor.shape, hidden_size=5)
    >>> out_tensor, _ = net(inp_tensor)
    >>>
    torch.Size([4, 10, 5])
    """

    def __init__(
        self,
        hidden_size,
        input_shape,
        nonlinearity="sin",
        normalization="batchnorm",
        num_layers=1,
        bias=True,
        dropout=0.0,
        re_init=True,
        bidirectional=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.num_layers = num_layers
        self.normalization = normalization
        self.bias = bias
        self.dropout = dropout
        self.re_init = re_init
        self.bidirectional = bidirectional
        self.reshape = False

    # Computing the feature dimensionality
        if len(input_shape) > 3:
            self.reshape = True
        self.fea_dim = float(torch.prod(torch.tensor(input_shape[2:])))
        self.batch_size = input_shape[0]
        self.rnn = self._init_layers()

        if self.re_init:
            rnn_init(self.rnn)

    def _init_layers(self):
        """Initializes the layers of the liGRU."""
        rnn = torch.nn.ModuleList([])
        current_dim = self.fea_dim

        for i in range(self.num_layers):
            rnn_lay = LiGRU_Layer(
                current_dim,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                dropout=self.dropout,
                nonlinearity=self.nonlinearity,
                normalization=self.normalization,
                bidirectional=self.bidirectional,
            )
            rnn.append(rnn_lay)

            if self.bidirectional:
                current_dim = self.hidden_size * 2
            else:
                current_dim = self.hidden_size
        return rnn

    def forward(self, x, hx: Optional[Tensor] = None):
        """Returns the output of the liGRU.
        Arguments
        ---------
        x : torch.Tensor
            The input tensor.
        hx : torch.Tensor
            Starting hidden state.
        """
        # Reshaping input tensors for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        # run ligru
        output, hh = self._forward_ligru(x, hx=hx)

        return output, hh

    def _forward_ligru(self, x, hx: Optional[Tensor]):
        """Returns the output of the vanilla liGRU.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        hx : torch.Tensor
        """
        h = []
        if hx is not None:
            if self.bidirectional:
                hx = hx.reshape(
                    self.num_layers, self.batch_size * 2, self.hidden_size
                )
        # Processing the different layers
        for i, ligru_lay in enumerate(self.rnn):
            if hx is not None:
                x = ligru_lay(x, hx=hx[i])
            else:
                x = ligru_lay(x, hx=None)
            h.append(x[:, -1, :])
        h = torch.stack(h, dim=1)

        if self.bidirectional:
            h = h.reshape(h.shape[1] * 2, h.shape[0], self.hidden_size)
        else:
            h = h.transpose(0, 1)

        return x, h

    def lambda_value(self):
        lmb = 0
        for i in range(self.num_layers):
            lmb += torch.sqrt(((self.rnn[i].lambda_regu - 1) ** 2))
        return lmb

class LiGRU_Layer(torch.nn.Module):
    """ This function implements Light-Gated Recurrent Units (ligru) layer.
    Arguments
    ---------
    input_size : int
        Feature dimensionality of the input tensors.
    batch_size : int
        Batch size of the input tensors.
    hidden_size : int
        Number of output neurons.
    num_layers : int
        Number of layers to employ in the RNN architecture.
    nonlinearity : str
        Type of nonlinearity (tanh, relu).
    normalization : str
        Type of normalization (batchnorm, layernorm).
        Every string different from batchnorm and layernorm will result
        in no normalization.
    dropout : float
        It is the dropout factor (must be between 0 and 1).
    bidirectional : bool
        if True, a bidirectional model that scans the sequence both
        right-to-left and left-to-right is used.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_size,
        dropout=0.0,
        nonlinearity="relu",
        normalization="batchnorm",
        bidirectional=False,
    ):

        super(LiGRU_Layer, self).__init__()
        self.hidden_size = int(hidden_size)
        self.input_size = int(input_size)
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.w = nn.Linear(self.input_size, 2 * self.hidden_size, bias=False)

        self.u = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=False)

        self.lambda_regu = None

        if self.bidirectional:
            self.batch_size = self.batch_size * 2

        # Initializing batch norm
        self.normalize = False

        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(2 * self.hidden_size, momentum=0.05)
            self.normalize = True

        elif normalization == "layernorm":
            self.norm = torch.nn.LayerNorm(2 * self.hidden_size)
            self.normalize = True
        else:
            # Normalization is disabled here. self.norm is only  formally
            # initialized to avoid jit issues.
            self.norm = torch.nn.LayerNorm(2 * self.hidden_size)
            self.normalize = True

        # Initial state
        self.register_buffer("h_init", torch.zeros(1, self.hidden_size))

        # Preloading dropout masks (gives some speed improvement)
        self._init_drop(self.batch_size)


        from speechbrain.nnet.ligru.cuda_ligru_cell import _ligru_cell_jit

        # jit the cell to gain some speed up
        self._ligru_cell_jit = torch.jit.script(_ligru_cell_jit(act=nonlinearity))

    def forward(self, x, hx: Optional[Tensor] = None):
        # type: (Tensor, Optional[Tensor]) -> Tensor # noqa F821
        """Returns the output of the liGRU layer.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        """
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        self._change_batch_size(x)

        # Feed-forward affine transformations (all steps in parallel)
        w = self.w(x)

        # Apply batch normalization
        if self.normalize:
            w_bn = self.norm(w.reshape(w.shape[0] * w.shape[1], w.shape[2]))
            w = w_bn.reshape(w.shape[0], w.shape[1], w.shape[2])

        # Sampling dropout mask
        drop_mask = self._sample_drop_mask(w).float()

        # Processing time steps
        if hx is not None:
            h, at = self._ligru_cell(w, hx, drop_mask)
        else:
            h, at = self._ligru_cell(w, self.h_init, drop_mask)

        # because I return h and at I can calculate lambda at the end easily
        #ht_max = h[:, :-1].max() 
        #self.lambda_regu = self._get_lambda(ht_max, self.u.weight, at)

        if FLAG == True:
            self.set_lambda(h, self.u.weight, at)

        if self.bidirectional:
            h_f, h_b = h.chunk(2, dim=0)
            h_b = h_b.flip(1)
            h = torch.cat([h_f, h_b], dim=2)

        return h

    def set_lambda(self, h, u, at):
        ht_max = h[:, :-1].max()
        uz, uh = u.chunk(2, 0)
        
        sz = torch.linalg.matrix_norm(uz, ord=2)   
        sh = torch.linalg.matrix_norm(uh, ord=2)
        
        self.lambda_regu = torch.cos(at).max() * sh + sz * ht_max / 4

        

    def _ligru_cell_cpu(self, w, ht, drop_mask):
        """Returns the hidden states for each time step.
        Arguments
        ---------
        wx : torch.Tensor
            Linearly transformed input.
        ht : torch.Tensor
            Hidden state at timestep - 1.
        drop_mask : torch.Tensor
            dropout masks.
        """
        ht, _, at, _ = self._ligru_cell_jit(w, self.u.weight, ht, drop_mask)
        return ht, at

    def _ligru_cell(self, w, ht, drop_mask):
        """Returns the hidden states for each time step.
        Choose the appropriate method to call based on the device of
        each tensors.
        Arguments
        ---------
        wx : torch.Tensor
            Linearly transformed input.
        ht : torch.Tensor
            Hidden state at timestep -1.
        drop_mask : torch.Tensor
            Dropout mask.
        """

        if w.is_cuda == ht.is_cuda == drop_mask.is_cuda is True:
            
            ht, at = _ligru_cell_cupy.apply(
                self._ligru_cell_jit, w, self.u.weight, ht, drop_mask
            )
            
            return ht, at
        else:
            return self._ligru_cell_cpu(w, ht, drop_mask)

    def _init_drop(self, batch_size):
        """Initializes the recurrent dropout operation. To speed it up,
        the dropout masks are sampled in advance.
        """
        self.drop = torch.nn.Dropout(p=self.dropout, inplace=False)
        self.N_drop_masks = 16000
        self.drop_mask_cnt = 0

        self.register_buffer(
            "drop_masks",
            self.drop(torch.ones(self.N_drop_masks, self.hidden_size)).data,
        )
        self.register_buffer("drop_mask_te", torch.tensor([1.0]).float())

    def _sample_drop_mask(self, w):
        """Selects one of the pre-defined dropout masks"""
        if self.training:

            # Sample new masks when needed
            if self.drop_mask_cnt + self.batch_size > self.N_drop_masks:
                self.drop_mask_cnt = 0
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks, self.hidden_size, device=w.device
                    )
                ).data

            # Sampling the mask
            drop_mask = self.drop_masks[
                self.drop_mask_cnt : self.drop_mask_cnt + self.batch_size
            ]
            self.drop_mask_cnt = self.drop_mask_cnt + self.batch_size

        else:
            self.drop_mask_te = self.drop_mask_te.to(w.device)
            drop_mask = self.drop_mask_te

        return drop_mask

    def _change_batch_size(self, x):
        """This function changes the batch size when it is different from
        the one detected in the initialization method. This might happen in
        the case of multi-gpu or when we have different batch sizes in train
        and test. We also update the h_int and drop masks.
        """
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

            if self.training:
                self.drop_masks = self.drop(
                    torch.ones(
                        self.N_drop_masks, self.hidden_size, device=x.device,
                    )
                ).data

    

def rnn_init(module):
    """This function is used to initialize the RNN weight.
    Recurrent connection: orthogonal initialization.
    Arguments
    ---------
    module: torch.nn.Module
        Recurrent neural network module.
    Example
    -------
    >>> inp_tensor = torch.rand([4, 10, 20])
    >>> net = RNN(hidden_size=5, input_shape=inp_tensor.shape)
    >>> out_tensor = net(inp_tensor)
    >>> rnn_init(net)
    """
    for name, param in module.named_parameters():
        if "weight_hh" in name or ".u.weight" in name:
            nn.init.orthogonal_(param)



if __name__ == "__main__" :
    torch.manual_seed(42)
    inp_tensor = torch.rand([4, 1000, 2])
    FLAG = True
    net = LiGRU(input_shape=inp_tensor.shape, hidden_size=128, num_layers=10)
    out_tensor, _= net(inp_tensor) 

    print(out_tensor)

    print(net.lambda_value())