import torch.nn as nn 
import torch 
import math 
from torch.nn import functional as F

class TimeMix(nn.Module):
    def __init__(self, ctx_len, n_embd, n_layer, dim_att, layer_id):
        super().__init__()
        self.ctx_len = ctx_len
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.dim_att = dim_att
        self.layer_id = layer_id
        
        with torch.no_grad():
            ratio_0_to_1 = self.layer_id / (self.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)
            ddd = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                ddd[0, 0, i] = i / self.n_embd

            # fancy time_decay
            decay_speed = torch.ones(self.dim_att)
            for h in range(self.dim_att):
                decay_speed[h] = -5 + 8 * (h / (self.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(self.dim_att)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(self.dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.value = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.output = nn.Linear(self.dim_att, self.n_embd, bias=False)


        def QKV(self, q, k, v):
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.att_mask == 0, float('-inf'))
            att = F.softmax(att, dim = -1)
            x = att @ v
            return x
        
        def forward(self, x):
            B, T, C = x.size()

            xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
            
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

            xqq = x * self.time_mix_qq + xx * (1 - self.time_mix_qq)
            xkk = x * self.time_mix_kk + xx * (1 - self.time_mix_kk)
            xvv = x * self.time_mix_vv + xx * (1 - self.time_mix_vv)

            k = self.key(xk)
            v = self.value(xv)
            r = self.receptance(xr)
            sr = torch.sigmoid(r)
            qq = self.qq(xqq)
            kk = self.kk(xkk)
            vv = self.vv(xvv)


            return sr, k, v, qq, kk, vv