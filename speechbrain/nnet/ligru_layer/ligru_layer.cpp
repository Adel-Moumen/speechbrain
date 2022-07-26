#include <torch/extension.h>
#include <vector>

// todo: 
// 1) append chaque timestep dans un vecteur et le renvoyer
// 2) en suivant la meme trame que  1. faire pareil pour les vecteurs des grads
// 3) faire d'abord une version sans layer norm cad ligru 1.0  a bien check avec des gradchecks


#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor undef;

std::tuple<torch::Tensor, torch::Tensor>
ligru_cell_forward(torch::Tensor wx,      // [B, H * 2]
                   torch::Tensor ht_pred, // [B, H]
                   torch::Tensor u,       // [H * 2, H]
                   torch::Tensor drop_mask) {

  CHECK_INPUT(wx);
  CHECK_INPUT(ht_pred);
  CHECK_INPUT(u);

  auto gates = wx + ht_pred.mm(u.t());
  auto chunked_gates = gates.chunk(2, /*dim=*/1);
  auto at = chunked_gates[0];
  auto zt = torch::sigmoid(chunked_gates[1]);
  auto hcand = torch::relu(at) * drop_mask;
  auto ht_next = zt * ht_pred + (1 - zt) * hcand;

  return {ht_next, gates};
}

std::vector<torch::Tensor>
ligru_layer_forward(torch::Tensor wx,        // [B, T, H * 2]
                    torch::Tensor h_init,    // [B, H]
                    torch::Tensor u,         // [H * 2, H]
                    torch::Tensor drop_mask) {

  CHECK_INPUT(wx);
  CHECK_INPUT(h_init);
  CHECK_INPUT(u);
  CHECK_INPUT(drop_mask);

  torch::Tensor ht_next, gates;

  int batch_size = wx.size(0);
  int seq_length = wx.size(1);
  int hidden_size = wx.size(2);

  auto ht = h_init;
  std::vector<torch::Tensor> hiddens = {ht}; 
  for (int t = 0; t < seq_length; ++t) {
    std::tie(ht_next, gates) = ligru_cell_forward(wx.select(1, t).contiguous(), ht, u, drop_mask);
    ht = ht_next;
    hiddens.push_back(ht);
  }

  std::cout << hiddens.size() << std::endl;

  return {gates};
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ligru_cell_backward(torch::Tensor grad_out,
                    torch::Tensor ht, torch::Tensor u, torch::Tensor wx) {

  auto dx = grad_out;
  auto dh = dx.mm(u);
  auto du = dx.t().mm(ht);

  return {dx, dh, du};
}

std::vector<torch::Tensor>
ligru_layer_backward(torch::Tensor grad_out,
                     torch::Tensor ht, 
                     torch::Tensor h_init, 
                     torch::Tensor u, 
                     torch::Tensor wx) {

  CHECK_INPUT(grad_out);
  CHECK_INPUT(ht);
  CHECK_INPUT(u);

  torch::Tensor dx, dh, du;

  int batch_size = wx.size(0);
  int seq_length = wx.size(1);
  int hidden_size = wx.size(2);

  torch::Tensor dwx = wx.new_empty(wx.sizes(), wx.options());
  for (int t = seq_length - 1; t > 0; --t) {
    
    // auto h = (t - 1 < 0) ? h_init : ht.select(1, t - 1).contiguous();
    auto h = h_init;
    std::tie(dx, dh, du) =
        ligru_cell_backward(grad_out, h, u,
                            wx.select(1, t).contiguous());
    dwx.index_put_({at::indexing::Slice(), t}, dx);
  }

  return {dx, dh, du};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ligru_layer_forward, "ligru layer forward");
  m.def("backward", &ligru_layer_backward, "ligru layer backward");
}