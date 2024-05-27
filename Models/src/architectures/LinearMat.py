import torch
import torch.nn as nn

class LinearMat(nn.Module):
    def __init__(self, in_channels, 
                 hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False
                 ):
        super(LinearMat, self).__init__()
        self.architecture_name = "LinearMat"

        self.weights_list = nn.ParameterList([nn.Parameter(torch.randn(432, 432), requires_grad=True) for i in range(in_channels)])
        self.bias = nn.Parameter(torch.randn(432, 432), requires_grad=True)

    def forward(self, x):
        #print(f"x.shape: {x.shape}")
        out = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
        #print(f"out.shape: {out.shape}")
        for i in range(x.shape[0]):
            xi = x[i].view(x.shape[1], x.shape[2], x.shape[3])
            #print(f"xi.shape: {xi.shape}")
            xi_accumulated = torch.zeros_like(xi).to(x.device)
            for j in range(x.shape[1]):
                maths = torch.matmul(xi[j], self.weights_list[j])
                #print(f"xi[j].shape: {xi[j].shape}")
                xi_accumulated[j] = maths
            xi_sum = xi_accumulated.sum(dim=0) + self.bias
            #print(f"xi_sum.shape: {xi_sum.shape}")
            out[i] = xi_sum.unsqueeze(0)
            #print(f"out[i].shape: {out[i].shape}")
        #print(f"out.shape: {out.shape}")
        return out