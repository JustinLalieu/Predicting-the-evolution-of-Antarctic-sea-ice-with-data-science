import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_channels, 
                 hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False
                 ):
        super(Linear, self).__init__()
        self.architecture_name = "Linear"

        #initial_weights = [0.03, 0.47, 0.51]
        #self.weights = nn.Parameter(torch.tensor(initial_weights), requires_grad=True)
        self.weights = nn.Parameter(torch.randn(in_channels), requires_grad=True)
        #self.bias = nn.Parameter(torch.randn(1), requires_grad=True)
        self.bias = 0

    def forward(self, x):
        print(self.weights, self.bias)
        out = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
        for i in range(x.shape[0]):
            xi = x[i].view(x.shape[1], x.shape[2], x.shape[3])
            xi_accumulated = torch.zeros_like(xi).to(x.device)
            for j in range(x.shape[1]):
                xi_accumulated[j] = xi[j] * self.weights[j]
            xi_sum = xi_accumulated.sum(dim=0) + self.bias
            out[i] = xi_sum.unsqueeze(0)
        return out