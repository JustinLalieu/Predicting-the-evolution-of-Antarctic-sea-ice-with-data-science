import torch
import torch.nn as nn

class Persistence(nn.Module):
    def __init__(self, in_channels, 
                 hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False
                 ):
        super(Persistence, self).__init__()
        self.architecture_name = "Persistence"

        self.weights = nn.Parameter(torch.randn(in_channels), requires_grad=True)

    def forward(self, x):
        return x