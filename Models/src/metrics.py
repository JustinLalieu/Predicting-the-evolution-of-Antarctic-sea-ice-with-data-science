import sys
import torch
import torch.nn as nn

sys.path.append("../src")
sys.path.append("../src/utils")
sys.path.append("../src/loaders")
sys.path.append("../src/architectures")

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_pred, y_true, mask=None):
        return torch.mean((y_pred - y_true) ** 2)
    
class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, y_pred, y_true, mask=None):
        if mask is not None:
            device = y_pred.device
            batch_mask = mask.repeat(y_pred.shape[0], 1, 1, 1).to(device)
            y_pred_mask = y_pred * batch_mask
            y_true_mask = y_true * batch_mask
            return torch.mean(torch.abs(y_pred_mask - y_true_mask))
        else:
            return torch.mean(torch.abs(y_pred - y_true))
    
class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, y_pred, y_true, mask=None):
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))
    
class NRMSE(nn.Module):
    def __init__(self):
        super(NRMSE, self).__init__()

    def forward(self, y_pred, y_true, mask=None):
        nrmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)) / torch.std(y_true)
        return nrmse
    
class IIEE(nn.Module):
    def __init__(self):
        super(IIEE, self).__init__()

    def forward(self, y_pred, y_true, mask=None):
        bin_truth = torch.where(y_true > 15, 1, 0)
        bin_pred = torch.where(y_pred > 15, 1, 0)
        xor = torch.logical_xor(bin_truth, bin_pred)
        tsum = torch.sum(xor)
        return tsum
    
class NIIEE(nn.Module):
    def __init__(self):
        super(NIIEE, self).__init__()

    def forward(self, y_pred, y_true, mask=None):
        bin_truth = torch.where(y_true > 15, 1, 0)
        bin_pred = torch.where(y_pred > 15, 1, 0)
        xor = torch.logical_xor(bin_truth, bin_pred)
        union = torch.logical_or(bin_truth, bin_pred)
        return torch.sum(xor) / torch.sum(union)
    
class Hybrid(nn.Module):
    def __init__(self, alpha, beta):
        super(Hybrid, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mae = MAE()
        self.niiee = NIIEE()

    def forward(self, y_pred, y_true, mask=None):
        return self.alpha * self.mae(y_pred, y_true) + self.beta * self.niiee(y_pred, y_true)
    
class BACC(nn.Module):
    def __init__(self):
        super(BACC, self).__init__()

    def forward(self, y_pred, y_true, mask=None):
        max_sie_sum = 37984 #En 1975, 273e jour
        bin_truth = torch.where(y_true > 15, 1, 0)
        bin_pred = torch.where(y_pred > 15, 1, 0)
        xor = torch.logical_xor(bin_truth, bin_pred)
        iiee = torch.sum(xor)
        return 1 - (iiee / max_sie_sum)
    
class NSE(nn.Module):
    def __init__(self):
        super(NSE, self).__init__()

    def forward(self, y_pred, y_true, mask=None):
        return 1 - (torch.sum((y_true - y_pred) ** 2) / torch.sum((y_true - torch.mean(y_true)) ** 2)) # 1 - retiré car on veut minimiser
    
class ANOM_CORR_COEF(nn.Module):
    def __init__(self):
        super(ANOM_CORR_COEF, self).__init__()

    def forward(self, y_pred, y_true, mask=None):
        den = torch.sum((y_true - torch.mean(y_true)) * (y_pred - torch.mean(y_pred)))
        num = torch.sqrt(torch.sum((y_true - torch.mean(y_true)) ** 2) * torch.sum((y_pred - torch.mean(y_pred)) ** 2))
        return (den / num)