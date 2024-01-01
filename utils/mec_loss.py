import torch
from torch import nn
import torch.nn.functional as F

def loss_func(p, z, lamda_inv, order=4):
    p = F.normalize(p)
    z = F.normalize(z)

    c = p @ z.T

    c = c / lamda_inv 

    power_matrix = c
    sum_matrix = torch.zeros_like(power_matrix)

    for k in range(1, order+1):
        if k > 1:
            power_matrix = torch.matmul(power_matrix, c)
        if (k + 1) % 2 == 0:
            sum_matrix += power_matrix / k
        else: 
            sum_matrix -= power_matrix / k

    trace = torch.trace(sum_matrix)

    return trace

class MECLoss(nn.Module):
    def forward(self, p1, p2, z1, z2, lamda_inv)->torch.Tensor:
        return (loss_func(p1, z2, lamda_inv) + loss_func(p2, z1, lamda_inv)) * 0.5 / p1.shape[0]