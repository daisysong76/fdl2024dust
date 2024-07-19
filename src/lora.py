# lora.py
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.linear_a = nn.Linear(in_features, rank, bias=False)
        self.linear_b = nn.Linear(rank, out_features, bias=False)
        self.scaling = 1 / (rank ** 0.5)

    def forward(self, x):
        return self.linear_b(self.linear_a(x)) * self.scaling
    


# class LoRALayer(nn.Module):
#     def __init__(self, linear_layer, rank=4):
#         super(LoRALayer, self).__init__()
#         self.linear_layer = linear_layer
#         self.rank = rank
#         self.A = nn.Parameter(torch.randn(linear_layer.in_features, rank))
#         self.B = nn.Parameter(torch.randn(rank, linear_layer.out_features))
#         self.scaling = 1 / (rank ** 0.5)

#     def forward(self, x):
#         lora_update = torch.matmul(torch.matmul(x, self.A), self.B)
#         return self.linear_layer(x) + self.scaling * lora_update

