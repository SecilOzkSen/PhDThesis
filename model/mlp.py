import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims:  List[int],
                 output_dim: int,
                 dropout: float = 0.3):
       super(MLP, self).__init__()
       self.model = nn.Sequential()
       layers = []
       all_dims = [input_dim] + hidden_dims

       for i in range(len(hidden_dims)):
           layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
           layers.append(nn.ReLU())
           layers.append(nn.Dropout(dropout))

       layers.append(nn.Linear(hidden_dims[-1], output_dim))
       layers.append(nn.Sigmoid())  # multi-label classification

       self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)





