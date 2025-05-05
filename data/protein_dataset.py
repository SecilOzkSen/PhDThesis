import torch
from torch.utils.data import Dataset
from typing import Union, List
import numpy as np
import os

class ProteinDataset(Dataset):
    def __init__(self,
                 embeddings: Union[List[torch.Tensor], np.ndarray],
                 labels: Union[List[List[int]], np.ndarray],
                 model_name: str = "protbert"
                 ):
        '''
        A dataset of protein embeddings and GO term labels.

        :param embeddings: List or array of [embedding_dim] tensors or numpy arrays
        :param labels: Multi one-hot vector for GO terms
        :param model_name: name of the embedding model.
        '''
        self.embeddings = embeddings
        self.labels = self.to_tensor(labels, dtype=torch.float32) ## Tensor'e çevirmesi lazım nn.Module e vermek için.
        self.model_name = model_name

        assert len(embeddings) == len(self.labels), \
            "Mismatch between data and labels"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

    @staticmethod
    def to_tensor(data, dtype=torch.float32):
        if isinstance(data, list):
            data = np.array([x.cpu().numpy() if torch.is_tensor(x) else x for x in data])
        return torch.tensor(data, dtype=dtype)
