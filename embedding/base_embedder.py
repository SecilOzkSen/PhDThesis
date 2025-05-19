from abc import ABC, abstractmethod

import torch.cuda


class BaseEmbedder(ABC):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def encode_batch(self, labels: list[str], sequences: list[str]):
        '''
        Returns:
        - token_reps: List[Tensor] of shape [seq_len, dim] for each sequence
        - attention: List[Tensor] of shape [layers, heads, seq_len, seq_len] or None
        '''
        pass

    def get_embeddings(self, sequences, batch_size=1,  strategy="mean", max_len=1024, use_watti=False):
        '''
        Compute embeddings with optional attention-based pooling
        '''
        embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            processed_seqs = [seq[:max_len].strip() for seq in batch_seqs]
            labels = [f"seq{i+j}" for j in range(len(processed_seqs))]
            token_reps_batch, attn_batch = self.encode_batch(labels, processed_seqs)
            for j in range(len(processed_seqs)):
                reps = token_reps_batch[j]
                attn = attn_batch[j] if attn_batch is not None else None

                if use_watti and attn is not None:
                    emb = self._watti_pool(attn, reps)
                else:
                    emb = reps.mean(dim=0)
                embeddings.append(emb.cpu())

        return embeddings
