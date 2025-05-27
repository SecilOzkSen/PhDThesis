import torch
import esm
from tqdm import tqdm
import gc
from embedding.base_embedder import BaseEmbedder

class ESM1bEmbedder(BaseEmbedder):
    def __init__(self, device=None, use_watti=False):
        super().__init__(device=device)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading ESM1b...")
        """34 layer transformer model with 670M params, trained on Uniref50 Sparse. Returns a tuple of (Model, Alphabet).
        """
        self.use_watti = use_watti
        self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.to(self.device).eval()
        print("ESM1b Loaded and Ready! The device: ", self.device)

    def encode_batch(self, labels: list[str], sequences: list[str]):
        batch_data = list(zip(sequences, labels))
        tokens = self.batch_converter(batch_data)[2].to(self.device)
        with torch.no_grad():
            out = self.model(
                tokens,
                repr_layers=[33],
                return_contacts=False,
                need_head_weights=self.use_watti
            )
            reps = out["representations"][33]  # [batch_size, seq_len, dim]
            attentions = out["attentions"] if self.use_watti else None

        token_reps_list = []
        attention_list = []

        for i, (_, seq) in enumerate(batch_data):
            seq_len = len(seq)
            rep = reps[i, 1:seq_len + 1]  # skip [CLS] token
            token_reps_list.append(rep)

            if self.use_watti:
                attn = attentions[-1][:, i, 0, 1:seq_len + 1]  # shape: [heads, tokens]
                attention_list.append(attn)

        return token_reps_list, attention_list if self.use_watti else [None] * len(token_reps_list)

    def get_embeddings(self, sequences, batch_size=1, strategy="mean", max_len=1024):
        embeddings = []
        print(f"Extracting ESM-1b embeddings... (strategy: {strategy})")

        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i+batch_size]
            processed_seqs = [seq.strip()[:max_len] for seq in batch_seqs]
            labels = [f"seq{i + j}" for j in range(len(processed_seqs))]
            token_reps_batch, attn_batch = self.encode_batch(labels, processed_seqs)

            for reps, attn in zip(token_reps_batch, attn_batch):
                if self.use_watti and attn is not None:
                    attn_weights = attn.mean(0)
                    attn_weights = attn_weights / attn_weights.sum()
                    emb = (reps.T * attn_weights).T.sum(dim=0)
                else:
                    emb = reps.mean(dim=0)
                embeddings.append(emb.cpu())
        return embeddings






