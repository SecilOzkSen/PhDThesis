import torch
from transformers import BertModel, BertTokenizer
from data.preprocessing import CAFA5_Preprocessor
from config.config import PLM_Config
from typing import List
from tqdm import tqdm
from embedding.base_embedder import BaseEmbedder
class ProtBERTEmbedder(BaseEmbedder):
    def __init__(self,
                 model_name: str = 'Rostlab/prot_bert',
                 model_path: str = None,
                 device: str = None,
                 truncation_strategy = "truncate"):
        super().__init__(device=device)
        self.truncation_strategy = truncation_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        if model_path is None:
            self.model = BertModel.from_pretrained(model_name, force_download=True).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        else:
            self.model = BertModel.from_pretrained(model_path, force_download=True).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        self.cafa5_preprocessor = CAFA5_Preprocessor(plm_name=model_name)
        self.model.eval()

    def encode_batch(self, labels: list[str], sequences: list[str]):
        truncate = True if self.truncation_strategy == "truncate" else False
        formatted = [self.cafa5_preprocessor.preprocess_sequence(seq, clean=True, truncate=truncate) for seq in sequences]
        tokens = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=PLM_Config.MAX_SEQ_LEN
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            out = self.model(**tokens)
            reps = out.last_hidden_state

        attention_mask = tokens["attention_mask"]
        token_reps_list = []

        for i, seq in enumerate(sequences):
            seq_len = attention_mask[i].sum().item() - 2  # exclude CLS and SEP
            rep = reps[i, 1:1 + seq_len]  # skip CLS and SEP
            token_reps_list.append(rep)

        return token_reps_list, [None] * len(token_reps_list)  # Always return dummy attn


    def get_embeddings(self, sequences, batch_size=1,  strategy="mean", max_len=1024, use_watti=False):
        embeddings = []
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i + batch_size]
            labels = [f"seq{i + j}" for j in range(len(batch_seqs))]
            token_reps_list, _ = self.encode_batch(labels, batch_seqs)

            for rep in token_reps_list:
                emb = rep.mean(dim=0)
                embeddings.append(emb.cpu())

        return embeddings

