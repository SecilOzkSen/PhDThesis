import torch
import esm
from tqdm import tqdm
import gc

class ESM1bEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading ESM1b...")
        """34 layer transformer model with 670M params, trained on Uniref50 Sparse. Returns a tuple of (Model, Alphabet).
        """
        self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.model.to(self.device)
        print("ESM1b Loaded and Ready! The device: ", self.device)

    def get_embeddings(self, sequences, batch_size=1, strategy="truncate", max_len=1024):
        assert strategy in {"truncate", "segment"}, "strategy must be 'truncate' or 'segment'"
        embeddings = []
        print(f"Extracting ESM-1b embeddings... (strategy: {strategy})")

        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_seqs = sequences[i:i+batch_size]
            batch_labels = [f"seq{i+j}" for j in range(len(batch_seqs))]
            batch_tuples = list(zip(batch_labels, batch_seqs))

            try:
                all_embs = []

                for label, seq in batch_tuples:
                    seq = seq.strip()
                    if len(seq) <= max_len or strategy == "truncate":
                        truncated_seq = seq[:max_len]
                        tokens = self.batch_converter([(label, truncated_seq)])[2].to(self.device)
                        with torch.no_grad():
                            out = self.model(tokens, repr_layers=[33], return_contacts=False)
                            rep = out["representations"][33][0, 1:len(truncated_seq)+1]
                            emb = rep.mean(dim=0)
                        all_embs.append(emb.cpu())

                    elif strategy == "segment":
                        segment_embs = []
                        for s in range(0, len(seq), max_len):
                            segment = seq[s:s+max_len]
                            if len(segment) == 0:
                                continue
                            tokens = self.batch_converter([(label, segment)])[2].to(self.device)
                            with torch.no_grad():
                                out = self.model(tokens, repr_layers=[33], return_contacts=False)
                                rep = out["representations"][33][0, 1:len(segment)+1]
                                segment_embs.append(rep.mean(dim=0))
                        if segment_embs:
                            emb = torch.stack(segment_embs).mean(dim=0)
                            all_embs.append(emb.cpu())
                        else:
                            print(f"❌ Empty segment embeddings for sequence {label}")
                            continue

                embeddings.extend(all_embs)
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"❌ Skipping batch {i}-{i+batch_size} due to error: {e}")
                continue

        return embeddings




