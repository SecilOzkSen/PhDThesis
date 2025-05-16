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

    def get_embeddings(self, sequences, batch_size=16): #TODO: change if you have bigger RAM then COLAB!
        embeddings = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting ESM-1b embeddings..."):
            batch_sequences = sequences[i:i+batch_size]
            batch_labels = [f"seq{i + j}" for j in range(len(batch_sequences))]
            batch_tuples = list(zip(batch_labels, batch_sequences))
            try:
                batch_tokens = self.batch_converter(batch_tuples)[2].to(self.device)

                with torch.no_grad():
                    results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
                    token_reps = results["representations"][33]

                    for j, (_, seq) in enumerate(batch_tuples):
                        seq_len = len(seq)
                        emb = token_reps[j, 1:seq_len + 1].mean(0)
                        embeddings.append(emb.cpu())  # GPU'dan al

                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"❌ Skipping batch {i}-{i + batch_size} due to error: {e}")
                continue

        return embeddings




