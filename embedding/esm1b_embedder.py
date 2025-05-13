import torch
import esm
from tqdm import tqdm

class ESM1bEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading ESM1b...")
        """34 layer transformer model with 670M params, trained on Uniref50 Sparse. Returns a tuple of (Model, Alphabet).
        """
        self.model, self.alphabet = esm.pretrained.esm1_t34_670M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.model.to(self.device)
        print("ESM1b Loaded and Ready! The device: ", self.device)

    def get_embeddings(self, sequences, batch_size=16): #TODO: change if you have bigger RAM then COLAB!
        embeddings = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting ESM-1b embeddings..."):
            batch_sequences = sequences[i:i+batch_size]
            batch_data = [(f"seq{i}", seq) for i,seq in enumerate(batch_sequences)]
            labels, strs, tokens = self.batch_converter(batch_data)
            tokens = tokens.to(self.device)

            with torch.no_grad():
                results = self.model(tokens, repr_layers = [33], return_contacts = False)
            reps = results["representations"][33]

            for j, (_, seq) in enumerate(batch_data):
                seq_len = len(seq)
                embedding = reps[j, 1:seq_len + 1].mean(0) # TODO: MEAN POOLING TO BE CHANGED! skipping class token
                embeddings.append(embedding.cpu())

        return embeddings




