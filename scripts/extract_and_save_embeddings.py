import os
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from config.config import DatasetConfig
import pickle

from data.loader import load_cafa5_dataframe
from embedding.protbert_embedding import ProtBERTEmbedder
from embedding.esm1b_embedder import ESM1bEmbedder
from config.config import PLM_Config, ModelConfig
import traceback

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def extract_and_save_embeddings(model_name, batch_size, use_watti=False):
    print(torch.cuda.is_available())
    print("Loading sequences and terms....")
    df = load_cafa5_dataframe(model_name=model_name)

    sequences = df["sequence"].tolist()
    sequence_ids = df["sequence_id"].tolist()
    labels = df["term"].tolist()

    print(f"Total sequences: {len(sequences)}")
    print(f"Unique GO terms: {len(set([t for sublist in labels for t in sublist]))}")

    print(f"🤖 Initializing embedder: {model_name}")

    if model_name == PLM_Config.PROTBERT:
        embedder = ProtBERTEmbedder()
        embedding_path = DatasetConfig.PROTBERT_EMBEDDING_PATH
    elif model_name == PLM_Config.ESM1B:
        embedder = ESM1bEmbedder(use_watti=use_watti)
        embedding_path = DatasetConfig.ESM1B_EMBEDDING_PATH
    else:
        raise ValueError("Unsupported Model Type!!")

    print("Extracting Embeddings....")
    embeddings = embedder.get_embeddings(sequences,
                                         strategy="truncate",
                                         batch_size=batch_size,
                                         max_len=PLM_Config.MAX_SEQ_LEN)

    print("Saving Embeddings....")
    os.makedirs("dataset", exist_ok=True)

    print("Binarizing and saving labels....")
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(labels)

    print("Saving data as .pickle....")
    os.makedirs("dataset", exist_ok=True)

    embedding_data = []
    for seq_id, seq, emb, label in zip(sequence_ids, sequences, embeddings, label_matrix):
        embedding_data.append({
            "sequence_id": seq_id,
            "sequence": seq,
            "embedding": emb.cpu(),  # save CPU tensor
            "label": label  # already numpy array
        })

    with open(embedding_path, "wb") as f:
        pickle.dump(embedding_data, f)

    print(f"✅ Data saved to {embedding_path} (.pickle)")
    print(f"DONE! Extracted embeddings and saved all to {embedding_path}.")



if __name__ == "__main__":
    os.makedirs(DatasetConfig.DATA_DIR, exist_ok=True)
    extract_and_save_embeddings(PLM_Config.PROTBERT, 2 )
    #extract_and_save_embeddings_protbert()
