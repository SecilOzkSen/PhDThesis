import os
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from config.config import DatasetConfig
import pickle

from data.loader import load_cafa5_dataframe
from embedding.protbert_embedding import ProtBERTEmbedder
from embedding.esm1b_embedder import ESM1bEmbedder
from config.config import PLM_Config, ModelConfig
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def extract_and_save_embeddings(model_name, batch_size, partition_size=10000, use_watti=False, save_dir="partitions"):
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
    #    embedding_path = DatasetConfig.PROTBERT_EMBEDDING_PATH
    elif model_name == PLM_Config.ESM1B:
        embedder = ESM1bEmbedder(use_watti=use_watti)
    #    embedding_path = DatasetConfig.ESM1B_EMBEDDING_PATH
    else:
        raise ValueError("Unsupported Model Type!!")

    print("Extracting Embeddings....")
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(labels)
    for i in tqdm(range(0, len(sequences), partition_size)):
        sequences_part = sequences[i:i + partition_size]
        sequence_ids_part = sequence_ids[i:i+partition_size]
        labels_part = label_matrix[i:i+partition_size, :]
        embedding_data = []
        try:
            embeddings_part = embedder.get_embeddings(sequences_part,
                                                 strategy="truncate",
                                                 batch_size=batch_size,
                                                 max_len=PLM_Config.MAX_SEQ_LEN)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            continue

        filename = f"{DatasetConfig.DATA_DIR}/{model_name}_embeddings_part_{i // partition_size}.pkl"
        for seq_id, seq, emb, label in zip(sequence_ids_part, sequences_part, embeddings_part, labels_part):
            embedding_data.append({
                "sequence_id": seq_id,
                "sequence": seq,
                "embedding": emb.cpu(),  # save CPU tensor
                "label": label  # already numpy array
            })
        with open(filename, "wb") as f:
            pickle.dump(embedding_data, f)




if __name__ == "__main__":
    os.makedirs(DatasetConfig.DATA_DIR, exist_ok=True)
    extract_and_save_embeddings(PLM_Config.PROTBERT, 5 )
    #extract_and_save_embeddings_protbert()
