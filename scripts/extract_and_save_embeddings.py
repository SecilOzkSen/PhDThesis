import os
import torch
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from config.config import DatasetConfig
from huggingface_hub import snapshot_download

from data.loader import load_cafa5_dataframe
from embedding.protbert_embedding import ProtBERTEmbedder
from embedding.esm1b_embedder import ESM1bEmbedder
from config.config import PLM_Config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def extract_and_save_embeddings(model_name):
    print(torch.cuda.is_available())
    print("Loading sequences and terms....")
    df = load_cafa5_dataframe(model_name=model_name)

    sequences = df["sequence"].tolist()
    labels = df["term"].tolist()

    print(f"Total sequences: {len(sequences)}")
    print(f"Unique GO terms: {len(set([t for sublist in labels for t in sublist]))}")

    print(f"🤖 Initializing embedder: {model_name}")

    if model_name == PLM_Config.PROTBERT:
        embedder = ProtBERTEmbedder()
        embedding_path = DatasetConfig.PROTBERT_EMBEDDING_PATH
    elif model_name == PLM_Config.ESM1B:
        embedder = ESM1bEmbedder()
        embedding_path = DatasetConfig.ESM1B_EMBEDDING_PATH
    else:
        raise ValueError("Unsupported Model Type!!")

    print("Extracting Embeddings....")
    embeddings = embedder.get_embeddings(sequences, batch_size=DatasetConfig.BATCH_SIZE)

    print("Saving Embeddings....")
    os.makedirs("dataset", exist_ok=True)
    embeddings_cpu = [e.cpu() for e in embeddings]
    np.save(embedding_path, torch.stack(embeddings_cpu).numpy())

    print("Saving labels....")
    print("🏷️  Saving labels...")
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(labels)
    np.save(DatasetConfig.LABELS_PATH, label_matrix)
    np.save(DatasetConfig.GO_TERM_PATH, mlb.classes_)
    print("Embeddings and labels saved!")
    print(f"DONE! saved embeddings and the labels to {embedding_path} folder.")

def extract_and_save_embeddings_protbert():
    print(torch.cuda.is_available())
    print("Loading sequences and terms....")
    df = load_cafa5_dataframe()
    print("Model downloading....")
    local_dir = snapshot_download(repo_id="Rostlab/prot_bert")
    print(f"Downloaded full model + tokenizer to: {local_dir}")
    sequences = df["sequence"].tolist()
    labels = df["term"].tolist()
    print(f"Total Sequences:{len(sequences)}")
    print("Initializing ProtBERT embedder...")
    embedder = ProtBERTEmbedder(model_path=local_dir)
    embeddings = embedder.get_embeddings(sequences, batch_size=DatasetConfig.BATCH_SIZE)

    print("Converting and saving embeddings...")
    embeddings_np = torch.stack(embeddings)
    np.save(DatasetConfig.PROTBERT_EMBEDDING_PATH, embeddings_np.numpy())

    print("Binarizing and saving labels....")
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(labels)
    np.save(DatasetConfig.LABELS_PATH, label_matrix)

    print("Saving go term classes....")
    np.save(DatasetConfig.GO_TERM_PATH, mlb.classes_)

    print(f"DONE! saved embeddings and the labels to {DatasetConfig.DATA_DIR} folder.")

if __name__ == "__main__":
    os.makedirs(DatasetConfig.DATA_DIR, exist_ok=True)
    extract_and_save_embeddings("esm1b")
