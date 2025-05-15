import os
import torch
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from config.config import DatasetConfig
from huggingface_hub import snapshot_download

from data.loader import load_cafa5_dataframe
from embedding.protbert_embedding import ProtBERTEmbedder
from embedding.esm1b_embedder import ESM1bEmbedder
from config.config import PLM_Config, ModelConfig
import traceback

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
    embeddings = embedder.get_embeddings(sequences, batch_size=ModelConfig.BATCH_SIZE)

    print("Saving Embeddings....")
    os.makedirs("dataset", exist_ok=True)
    embeddings_cpu = [e.cpu() for e in embeddings]
    np.save(embedding_path, torch.stack(embeddings_cpu).numpy())

    print("Binarizing and saving labels....")
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(labels)
    try:
        # Daha küçük boyut için dtype dönüşümü
        label_matrix = label_matrix.astype(np.uint8)

        # Önce .npy olarak dene
        np.save(DatasetConfig.LABEL_PATH, label_matrix)
        print(f"✅ Labels saved to {DatasetConfig.LABEL_PATH} (.npy)")

    except Exception as e:
        print("⚠️ .npy save failed, falling back to .npz")
        traceback.print_exc()

        # Fallback: .npz ile sıkıştırılmış güvenli kayıt
        fallback_path = DatasetConfig.LABEL_PATH.replace(".npy", ".npz")
        np.savez_compressed(fallback_path, labels=label_matrix)
        print(f"✅ Labels saved to {fallback_path} (.npz fallback)")
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
 #   embedder = ProtBERTEmbedder(model_path=local_dir)
 #   embeddings = embedder.get_embeddings(sequences, batch_size=DatasetConfig.BATCH_SIZE)

    print("Converting and saving embeddings...")
  #  embeddings_np = torch.stack(embeddings)
  #  np.save(DatasetConfig.PROTBERT_EMBEDDING_PATH, embeddings_np.numpy())

    print("Binarizing and saving labels....")
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(labels)
    try:
        # Daha küçük boyut için dtype dönüşümü
        label_matrix = label_matrix.astype(np.uint8)

        # Önce .npy olarak dene
        np.save(DatasetConfig.LABELS_PATH, label_matrix)
        print(f"✅ Labels saved to {DatasetConfig.LABELS_PATH} (.npy)")

    except Exception as e:
        print("⚠️ .npy save failed, falling back to .npz")
       # traceback.print_exc()

        # Fallback: .npz ile sıkıştırılmış güvenli kayıt
        fallback_path = DatasetConfig.LABELS_PATH.replace(".npy", ".npz")
        np.savez_compressed(fallback_path, labels=label_matrix)
        print(f"✅ Labels saved to {fallback_path} (.npz fallback)")

    print("Saving go term classes....")
    np.save(DatasetConfig.GO_TERM_PATH, mlb.classes_)

    print(f"DONE! saved embeddings and the labels to {DatasetConfig.DATA_DIR} folder.")

if __name__ == "__main__":
    os.makedirs(DatasetConfig.DATA_DIR, exist_ok=True)
    extract_and_save_embeddings(PLM_Config.ESM1B)
    #extract_and_save_embeddings_protbert()
