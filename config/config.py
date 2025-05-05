import os
import torch
class DatasetConfig:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "dataset")
    SEQ_FILE = os.path.join(DATA_DIR, "Train/train_sequences.fasta")
    TERM_FILE = os.path.join(DATA_DIR, "Train/train_terms.tsv")
    PROTBERT_EMBEDDING_PATH = os.path.join(DATA_DIR, "protbert_embeddings.npy")
    LABELS_PATH = os.path.join(DATA_DIR, "train_labels.npy")
    GO_TERM_PATH = os.path.join(DATA_DIR, "go_terms.npy")

    # Training hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 1e-4

    # Embedding
    EMBEDDING_DIM = 1024

    # Model
    HIDDEN_DIM_1 = 512
    HIDDEN_DIM_2 = 256
    DROPOUT = 0.3

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PLM_Config:
    VALID_AMINO_ACIDS_PROTBERT = set("ACDEFGHIKLMNPQRSTVWY")
    MAX_SEQ_LEN_PROTBERT = 1024