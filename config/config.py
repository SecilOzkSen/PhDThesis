import os
import torch
class DatasetConfig:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "dataset")
    SEQ_FILE = os.path.join(DATA_DIR, "train_sequences.csv")
    TERM_FILE = os.path.join(DATA_DIR, "train_terms.csv")

    # Training hyperparameters
    BATCH_SIZE = 32
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