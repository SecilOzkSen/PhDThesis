import os
import torch
class DatasetConfig:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "dataset")
    SEQ_FILE = os.path.join(DATA_DIR, "Train/train_sequences.fasta")
    TERM_FILE = os.path.join(DATA_DIR, "Train/train_terms.tsv")
    PROTBERT_EMBEDDING_PATH = os.path.join(DATA_DIR, "protbert_embeddings.npy")
    ESM1B_EMBEDDING_PATH = os.path.join(DATA_DIR, "esm1b_embeddings.npy")
    LABELS_PATH = os.path.join(DATA_DIR, "train_labels.npy")
    GO_TERM_PATH = os.path.join(DATA_DIR, "go_terms.npy")
    SEQ_IDS_PATH = os.path.join(DATA_DIR, "seq_ids.npy")
    PROTBERT_TEST_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "protbert_test_embeddings.npy")

class PLM_Config:
    PROTBERT = "protbert"
    ESM1B = "esm1b"
    VALID_AMINO_ACIDS_PROTBERT = set("ACDEFGHIKLMNPQRSTVWY")
    MAX_SEQ_LEN_PROTBERT = 1024

class ModelConfig:
    # Training hyperparameters
    TRAIN_TEST_SPLIT = 0.1
    LR = 1e-4
    EPOCHS = 1
    BATCH_SIZE = 32
    HIDDEN_DIMS = [512, 256]
    DROPOUT = 0.3
    SAVE_PATH = "saved_models/protbert_mlp_best_model.pt"

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
