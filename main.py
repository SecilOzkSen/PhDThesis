import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from utils.cafa_formatter import predict_and_export_cafa_pred

from config.config import DatasetConfig, ModelConfig
from data.protein_dataset import ProteinDataset
from model.mlp import MLP
from train.trainer import Trainer
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def load_best_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model



def main_mlp():
    print("Loading embeddings and labels..")

    embeddings = np.load(DatasetConfig.PROTBERT_EMBEDDING_PATH)
    label_path = DatasetConfig.LABELS_PATH
    if os.path.exists(label_path):
        labels = np.load(label_path)
    elif os.path.exists(label_path.replace(".npy", ".npz")):
        labels = np.load(label_path.replace(".npy", ".npz"))
        labels = labels["labels"]
    else:
        raise FileNotFoundError("Neither .npy nor .npz label file found.")

    # Train / val split
    # Stratified K-Fold (first fold only)
    print("Loading labels are done!")
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in mskf.split(embeddings, labels):
        X_train, X_val = embeddings[train_index], embeddings[val_index]
        Y_train, Y_val = labels[train_index], labels[val_index]
        break

    train_dataset = ProteinDataset(X_train, Y_train)
    val_dataset = ProteinDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, batch_size=ModelConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=ModelConfig.BATCH_SIZE)

    # Model Setup
    model = MLP(input_dim=embeddings.shape[1],
                output_dim=labels.shape[1],
                hidden_dims=ModelConfig.HIDDEN_DIMS,
                dropout=ModelConfig.DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=ModelConfig.LR)
    #Trainer setup
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      device=ModelConfig.DEVICE)
    best_fmax = 0.0
    os.makedirs(os.path.dirname(ModelConfig.SAVE_PATH), exist_ok=True)

    for epoch in range(1, ModelConfig.EPOCHS + 1):
        print(f"\n Epoch {epoch}/{ModelConfig.EPOCHS}")

        train_loss = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)

        print(f"Train Loss {train_loss:.4f} | Val Loss {val_metrics['val_loss']:.4f}")

        if val_metrics['fmax'] > best_fmax:
            best_fmax = val_metrics['fmax']
            torch.save(model.state_dict(), ModelConfig.SAVE_PATH)
            print("Best Model Saved. Fmax improved.")

    ## Final Evaluation
    best_model = load_best_model(model, ModelConfig.SAVE_PATH, ModelConfig.DEVICE)

    predict_and_export_cafa_pred(
        model=best_model,
        test_embeddings=np.load(DatasetConfig.PROTBERT_TEST_EMBEDDINGS_PATH),
        sequence_ids=np.load(DatasetConfig.SEQ_IDS_PATH),
        go_terms=np.load(DatasetConfig.GO_TERM_PATH),
        save_path="cafa_outputs/test_predictions.pred",
        batch_size=ModelConfig.BATCH_SIZE,
        device=ModelConfig.DEVICE
    )

if __name__ == '__main__':
    main_mlp()