import numpy as np
from data.protein_dataset import ProteinDataset
import torch
from torch.utils.data import DataLoader
import os


def save_as_cafa_pred_file(predictions, sequence_ids, go_terms, file_path, author="Secil Sen"):
    with open(file_path, "w") as f:
        f.write(f"AUTHOR: {author}\n")
        f.write("MODEL: 1\n")
        f.write("KEYWORDS MLP, PROTBERT\n\n")

        for i, seq_id in enumerate(sequence_ids):
            for j, go_term in enumerate(go_terms):
                score = predictions[i][j]
                if score > 0:
                    f.write(f"{seq_id}\t{go_term}\t{score:.4f}\n")

def predict_and_export_cafa_pred(model, test_embeddings, sequence_ids, go_terms, save_path, batch_size=32, device="cpu"):
    print("Preparing test loader...")

    # dummy labels - not used but needed for dataset class.
    dummy_labels = np.zeros((len(test_embeddings), len(go_terms)))
    test_dataset = ProteinDataset(test_embeddings, dummy_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Now running prediction on test set...")
    model.eval()
    all_outputs = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu())

    predictions = torch.cat(all_outputs).numpy()

    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving predictions in CAFA format: {save_path}")
    save_as_cafa_pred_file(predictions, sequence_ids, go_terms, save_path)
