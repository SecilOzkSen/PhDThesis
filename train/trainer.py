import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional
from eval.evaluator import compute_aupr, compute_fmax

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.optimizer,
                 loss_fn: Optional[nn.Module] = None,
                 device: Optional[str] = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn if loss_fn else nn.BCELoss()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def train_epoch(self, data_loader: DataLoader) -> float:
        '''
        trains only one epoch
        :param data_loader: data loader
        :return: average loss
        '''

        self.model.train()
        total_loss = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss/len(data_loader)
        return avg_loss

    def validate(self, data_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in data_loader:
                all_targets.append(targets.cpu())
                inputs, targets = inputs.to(self.device),  targets.to(self.device)
                outputs = self.model(inputs)
                all_preds.append(outputs.cpu())
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_preds).numpy()

        f_max = compute_fmax(y_true, y_pred)
        aupr = compute_aupr(y_true, y_pred)

        avg_loss = total_loss / len(data_loader)
        return {'val_loss': avg_loss, 'fmax': f_max, 'aupr': aupr}



