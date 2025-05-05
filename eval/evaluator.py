import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

def compute_aupr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
    Macro averaged AUPR - across GO terms
    :param y_true: true values
    :param y_pred: predicted values
    :return: area under precision recall curve
    '''
    return average_precision_score(y_true, y_pred, average='macro')

def compute_fmax(y_true: np.ndarray, y_pred: np.ndarray, thresholds=None) -> float:
    '''
    Compute maximum F1 across the thresholds
    '''
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    fmax = 0.0
    for t in thresholds:
        binarized = (y_pred >= t).astype(int)
        f1 = f1_score(y_true, binarized, average='micro', zero_division=0)
        fmax = max(fmax, f1)
    return fmax


