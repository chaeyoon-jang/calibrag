import numpy as np
from sklearn.metrics import roc_auc_score

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    Parameters:
    y_true (array-like): True labels (0 or 1)
    y_prob (array-like): Predicted probabilities/confidences
    n_bins (int): Number of bins (default 10)

    Returns:
    ece (float): Expected Calibration Error
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True) - 1

    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_mask = bin_indices == i
        bin_size = np.sum(bin_mask)

        if bin_size > 0:
            avg_confidence = np.mean(y_prob[bin_mask])
            avg_accuracy = np.mean(y_true[bin_mask])
            ece += (bin_size / n) * np.abs(avg_confidence - avg_accuracy)

    return ece

def compute_brier_score(y_true, y_prob):
    """
    Compute Brier score.

    Parameters:
    y_true (array-like): True labels (0 or 1)
    y_prob (array-like): Predicted probabilities/confidences

    Returns:
    brier_score (float): Brier score
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    brier_score = np.mean((y_prob - y_true) ** 2)
    return brier_score

def compute_auroc(y_true, y_prob):
    """
    Compute AUROC.

    Parameters:
    y_true (array-like): True labels (0 or 1)
    y_prob (array-like): Predicted probabilities/confidences

    Returns:
    auroc (float): Area Under the Receiver Operating Characteristic Curve
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    return auroc

def compute_nll(y_true, y_prob):
    """
    Compute Negative Log-Likelihood (NLL).

    Parameters:
    y_true (array-like): True labels (0 or 1)
    y_prob (array-like): Predicted probabilities/confidences

    Returns:
    nll (float): Negative Log-Likelihood
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    nll = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return nll