"""src/utils.py

Helper functions for dataset handling, evaluation and plotting.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


def plot_confusion_matrix(y_true, y_pred, labels=None, savepath=None):
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Add color bar
    plt.colorbar(im, ax=ax)

    # Label ticks
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)

    return fig


def classification_metrics(y_true, y_pred):
    """Return accuracy, precision, recall, and F1 as a dictionary."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
