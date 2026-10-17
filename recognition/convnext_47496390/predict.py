from train import train_test_baseline, pipeline
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
import torch

# https://share.gemini.google/0E2OlinoIE2n
def plot_training_and_test_curves(
    train_losses,
    train_accuracies,
    val_losses,
    val_accuracies,
    val_epochs,
    test_loss=None,
    test_accuracy=None,
    save_path="training_test_curves.png",
):
  """Plots training and validation curves alongside final test performance markers."""
  train_epochs = list(range(1, len(train_losses) + 1))
  total_epochs = len(train_losses)

  fig, axes = plt.subplots(1, 2, figsize=(14, 5))

  # 1. Loss Curve
  axes[0].plot(train_epochs, train_losses, label="Train Loss", marker="o")
  axes[0].plot(val_epochs, val_losses, label="Val Loss", marker="s")
  if test_loss is not None:
    axes[0].axhline(
        y=test_loss,
        color="crimson",
        linestyle="--",
        alpha=0.8,
        label=f"Test Loss ({test_loss:.4f})",
    )
    axes[0].scatter(
        [total_epochs],
        [test_loss],
        color="crimson",
        s=80,
        zorder=5,
        marker="*",
    )

  axes[0].set_title("Loss vs. Epochs", fontsize=14)
  axes[0].set_xlabel("Epoch", fontsize=12)
  axes[0].set_ylabel("Loss", fontsize=12)
  axes[0].set_xticks(train_epochs)
  axes[0].legend()
  axes[0].grid(True, linestyle="--", alpha=0.6)

  # 2. Accuracy Curve
  axes[1].plot(
      train_epochs, train_accuracies, label="Train Accuracy", marker="o"
  )
  axes[1].plot(val_epochs, val_accuracies, label="Val Accuracy", marker="s")
  if test_accuracy is not None:
    axes[1].axhline(
        y=test_accuracy,
        color="crimson",
        linestyle="--",
        alpha=0.8,
        label=f"Test Acc ({test_accuracy:.2f}%)",
    )
    axes[1].scatter(
        [total_epochs],
        [test_accuracy],
        color="crimson",
        s=80,
        zorder=5,
        marker="*",
    )

  axes[1].set_title("Accuracy vs. Epochs", fontsize=14)
  axes[1].set_xlabel("Epoch", fontsize=12)
  axes[1].set_ylabel("Accuracy (%)", fontsize=12)
  axes[1].set_xticks(train_epochs)
  axes[1].legend()
  axes[1].grid(True, linestyle="--", alpha=0.6)

  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=300)
    print(f"Curves with test results saved to {save_path}")


def plot_split_comparison_bar(
    train_acc,
    val_acc,
    test_acc,
    train_loss,
    val_loss,
    test_loss,
    save_path="split_comparison.png",
):
  """Side-by-side bar chart comparing final Train, Best Validation, and Test metrics."""
  splits = ["Train (Final)", "Val (Best)", "Test"]
  accuracies = [train_acc, val_acc, test_acc]
  losses = [train_loss, val_loss, test_loss]

  fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

  # Accuracy comparison
  bars_acc = axes[0].bar(
      splits, accuracies, color=["#4A90E2", "#50E3C2", "#F5A623"], width=0.55
  )
  axes[0].set_title("Accuracy Across Splits (%)", fontsize=13)
  axes[0].set_ylabel("Accuracy (%)", fontsize=11)
  axes[0].set_ylim(0, 105)
  axes[0].grid(axis="y", linestyle="--", alpha=0.6)
  for bar in bars_acc:
    height = bar.get_height()
    axes[0].annotate(
        f"{height:.2f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

  # Loss comparison
  bars_loss = axes[1].bar(
      splits, losses, color=["#4A90E2", "#50E3C2", "#F5A623"], width=0.55
  )
  axes[1].set_title("Loss Across Splits", fontsize=13)
  axes[1].set_ylabel("Loss", fontsize=11)
  axes[1].grid(axis="y", linestyle="--", alpha=0.6)
  for bar in bars_loss:
    height = bar.get_height()
    axes[1].annotate(
        f"{height:.4f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=300)
    print(f"Split comparison bar chart saved to {save_path}")


def plot_confusion_matrix(
    y_true, y_pred, class_names=("NC", "AD"), save_path="baseline_test_cm.png"
):
  """Plots and annotates the test set confusion matrix."""
  cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

  fig, ax = plt.subplots(figsize=(5.5, 4.5))
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  ax.set_xticks(range(len(class_names)))
  ax.set_yticks(range(len(class_names)))
  ax.set_xticklabels(class_names)
  ax.set_yticklabels(class_names)
  ax.set_xlabel("Predicted Label", labelpad=8)
  ax.set_ylabel("True Label")
  ax.set_title("Test Confusion Matrix", pad=15)

  thresh = cm.max() / 2.0
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      ax.text(
          j,
          i,
          str(cm[i, j]),
          ha="center",
          va="center",
          color="white" if cm[i, j] > thresh else "black",
          fontsize=12,
          fontweight="bold",
      )

  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

def plot_roc_curve(
    y_true, y_scores, pos_label=1, save_path="baseline_test_roc.png"
):
  """Calculates and plots the test ROC curve and Area Under Curve (AUC)."""
  fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
  roc_auc = auc(fpr, tpr)

  fig, ax = plt.subplots(figsize=(6, 5))
  ax.plot(
      fpr,
      tpr,
      color="darkorange",
      lw=2,
      label=f"Test ROC curve (AUC = {roc_auc:.3f})",
  )
  ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance")

  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.05])
  ax.set_xlabel("False Positive Rate (1 - Specificity)")
  ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
  ax.set_title("Receiver Operating Characteristic (ROC)")
  ax.legend(loc="lower right")
  ax.grid(True, linestyle="--", alpha=0.5)

  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

def plot_misclassified_grid(
    misclassified_samples,
    class_names=("NC", "AD"),
    max_samples=8,
    ncols=4,
    save_path="misclassified_samples.png",
):
  """Plots a grid of misclassified MRI scans showing true vs.

  pred labels and confidence.
  """
  n_samples = min(len(misclassified_samples), max_samples)
  if n_samples == 0:
    print("No misclassified samples found to plot.")
    return

  nrows = int(np.ceil(n_samples / ncols))
  fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.8 * nrows))

  # Flatten axes if grid is 1D or single subplot
  if n_samples == 1:
    axes = np.array([axes])
  axes = axes.flatten()

  for idx in range(n_samples):
    sample = misclassified_samples[idx]
    img = sample["image"]
    true_label = class_names[sample["true"]]
    pred_label = class_names[sample["pred"]]
    confidence = sample["confidence"] * 100

    # Handle Grayscale (1, H, W) or multi-channel tensors
    if isinstance(img, torch.Tensor):
      img = img.detach().cpu().numpy()
    
    if img.shape[0] == 1: # grayscale: mean 0.5, std 0.5
        img = img.squeeze(0) * 0.5 + 0.5
    elif img.shape[0] == 3: # ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
        img = np.transpose(img * std + mean, (1, 2, 0))

    img = np.clip(img, 0, 1)

    axes[idx].imshow(img, cmap="gray")
    axes[idx].axis("off")

    # Flag False Positives (NC classified as AD) vs False Negatives (AD classified as NC)
    error_type = (
        "False Positive" if pred_label == "AD" else "False Negative"
    )
    axes[idx].set_title(
        f"Pred: {pred_label} ({confidence:.1f}%)\n"
        f"True: {true_label} [{error_type}]",
        fontsize=10,
        color="crimson",
        fontweight="bold",
    )

  # Turn off unused subplot axes
  for idx in range(n_samples, len(axes)):
    axes[idx].axis("off")

  plt.tight_layout()
  if save_path:
    plt.savefig(save_path, dpi=300)
    print(f"Saved misclassified grid to {save_path}")



if __name__ == "__main__":
    print("---------CNN--------------")
    (train_losses, train_accuracies, val_losses, 
            val_accuracies, test_loss, test_accuracy, 
            val_epochs, all_labels, all_predictions, 
            all_probabilities, misclassified_samples) = pipeline(model_name="cnn")

    plot_training_and_test_curves(
        train_losses, train_accuracies, val_losses, val_accuracies, val_epochs, 
        test_loss=test_loss, test_accuracy=test_accuracy, save_path="cnn_training_test_curves.png")

    plot_confusion_matrix(
      y_true=all_labels,
      y_pred=all_predictions,
      class_names=["NC", "AD"],
      save_path="cnn_test_cm.png",
    )

    plot_roc_curve(
      y_true=all_labels,
      y_scores=[p[1] for p in all_probabilities],
      pos_label=1,
      save_path="cnn_test_roc.png",
    )

    plot_misclassified_grid(
        misclassified_samples=misclassified_samples,
        class_names=["NC", "AD"],
        max_samples=8,
        ncols=4,
        save_path="cnn_misclassified_slices.png",
    )

    print("---------CONVNEXT--------------")
    (train_losses, train_accuracies, val_losses, 
            val_accuracies, test_loss, test_accuracy, 
            val_epochs, all_labels, all_predictions, 
            all_probabilities, misclassified_samples) = pipeline(model_name="convnext")

    plot_training_and_test_curves(
        train_losses, train_accuracies, val_losses, val_accuracies, val_epochs, 
        test_loss=test_loss, test_accuracy=test_accuracy, save_path="convnext_training_test_curves.png")

    plot_confusion_matrix(
      y_true=all_labels,
      y_pred=all_predictions,
      class_names=["NC", "AD"],
      save_path="convnext_test_cm.png",
    )

    plot_roc_curve(
      y_true=all_labels,
      y_scores=[p[1] for p in all_probabilities],
      pos_label=1,
      save_path="convnext_test_roc.png",
    )

    plot_misclassified_grid(
        misclassified_samples=misclassified_samples,
        class_names=["NC", "AD"],
        max_samples=8,
        ncols=4,
        save_path="convnext_misclassified_slices.png",
    )
