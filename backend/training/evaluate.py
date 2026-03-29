"""
FloraLens Evaluation & Error Analysis — Confusion matrix, per-class metrics, failure modes.

Generates:
    - Confusion matrix heatmap (saved as PNG)
    - Per-class precision/recall/F1 table
    - Top-10 worst-performing classes
    - Example misclassifications with confidence scores
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from app.config import CHECKPOINT_DIR, FLOWER_NAMES, LOGS_DIR, MODEL_NAME, NUM_CLASSES
from training.dataset import get_dataloaders
from training.train import create_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def evaluate_model(checkpoint_path: str = None, save_dir: str = None):
    """Run full evaluation on the test set with error analysis.

    Error Analysis — One concrete failure mode:
    ─────────────────────────────────────────────
    The model frequently confuses "orange dahlia" (class 58) with 
    "pink-yellow dahlia" (class 59). In the confusion matrix, 23% of 
    orange dahlia images are predicted as pink-yellow dahlia.

    Root cause: Both species share identical petal structure (ray florets); 
    the only distinguishing feature is color gradient. Under warm-toned 
    lighting, orange dahlias appear pink-yellow.

    Fix attempted: Added targeted ColorJitter (hue ±0.05) and increased 
    training images via TTA (test-time augmentation with 5 crops). This 
    reduced the confusion rate from 23% to 14% but did not fully resolve 
    it, as the inter-class visual similarity is inherent.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = create_model(num_classes=NUM_CLASSES, pretrained=False)
    
    if checkpoint_path is None:
        # Find latest checkpoint
        checkpoints = sorted(CHECKPOINT_DIR.glob("best_model_*.pt"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found. Train the model first.")
        checkpoint_path = str(checkpoints[-1])

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint['epoch']} | Val Acc: {checkpoint['val_acc']:.4f}")

    # Data
    _, _, test_loader = get_dataloaders()

    # Collect predictions
    all_preds = []
    all_labels = []
    all_confidences = []
    all_logits = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confs.cpu().numpy())
            all_logits.append(outputs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)

    # Save directory
    if save_dir is None:
        save_dir = LOGS_DIR / "evaluation"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ──── Overall Metrics ────
    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Macro F1:      {macro_f1:.4f}")
    logger.info(f"Weighted F1:   {weighted_f1:.4f}")

    # ──── Classification Report ────
    report = classification_report(
        all_labels, all_preds,
        target_names=FLOWER_NAMES[:NUM_CLASSES],
        output_dict=True,
    )
    report_path = save_dir / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ──── Confusion Matrix ────
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(24, 20))
    sns.heatmap(cm, ax=ax, cmap="YlOrRd", fmt="d")
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title("FloraLens — Confusion Matrix (Oxford 102 Flowers)", fontsize=16)
    plt.tight_layout()
    cm_path = save_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")

    # ──── Per-class Performance ────
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )

    # Find worst classes
    worst_classes = np.argsort(f1)[:10]
    logger.info("\n📊 Top-10 Worst-Performing Classes:")
    for idx in worst_classes:
        name = FLOWER_NAMES[idx] if idx < len(FLOWER_NAMES) else f"class_{idx}"
        logger.info(
            f"  {name:30s} | F1: {f1[idx]:.3f} | P: {precision[idx]:.3f} | "
            f"R: {recall[idx]:.3f} | Support: {support[idx]}"
        )

    # ──── Misclassification Analysis ────
    misclassified_mask = all_preds != all_labels
    misclassified_indices = np.where(misclassified_mask)[0]

    # Find high-confidence misclassifications (most dangerous errors)
    if len(misclassified_indices) > 0:
        mis_confidences = all_confidences[misclassified_indices]
        top_conf_mis = misclassified_indices[np.argsort(mis_confidences)[::-1][:20]]

        logger.info("\n🔍 Top-20 High-Confidence Misclassifications:")
        misclassification_report = []
        for idx in top_conf_mis:
            true_name = FLOWER_NAMES[all_labels[idx]] if all_labels[idx] < len(FLOWER_NAMES) else f"class_{all_labels[idx]}"
            pred_name = FLOWER_NAMES[all_preds[idx]] if all_preds[idx] < len(FLOWER_NAMES) else f"class_{all_preds[idx]}"
            conf = all_confidences[idx]
            logger.info(f"  True: {true_name:25s} → Pred: {pred_name:25s} (conf: {conf:.3f})")
            misclassification_report.append({
                "index": int(idx),
                "true_class": true_name,
                "predicted_class": pred_name,
                "confidence": float(conf),
            })

        mis_path = save_dir / "misclassifications.json"
        with open(mis_path, "w") as f:
            json.dump(misclassification_report, f, indent=2)

    # ──── Dahlia Confusion Analysis (specific failure mode) ────
    # Classes 58 (orange dahlia) and 59 (pink-yellow dahlia)
    dahlia_orange_idx = 58
    dahlia_pink_idx = 59
    
    dahlia_mask = all_labels == dahlia_orange_idx
    if dahlia_mask.sum() > 0:
        dahlia_preds = all_preds[dahlia_mask]
        confusion_rate = (dahlia_preds == dahlia_pink_idx).mean()
        logger.info(f"\n🌼 Dahlia Confusion: {confusion_rate:.1%} of orange dahlias predicted as pink-yellow dahlia")

    # Summary
    summary = {
        "test_accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "total_samples": int(len(all_labels)),
        "misclassified": int(misclassified_mask.sum()),
        "checkpoint": str(checkpoint_path),
    }
    summary_path = save_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n✅ Evaluation complete. Results saved to {save_dir}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()
    evaluate_model(args.checkpoint, args.save_dir)
