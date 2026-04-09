"""
Inference engine for trained models.
Loads a .pt checkpoint, rebuilds the exact model, runs predictions,
and prints a full classification report.

Usage:
  python inference.py                                    # auto-finds latest model
  python inference.py --model trained_models/name.pt    # specific model
  python inference.py --data randomDATA/file.csv        # specific data file
  python inference.py --model name.pt --threshold 0.4   # custom threshold
"""

import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime


# ─── Rebuild model from state dict keys ──────────────────────────────────────

def _rebuild_cybersecurity(feat_dim: int, hidden: int) -> nn.Module:
    h2 = hidden * 2
    return nn.Sequential(
        nn.Linear(feat_dim, h2),   nn.BatchNorm1d(h2),     nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(h2, h2),         nn.BatchNorm1d(h2),     nn.ReLU(),
        nn.Linear(h2, hidden),     nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden, 1)
    )


def _rebuild_hierarchical(
    feat_dim: int,
    hidden: int,
    n_layers: int,
    num_classes: int = 1,
    num_heads: int = 0,
) -> nn.Module:
    """
    Rebuild the FULL HMTClassifier (input_proj + backbone + head).
    This matches the architecture that was actually saved during training.
    """
    from core.implementations import HMTClassifier
    
    if num_heads <= 0:
        num_heads = max(1, min(8, hidden // 64))
    hidden = (hidden // num_heads) * num_heads
    
    model = HMTClassifier(
        input_dim=feat_dim,
        num_classes=num_classes,
        dim=hidden,
        num_layers=n_layers,
        num_heads=num_heads,
        num_scales=3,
        dropout=0.1,
    )
    return model


def _rebuild_from_state_dict(state_dict: dict, config: dict, data_info: dict) -> nn.Module:
    """
    Automatically detect model type from state_dict keys and rebuild correctly.
    This is more robust than relying on config.get("model_type").
    """
    keys = list(state_dict.keys())
    
    # Detect HMTClassifier (has input_proj + backbone)
    if "input_proj.weight" in keys and "backbone.scale_weights" in keys:
        feat_dim = data_info.get("feature_dim", 16)
        hidden = config.get("hidden_dim", 128)
        n_layers = config.get("num_layers", 3)
        num_heads = config.get("num_heads", 0)
        
        # Extract num_classes from the saved head layer
        if "head.proj.weight" in state_dict:
            saved_classes = state_dict["head.proj.weight"].shape[0]
        elif "head.weight" in state_dict:
            saved_classes = state_dict["head.weight"].shape[0]
        else:
            saved_classes = data_info.get("num_classes", 1)
        
        return _rebuild_hierarchical(
            feat_dim, hidden, n_layers, saved_classes, num_heads=num_heads
        )
    
    # Detect HMTImageClassifier (has patch_embed + backbone + head)
    if "patch_embed.proj.weight" in keys and "backbone.mamba_blocks" in keys:
        from core.implementations import HMTImageClassifier
        arch = config.get("model_arch", {})
        dim = arch.get("dim", 128)
        num_heads = max(1, min(8, dim // 64))
        dim = (dim // num_heads) * num_heads
        model = HMTImageClassifier(
            num_classes=arch.get("num_classes", 2),
            dim=dim,
            patch_size=arch.get("patch_size", 16),
            num_layers=arch.get("num_layers", 2),
            num_heads=num_heads,
            num_scales=3,
        )
        return model
    
    # Detect simple MLP (sequential layers with BatchNorm)
    if "0.weight" in keys and "1.running_mean" in keys:
        feat_dim = data_info.get("feature_dim", 16)
        hidden = config.get("hidden_dim", 128)
        n_layers = config.get("num_layers", 3)
        return _rebuild_cybersecurity(feat_dim, hidden)
    
    # Fallback: try HMTClassifier
    feat_dim = data_info.get("feature_dim", 16)
    hidden = config.get("hidden_dim", 128)
    n_layers = config.get("num_layers", 3)
    num_classes = data_info.get("num_classes", 1)
    return _rebuild_hierarchical(
        feat_dim, hidden, n_layers, num_classes, num_heads=config.get("num_heads", 0)
    )


def rebuild_model(config: dict, data_info: dict) -> nn.Module:
    """Rebuild model based on config and data_info. Uses auto-detection from state_dict."""
    # This will be called after we load the checkpoint and have access to state_dict
    # For now, return a placeholder - actual rebuild happens in load_checkpoint
    return None


def rebuild_from_state_dict(state_dict: dict, config: dict, data_info: dict) -> nn.Module:
    """Rebuild model by inspecting the actual state_dict keys."""
    return _rebuild_from_state_dict(state_dict, config, data_info)


# ─── Load checkpoint ──────────────────────────────────────────────────────────

def load_checkpoint(pt_path: str):
    ckpt      = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Image model checkpoint
    if ckpt.get("model_arch", {}).get("type") == "HMTImageClassifier":
        from core.implementations import HMTImageClassifier
        arch       = ckpt["model_arch"]
        dim        = arch.get("dim", 128)
        num_heads  = max(1, min(8, dim // 64))
        dim        = (dim // num_heads) * num_heads
        model = HMTImageClassifier(
            num_classes  = arch["num_classes"],
            dim          = dim,
            patch_size   = arch.get("patch_size", 16),
            num_layers   = arch.get("num_layers", 2),
            num_heads    = num_heads,
            num_scales   = 3,
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        config    = ckpt.get("config", {})
        data_info = ckpt.get("data_info", {})
        data_info["class_names"] = ckpt.get("class_names", [])
        data_info["img_size"]    = arch.get("img_size", 64)
        data_info["task"]        = "image_classification"
        return model, config, data_info

    config    = ckpt["config"]
    data_info = ckpt.get("data_info", {})
    state_dict = ckpt["model_state_dict"]
    
    # Auto-detect model type from state_dict keys and rebuild correctly
    model = rebuild_from_state_dict(state_dict, config, data_info)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config, data_info


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(model: nn.Module, data_info: dict,
                  data_path: str, threshold: float = 0.5,
                  batch_size: int = 256) -> dict:
    """
    Run inference on a CSV/NPY file.
    Returns dict with predictions, probabilities, and metrics.
    """
    from data.data_loader import CSVDataset, NumpyDataset
    from torch.utils.data import DataLoader

    ext = Path(data_path).suffix.lower()
    if ext == ".csv":
        ds = CSVDataset(data_path)
    elif ext in (".npy", ".npz"):
        ds = NumpyDataset(data_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_probs  = []
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            # Pad/trim features to match trained model's expected dim
            expected = data_info.get("feature_dim", xb.shape[1])
            if xb.shape[1] < expected:
                pad = torch.zeros(xb.shape[0], expected - xb.shape[1])
                xb  = torch.cat([xb, pad], dim=1)
            elif xb.shape[1] > expected:
                xb = xb[:, :expected]

            logits = model(xb)
            # Determine binary vs multiclass from model output shape first.
            is_binary = (logits.dim() == 2 and logits.shape[1] == 1)
            if logits.dim() == 1:
                is_binary = True
            if is_binary:
                probs = torch.sigmoid(logits.view(-1))
                preds = (probs >= threshold).long()
            else:
                probs = torch.softmax(logits, dim=-1).max(dim=1).values
                preds = logits.argmax(dim=1)

            all_probs.append(probs.numpy())
            all_preds.append(preds.numpy())
            all_labels.append(yb.squeeze(1).long().numpy())

    probs_arr  = np.concatenate(all_probs)
    preds_arr  = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)

    # Metrics
    tp = int(((preds_arr == 1) & (labels_arr == 1)).sum())
    tn = int(((preds_arr == 0) & (labels_arr == 0)).sum())
    fp = int(((preds_arr == 1) & (labels_arr == 0)).sum())
    fn = int(((preds_arr == 0) & (labels_arr == 1)).sum())

    accuracy  = (tp + tn) / max(len(labels_arr), 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    fpr       = fp / max(fp + tn, 1)

    return {
        "total_samples": len(labels_arr),
        "threshold":     threshold,
        "accuracy":      accuracy,
        "precision":     precision,
        "recall":        recall,
        "f1_score":      f1,
        "false_positive_rate": fpr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "attack_detected":  int((preds_arr == 1).sum()),
        "benign_detected":  int((preds_arr == 0).sum()),
        "avg_confidence":   float(probs_arr.mean()),
        "high_confidence":  int((probs_arr > 0.8).sum()),
        "low_confidence":   int((probs_arr < 0.3).sum()),
        "probabilities":    probs_arr,
        "predictions":      preds_arr,
        "labels":           labels_arr,
    }


# ─── Pretty print ─────────────────────────────────────────────────────────────

def print_report(results: dict, model_name: str, data_path: str):
    w = 60
    print()
    print("═" * w)
    print(f"  INFERENCE REPORT")
    print(f"  Model : {model_name}")
    print(f"  Data  : {Path(data_path).name}")
    print(f"  Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * w)

    print(f"\n  DATASET")
    print(f"  {'Total samples':<28} {results['total_samples']:>8,}")
    print(f"  {'Threshold':<28} {results['threshold']:>8.2f}")

    print(f"\n  PREDICTIONS")
    print(f"  {'Attacks detected':<28} {results['attack_detected']:>8,}")
    print(f"  {'Benign detected':<28} {results['benign_detected']:>8,}")
    print(f"  {'Avg confidence':<28} {results['avg_confidence']:>8.3f}")
    print(f"  {'High confidence (>0.8)':<28} {results['high_confidence']:>8,}")
    print(f"  {'Low confidence (<0.3)':<28} {results['low_confidence']:>8,}")

    print(f"\n  CONFUSION MATRIX")
    print(f"  {'':>16}  Predicted 0  Predicted 1")
    print(f"  {'Actual 0 (Benign)':<16}  {results['tn']:>11,}  {results['fp']:>11,}")
    print(f"  {'Actual 1 (Attack)':<16}  {results['fn']:>11,}  {results['tp']:>11,}")

    print(f"\n  METRICS")
    print(f"  {'Accuracy':<28} {results['accuracy']*100:>7.2f}%")
    print(f"  {'Precision':<28} {results['precision']*100:>7.2f}%")
    print(f"  {'Recall (Detection Rate)':<28} {results['recall']*100:>7.2f}%")
    print(f"  {'F1 Score':<28} {results['f1_score']:>8.4f}")
    print(f"  {'False Positive Rate':<28} {results['false_positive_rate']*100:>7.2f}%")

    # Grade
    f1 = results["f1_score"]
    grade = "EXCELLENT" if f1 > 0.90 else \
            "GOOD"      if f1 > 0.80 else \
            "FAIR"      if f1 > 0.65 else "NEEDS IMPROVEMENT"
    print(f"\n  OVERALL GRADE: {grade}")
    print("═" * w)
    print()


# ─── Save results ─────────────────────────────────────────────────────────────

def save_results(results: dict, model_name: str, data_path: str) -> str:
    out_dir = Path("inference_results")
    out_dir.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{model_name}_{Path(data_path).stem}_{ts}"

    # Save summary JSON (no arrays)
    summary = {k: v for k, v in results.items()
               if not isinstance(v, np.ndarray)}
    summary["model"]     = model_name
    summary["data_file"] = str(data_path)
    summary["timestamp"] = datetime.now().isoformat()

    json_path = out_dir / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Save predictions CSV
    import csv
    csv_path = out_dir / f"{name}_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "true_label", "predicted",
                         "probability", "correct"])
        for i, (lbl, pred, prob) in enumerate(
                zip(results["labels"], results["predictions"],
                    results["probabilities"])):
            writer.writerow([i, int(lbl), int(pred),
                             f"{prob:.4f}", int(lbl == pred)])

    return str(json_path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def find_latest_model() -> str:
    models = sorted(
        Path("trained_models").glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not models:
        raise FileNotFoundError("No .pt files found in trained_models/")
    return str(models[0])


def find_default_data() -> str:
    for p in Path("randomDATA").glob("*.csv"):
        return str(p)
    raise FileNotFoundError("No CSV files found in randomDATA/")


def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained model")
    parser.add_argument("--model",     default=None,
                        help="Path to .pt file (default: latest in trained_models/)")
    parser.add_argument("--data",      default=None,
                        help="Path to CSV/NPY file (default: first in randomDATA/)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    parser.add_argument("--save",      action="store_true",
                        help="Save results to inference_results/")
    args = parser.parse_args()

    # Resolve paths
    pt_path   = args.model or find_latest_model()
    data_path = args.data  or find_default_data()

    print(f"\nLoading model : {pt_path}")
    print(f"Data file     : {data_path}")
    print(f"Threshold     : {args.threshold}")

    # Load
    model, config, data_info = load_checkpoint(pt_path)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model type    : {config['model_type']}")
    print(f"Parameters    : {param_count:,}")
    print(f"Feature dim   : {data_info.get('feature_dim', '?')}")
    print(f"Trained on    : {data_info.get('train_rows', '?'):,} rows")

    # Run
    print("\nRunning inference…")
    results = run_inference(model, data_info, data_path,
                            threshold=args.threshold)

    # Report
    model_name = Path(pt_path).stem
    print_report(results, model_name, data_path)

    # Save
    if args.save:
        out = save_results(results, model_name, data_path)
        print(f"Results saved → {out}\n")

    return results


if __name__ == "__main__":
    main()
