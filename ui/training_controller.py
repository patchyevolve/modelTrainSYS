"""
Training Controller
Handles the logic for model configurations, training lifecycle, and checkpointing.
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from training.unified_trainer import UnifiedTrainer, TrainConfig, ModelType

# ─── Model type registry — what each option does ─────────────────────────────
MODEL_REGISTRY = {
    "Hierarchical Mamba": {
        "desc": (
            "Multi-scale Mamba encoder → classifier head.\n"
            "Best for: tabular/CSV data, cybersecurity logs.\n"
            "Output: binary or multi-class label (a number, not text)."
        ),
        "input_dim_auto": True,
        "task": "binary_classification",
    },
    "Transformer Only": {
        "desc": (
            "Residual MLP with skip connections.\n"
            "Best for: clean tabular data, fast baseline.\n"
            "Output: binary or multi-class label."
        ),
        "input_dim_auto": True,
        "task": "binary_classification",
    },
    "Mamba+Transformer": {
        "desc": (
            "Hierarchical Mamba with deeper fusion layers.\n"
            "Best for: large tabular datasets (10k+ rows).\n"
            "Output: binary or multi-class label."
        ),
        "input_dim_auto": True,
        "task": "binary_classification",
    },
    "Cybersecurity": {
        "desc": (
            "Adversarial trainer for attack detection.\n"
            "Trains on: SQL injection, XSS, DDoS, malware, zero-day.\n"
            "Best for: cybersecurity_intrusion_data.csv\n"
            "Output: attack probability (0.0–1.0)."
        ),
        "input_dim_auto": True,
        "task": "binary_classification",
    },
    "Image Classification": {
        "desc": (
            "HMT Vision Transformer — patch-based image classifier.\n"
            "Train on: folders of images (one subfolder per class).\n"
            "  root/cats/img1.jpg  root/dogs/img2.jpg  ...\n"
            "Or flat folder (single class). Supports JPG/PNG/BMP/TIFF.\n"
            "Output: class label + confidence."
        ),
        "input_dim_auto": False,
        "task": "image_classification",
    },
    "Text Generation": {
        "desc": (
            "Hierarchical Mamba Language Model — learns to generate text.\n"
            "Train on: .txt, .csv, .json, .jsonl files OR HuggingFace datasets.\n"
            "Enter dataset name (e.g., wikitext, ianncity/General-Distillation-Prompts-1M)\n"
            "to train on online datasets. Enable reasoning for logical tasks.\n"
            "After training: type a prompt → model continues writing."
        ),
        "input_dim_auto": False,
        "task": "language_model",
    },
}

class TrainingController:
    @staticmethod
    def get_model_type_enum(ui_model_type: str) -> ModelType:
        task = MODEL_REGISTRY.get(ui_model_type, {}).get("task", "binary_classification")
        if task == "language_model":
            return ModelType.LANGUAGE_MODEL
        elif task == "image_classification":
            return ModelType.IMAGE_CLASSIFIER
        return ModelType.CLASSIFIER

    @staticmethod
    def save_checkpoint(cfg, model, best_state, data_info, final_stats):
        """Logic to save weights and metadata to disk."""
        save_dir = Path("trained_models")
        save_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{cfg['model_type'].replace(' ', '_')}_{ts}"

        # Save weights
        weights_path = save_dir / f"{name}.pt"
        state = best_state if best_state else model.state_dict()
        torch.save({
            "model_state_dict": state,
            "config":           cfg,
            "data_info":        data_info or {},
        }, weights_path)

        # Save metadata
        meta = {
            "name":         name,
            "model_type":   cfg["model_type"],
            "epochs":       cfg["epochs"],
            "loss":         final_stats.get("loss", "—"),
            "accuracy":     final_stats.get("acc", "—"),
            "reflector":    cfg.get("reflector", False),
            "created":      datetime.now().isoformat(),
            "status":       "ready",
            "config":       cfg,
            "weights_file": str(weights_path),
            "feature_dim":  (data_info or {}).get("feature_dim"),
            "num_classes":  (data_info or {}).get("num_classes"),
            "train_rows":   (data_info or {}).get("train_rows"),
        }
        meta_path = save_dir / f"{name}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            
        return name

    @staticmethod
    def validate_runtime_config(cfg: dict) -> tuple[bool, str]:
        """
        Production guardrails for known unstable runtime combinations.
        Returns (is_valid, message). Empty message means OK.
        """
        model_type = cfg.get("model_type", "")
        task = MODEL_REGISTRY.get(model_type, {}).get("task", "binary_classification")
        batch_size = int(cfg.get("batch_size", 32))
        reasoning_weight = float(cfg.get("reasoning_weight", 1.0))

        # LM on CPU with large batches appears "stuck" for long intervals.
        if task == "language_model" and not torch.cuda.is_available():
            if batch_size > 16:
                return (
                    False,
                    "Text Generation on CPU is too slow with Batch Size > 16.\n"
                    "Set Batch Size to 16 or lower, then start training again.",
                )
            if reasoning_weight > 1.5:
                return (
                    False,
                    "Text Generation on CPU with high Reasoning Weight is unstable/very slow.\n"
                    "Set Reasoning Weight to 1.0–1.5 for production-safe CPU training.",
                )

        return True, ""