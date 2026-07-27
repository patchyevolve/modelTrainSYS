import json
import torch
from pathlib import Path
from datetime import datetime
from training.unified_trainer import UnifiedTrainer, TrainConfig
from core.text_model import save_lm


class TrainingController:
    @staticmethod
    def save_checkpoint(cfg, model, best_state, data_info, final_stats, tokenizer=None,
                        files_used=None, resume_count=0):
        save_dir = Path("trained_models")
        save_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"trained_{ts}"

        weights_path = save_dir / f"{name}.pt"
        state = best_state if best_state else model.state_dict()
        save_lm(model, tokenizer,
                config={**cfg, **(data_info or {}),
                        "_files_trained_on": files_used or [],
                        "_resume_count": resume_count,
                        "_version": 2},
                path=str(weights_path),
                files_trained_on=files_used or [],
                resume_count=resume_count)

        meta = {
            "name":         name,
            "model_type":   "text_generation",
            "epochs":       cfg["epochs"],
            "loss":         final_stats.get("loss", "—"),
            "created":      datetime.now().isoformat(),
            "status":       "ready",
            "config":       cfg,
            "weights_file": str(weights_path),
            "files_trained_on": files_used or [],
            "_resume_count": resume_count,
        }
        meta_path = save_dir / f"{name}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return name

    @staticmethod
    def validate_runtime_config(cfg: dict) -> tuple[bool, str]:
        batch_size = int(cfg.get("batch_size", 32))
        if not torch.cuda.is_available() and batch_size > 16:
            return False, "CPU is too slow with Batch Size > 16.\nSet to 16 or lower."
        return True, ""
