"""
Production smoke checks for critical training paths.

Run:
  python production_smoke.py
"""

from pathlib import Path
import tempfile

from training.unified_trainer import UnifiedTrainer, TrainConfig, ModelType


def _run_lm_smoke() -> None:
    text = ("reasoning because therefore step by step.\n" * 200).strip()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tiny.txt"
        p.write_text(text, encoding="utf-8")

        cfg = TrainConfig(
            model_type=ModelType.LANGUAGE_MODEL,
            epochs=1,
            batch_size=4,
            lr=1e-3,
            hidden_dim=64,
            num_layers=1,
            num_heads=2,
            seq_len=32,
            reasoning_weight=1.0,
            curriculum=False,
        )
        trainer = UnifiedTrainer(cfg, [str(p)])
        result = trainer.run()
        assert result is not None, "LM smoke run returned no result"
        assert result.model is not None, "LM smoke run missing model"


def _run_tabular_smoke() -> None:
    csv = "f1,f2,f3,label\n" + "\n".join(
        f"{i%3},{(i*2)%5},{(i*3)%7},{i%2}" for i in range(200)
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tiny.csv"
        p.write_text(csv, encoding="utf-8")

        cfg = TrainConfig(
            model_type=ModelType.CLASSIFIER,
            epochs=1,
            batch_size=16,
            lr=1e-3,
            hidden_dim=64,
            num_layers=1,
            num_heads=2,
        )
        trainer = UnifiedTrainer(cfg, [str(p)])
        result = trainer.run()
        assert result is not None, "Tabular smoke run returned no result"
        assert result.model is not None, "Tabular smoke run missing model"


if __name__ == "__main__":
    _run_lm_smoke()
    _run_tabular_smoke()
    print("production_smoke: PASS")
