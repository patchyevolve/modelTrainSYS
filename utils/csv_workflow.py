"""
CSV workflow helpers for quick model training and prediction.

Usage examples:
  python utils/csv_workflow.py train --data randomDATA/sample.csv --epochs 10
  python utils/csv_workflow.py predict --model trained_models/name.pt --data randomDATA/sample.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root imports work when running this file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.unified_trainer import TrainConfig, ModelType, UnifiedTrainer
from ui.training_controller import TrainingController
from utils.inference import load_checkpoint, run_inference, print_report, save_results


def train_csv(args) -> int:
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV file not found: {data_path}")

    cfg = TrainConfig(
        model_type=ModelType.CLASSIFIER,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
    )
    trainer = UnifiedTrainer(cfg, [str(data_path)], log_callback=lambda m, l="info": print(f"[{l}] {m}"))
    result = trainer.run()
    if not result:
        raise RuntimeError("Training returned no result")

    tc = TrainingController()
    ui_cfg = {
        "model_type": "Hierarchical Mamba",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "reflector": False,
    }
    name = tc.save_checkpoint(
        ui_cfg,
        result.model,
        result.best_state,
        result.info,
        {"loss": result.metrics["train_loss"][-1] if result.metrics["train_loss"] else "—", "acc": "—"},
    )
    print(f"Saved model: trained_models/{name}.pt")
    return 0


def predict_csv(args) -> int:
    model_path = Path(args.model)
    data_path = Path(args.data)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"CSV file not found: {data_path}")

    model, config, data_info = load_checkpoint(str(model_path))
    results = run_inference(model, data_info, str(data_path), threshold=args.threshold)
    try:
        print_report(results, model_path.stem, str(data_path))
    except UnicodeEncodeError:
        # Fallback for Windows terminals with cp1252 output.
        print("INFERENCE REPORT")
        print(f"model={model_path.stem} data={data_path.name}")
        print(
            f"accuracy={results['accuracy']:.4f} "
            f"precision={results['precision']:.4f} "
            f"recall={results['recall']:.4f} "
            f"f1={results['f1_score']:.4f}"
        )
    if args.save:
        out = save_results(results, model_path.stem, str(data_path))
        print(f"Saved inference results: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CSV workflow (train/predict)")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train classifier from CSV")
    tr.add_argument("--data", required=True, help="Path to CSV file")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--hidden-dim", type=int, default=256)
    tr.add_argument("--num-layers", type=int, default=4)
    tr.add_argument("--num-heads", type=int, default=8)
    tr.add_argument("--optimizer", default="Adam", choices=["Adam", "AdamW", "SGD"])
    tr.add_argument("--scheduler", default="None", choices=["None", "CosineAnnealing", "StepLR"])

    pr = sub.add_parser("predict", help="Predict from CSV using trained model")
    pr.add_argument("--model", required=True, help="Path to .pt checkpoint")
    pr.add_argument("--data", required=True, help="Path to CSV file")
    pr.add_argument("--threshold", type=float, default=0.5)
    pr.add_argument("--save", action="store_true")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        return train_csv(args)
    if args.cmd == "predict":
        return predict_csv(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
