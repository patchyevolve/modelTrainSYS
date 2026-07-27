"""
End-to-end pipeline: pre-train foundation → fine-tune reasoning → final model.
Run once, get a reasoning-capable model.

Usage:
    python scripts/pipeline.py                           # full pipeline, defaults
    python scripts/pipeline.py --dim 512 --layers 8      # bigger model
    python scripts/pipeline.py --data-chars 10000000     # more data
    python scripts/pipeline.py --device cuda             # GPU
"""

import argparse, sys, os, json, time, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["DML_TRAINING"] = "1"

import torch

FOUNDATION_DIR = Path(__file__).resolve().parent.parent / "foundation"
FINETUNE_DIR = FOUNDATION_DIR / "finetuned"


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: foundation → fine-tune → final model")
    # Foundation params
    parser.add_argument("--data-chars", type=int, default=80000000, help="Foundation corpus size")
    parser.add_argument("--epochs", type=int, default=30, help="Foundation epochs")
    parser.add_argument("--dim", type=int, default=384, help="Model dimension")
    parser.add_argument("--layers", type=int, default=6, help="Transformer layers")
    parser.add_argument("--heads", type=int, default=6, help="Attention heads")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=128, help="Foundation batch size")
    parser.add_argument("--foundation-lr", type=float, default=0.001, help="Foundation learning rate")
    # Fine-tune params
    parser.add_argument("--ft-epochs", type=int, default=8, help="Fine-tuning epochs")
    parser.add_argument("--ft-batch-size", type=int, default=64, help="Fine-tuning batch size")
    parser.add_argument("--ft-lr", type=float, default=0.0003, help="Fine-tuning learning rate")
    parser.add_argument("--generate", type=int, default=8000, help="Q&A examples to generate")
    parser.add_argument("--template", default="step_by_step", help="Template name")
    # Data
    parser.add_argument("--data-dir", default="",
                        help="Directory with code/text data (downloads Gutenberg if empty)")
    # Other
    parser.add_argument("--device", default="cpu", help="Device (auto-detects cuda)")
    parser.add_argument("--resume", action="store_true", help="Resume foundation from checkpoint")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else args.device
    print(f"Device: {device.upper()}")

    # Phase 1: Foundation training
    print("\n" + "=" * 60)
    print("  PHASE 1: FOUNDATION PRE-TRAINING")
    print("=" * 60)

    from scripts.pretrain_foundation import train_foundation
    train_foundation(
        data_paths=[],
        data_dir=args.data_dir,
        max_chars=args.data_chars,
        epochs=args.epochs,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        prompt_ratio=0.0,
        resume=args.resume,
        save_every=5,
        lr=args.foundation_lr,
        device=device,
        use_amp=not args.no_amp,
    )

    # Phase 2: Fine-tune on reasoning data
    print("\n" + "=" * 60)
    print("  PHASE 2: REASONING FINE-TUNING")
    print("=" * 60)

    from scripts.auto_finetune import auto_finetune
    auto_finetune(
        data_dir=None,
        epochs=args.ft_epochs,
        dim=args.dim,
        layers=args.layers,
        heads=args.heads,
        seq_len=args.seq_len,
        batch_size=args.ft_batch_size,
        template_name=args.template,
        generate_count=args.generate,
        lr=args.ft_lr,
        device=device,
        use_amp=not args.no_amp,
    )

    # Phase 3: Verify
    print("\n" + "=" * 60)
    print("  PHASE 3: VERIFICATION")
    print("=" * 60)

    from core.text_model import load_lm
    from core.implementations import GenConfig

    model, tokenizer = load_lm(str(FINETUNE_DIR / "finetuned.pt"), device=device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Vocab: {tokenizer.vocab_size_real}")

    gcfg = GenConfig(
        temperature=0.1, max_new_tokens=80,
        repetition_penalty=1.3, top_k=30, top_p=0.9,
    )

    tests = [
        "Query: What is 15 + 27?\nReasoning:",
        "Query: A rectangle has length 8 and width 6. What is its area?\nReasoning:",
        "Query: What is the capital of Japan?\nReasoning:",
        "Query: What planet is known as the Red Planet?\nReasoning:",
    ]
    for prompt in tests:
        ids = tokenizer.encode(prompt, add_special=False)
        out_ids = model.generate(ids, gcfg, device=device)
        text = tokenizer.decode(out_ids)
        print(f"  [{prompt[:40]}...]")
        print(f"  → {text[:120]}")
        print()

    # Phase 4: Save final
    FINAL_DIR = FOUNDATION_DIR / "final"
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for f in ["finetuned.pt", "finetuned_tokenizer.json",
              "finetuned.tokenizer", "finetuned.json", "manifest.json"]:
        src = FINETUNE_DIR / f
        if src.exists():
            shutil.copy2(str(src), str(FINAL_DIR / f.replace("finetuned", "final")))

    print(f"  Final model saved to: {FINAL_DIR}")
    print(f"  Done! Pipeline complete.")


if __name__ == "__main__":
    main()
