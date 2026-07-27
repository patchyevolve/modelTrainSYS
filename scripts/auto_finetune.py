"""
Auto Fine-Tune — loads foundation model, trains on structured reasoning data,
then tests inference so you can verify the pipeline works end-to-end.

Run BEFORE starting the long foundation run to make sure everything works.

Usage:
    python scripts/auto_finetune.py                                    # use generated math Q&A
    python scripts/auto_finetune.py --data-dir ./my_qa_data           # use your own data
    python scripts/auto_finetune.py --epochs 5 --template step_by_step
"""

import argparse, json, sys, os, logging, re, time, random, math as _math
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["DML_TRAINING"] = "1"

import torch
import torch.nn.functional as F

from core.implementations import HMTLanguageModel, GenConfig
from core.text_model import load_lm, save_lm
from data.advanced_tokenizer import train_tokenizer, AdvancedTokenizer
from data.text_dataset import build_text_loaders
from data.templates import init_templates, get_template
from data.template_pipeline import parse_examples, render_examples_to_windows, StructuredExample
try:
    from torch.amp import autocast, GradScaler
    _new_amp = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _new_amp = False

logging.disable(logging.CRITICAL)
logging.getLogger().disabled = True

FOUNDATION_DIR = Path(__file__).resolve().parent.parent / "foundation"
FINETUNE_DIR = FOUNDATION_DIR / "finetuned"
FOUNDATION_MODEL = FOUNDATION_DIR / "base_model.pt"


# ── Structured Q&A generation (math + reasoning) ────────────────────────────

MATH_TOPICS = [
    # Linear equations
    ("Solve for x: {a}x + {b} = {c}",
     lambda a,b,c: f"Subtract {b} from both sides: {a}x = {c} - {b} = {c-b}. Then divide by {a}: x = {c-b}/{a} = {(c-b)/a:.0f}.",
     lambda a,b,c: f"x = {(c-b)/a:.0f}"),
    # Quadratic
    ("What is the square root of {n}?",
     lambda n: f"The square root of {n} is the number that when multiplied by itself gives {n}. {_math.isqrt(n)} × {_math.isqrt(n)} = {_math.isqrt(n)**2}.",
     lambda n: str(_math.isqrt(n))),
    # Area
    ("A rectangle has length {l} and width {w}. What is its area?",
     lambda l,w: f"Area of a rectangle = length × width = {l} × {w} = {l*w}.",
     lambda l,w: str(l*w)),
    # Perimeter
    ("A square has side length {s}. What is its perimeter?",
     lambda s: f"Perimeter of a square = 4 × side length = 4 × {s} = {4*s}.",
     lambda s: str(4*s)),
    # Average
    ("What is the average of {a} and {b}?",
     lambda a,b: f"Average = (sum of values) / (number of values) = ({a} + {b}) / 2 = {a+b} / 2 = {(a+b)/2:.0f}.",
     lambda a,b: f"{(a+b)/2:.0f}"),
    # Simple interest
    ("Simple interest on ${p} at {r}% for {t} year(s) is?",
     lambda p,r,t: f"Simple interest = P × R × T / 100 = ${p} × {r}% × {t} / 100 = ${p*r*t/100:.0f}.",
     lambda p,r,t: f"${p*r*t/100:.0f}"),
    # Distance
    ("A car travels at {s} mph for {h} hours. Distance traveled?",
     lambda s,h: f"Distance = speed × time = {s} × {h} = {s*h} miles.",
     lambda s,h: f"{s*h} miles"),
    # Percentage
    ("What is {p}% of {n}?",
     lambda p,n: f"{p}% of {n} = ({p}/100) × {n} = {p*n/100:.1f}.",
     lambda p,n: f"{p*n/100:.1f}"),
]

REASONING_TOPICS = [
    ("What is the capital of {country}?",
     lambda country: f"The capital of {country} is a well-known geographical fact. It is the political and administrative center.",
     {"France": "Paris", "Japan": "Tokyo", "Brazil": "Brasilia", "Egypt": "Cairo",
      "Australia": "Canberra", "Canada": "Ottawa", "India": "New Delhi", "Italy": "Rome"}),
    ("What planet is known as the {nickname}?",
     lambda nickname: f"The planet known as {nickname} is famous in astronomy. It has distinct features that earned it this nickname.",
     {"Red Planet": "Mars", "Morning Star": "Venus", "Ringed Planet": "Saturn",
      "Blue Planet": "Earth", "Giant Planet": "Jupiter"}),
    ("What is the chemical symbol for {element}?",
     lambda element: f"The chemical symbol for {element} comes from its Latin or English name. It is a standard notation in chemistry.",
     {"Hydrogen": "H", "Helium": "He", "Carbon": "C", "Nitrogen": "N",
      "Oxygen": "O", "Iron": "Fe", "Gold": "Au", "Silver": "Ag"}),
]


def generate_math_qa(count: int = 300) -> List[dict]:
    """Generate structured math Q&A as dicts with question/steps/answer."""
    rng = random.Random(42)
    examples = []
    def _has_keys(t, *keys):
        return all("{" + k + "}" in t for k in keys)

    # Build a dispatch table: (check_fn, fill_fn) where check_fn(title) → bool
    # and fill_fn(title, reason_fn, answer_fn) → dict
    def _fill(title_str, reason_fn, answer_fn, **kw):
        return {"question": title_str.format(**kw), "steps": reason_fn(**kw), "answer": answer_fn(**kw)}

    dispatchers = [
        (lambda ti: _has_keys(ti, "a", "b", "c"),
         lambda ti, rf, af: _fill(ti, rf, af, a=rng.randint(2, 9), b=rng.randint(-20, 20), c=rng.randint(-50, 50))),
        (lambda ti: _has_keys(ti, "p", "r", "t"),
         lambda ti, rf, af: _fill(ti, rf, af, p=rng.choice([100, 200, 500, 1000]), r=rng.choice([5, 8, 10, 12, 15]), t=rng.choice([1, 2, 3]))),
        (lambda ti: _has_keys(ti, "s", "h"),
         lambda ti, rf, af: _fill(ti, rf, af, s=rng.choice([30, 40, 50, 60, 70]), h=rng.choice([1, 2, 3, 4]))),
        (lambda ti: _has_keys(ti, "p", "n"),
         lambda ti, rf, af: _fill(ti, rf, af, p=rng.choice([10, 15, 20, 25, 50, 75]), n=rng.choice([50, 100, 200, 500, 1000]))),
        (lambda ti: _has_keys(ti, "l", "w"),
         lambda ti, rf, af: _fill(ti, rf, af, l=rng.randint(3, 20), w=rng.randint(2, 15))),
        (lambda ti: _has_keys(ti, "n"),
         lambda ti, rf, af: _fill(ti, rf, af, n=rng.choice([16, 25, 36, 49, 64, 81, 100, 121, 144]))),
        (lambda ti: _has_keys(ti, "s"),
         lambda ti, rf, af: _fill(ti, rf, af, s=rng.randint(3, 15))),
    ]

    for i in range(count):
        topic = rng.choice(MATH_TOPICS)
        title, reason_fn, answer_fn = topic
        for check, fill in dispatchers:
            if check(title):
                examples.append(fill(title, reason_fn, answer_fn))
                break
    return examples


def generate_knowledge_qa(count: int = 200) -> List[dict]:
    """Generate knowledge-based Q&A."""
    rng = random.Random(7)
    examples = []
    for i in range(count):
        topic = rng.choice(REASONING_TOPICS)
        title, reason_lambda, answer_dict = topic
        key = rng.choice(list(answer_dict.keys()))
        if "capital" in title:
            q = f"What is the capital of {key}?"
            ans = answer_dict[key]
        elif "planet" in title:
            q = f"What planet is known as the {key}?"
            ans = answer_dict[key]
        elif "chemical symbol" in title:
            q = f"What is the chemical symbol for {key}?"
            ans = answer_dict[key]
        else:
            continue
        examples.append({"question": q, "steps": f"The answer is a well-established fact about {key}.",
                        "answer": ans})
    return examples


def save_qa_data(examples: List[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(examples, f, indent=2)


# ── Inference test ───────────────────────────────────────────────────────────

def test_inference(model, tokenizer, prompt: str, label: str = "Model", device: str = "cpu"):
    gcfg = GenConfig(max_new_tokens=60, temperature=0.6, top_k=30, top_p=0.9,
                     repetition_penalty=2.0, repetition_penalty_range=100)
    eos_id = tokenizer._vocab.get("<EOS>", 3) if hasattr(tokenizer, "_vocab") else 3
    gcfg.eos_id = eos_id
    ids = tokenizer.encode(prompt, add_special=False)
    out = model.generate(ids, gcfg, device=device)
    text = tokenizer.decode(out).replace("\n", " ").strip()[:120]
    print(f"  [{label}] \"{prompt}\"")
    print(f"           → {text}")
    return text


# ── Main ─────────────────────────────────────────────────────────────────────

def auto_finetune(
    data_dir: Optional[str] = None,
    epochs: int = 6,
    dim: int = 256,
    layers: int = 4,
    heads: int = 4,
    seq_len: int = 128,
    batch_size: int = 16,
    template_name: str = "step_by_step",
    generate_count: int = 1000,
    lr: float = 0.0005,
    device: str = "cpu",
    use_amp: bool = True,
):
    sep = "=" * 60
    print(f"\n{sep}")
    print("  AUTO FINE-TUNE")
    print(f"{sep}")

    # ── Check foundation model ───────────────────────────────────────────────
    if not FOUNDATION_MODEL.exists():
        print(f"\n  Foundation model not found at {FOUNDATION_MODEL}")
        print("  Run `python scripts/pretrain_foundation.py` first.")
        return

    device = "cuda" if torch.cuda.is_available() else device
    print(f"  Device: {device.upper()}")
    if device.startswith("cuda"):
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"\n  Loading foundation model...")
    model, tokenizer = load_lm(str(FOUNDATION_MODEL), device=device)
    print(f"    vocab={tokenizer.vocab_size_real}, dim={dim}, layers={layers}")

    # ── Prepare fine-tuning data ─────────────────────────────────────────────
    data_file = FINETUNE_DIR / "finetune_data.json"

    if data_dir:
        # Load user-provided data
        p = Path(data_dir)
        files = sorted([str(f) for f in p.glob("*.json")] + [str(f) for f in p.glob("*.jsonl")])
        if not files:
            print(f"  No JSON/JSONL files in {data_dir}")
            return
        print(f"\n  Loading {len(files)} file(s) from {data_dir}")
        train_loader, val_loader, tokenizer, info = build_text_loaders(
            files, seq_len=seq_len, batch_size=batch_size,
            val_split=0.05, tokenizer=tokenizer,
            template_name=template_name,
        )
    else:
        # Generate Q&A data
        print(f"\n  Generating {generate_count} math + knowledge Q&A examples...")
        math_qa = generate_math_qa(generate_count)
        know_qa = generate_knowledge_qa(generate_count // 2)
        all_qa = math_qa + know_qa
        rng = random.Random(42)
        rng.shuffle(all_qa)
        print(f"    {len(math_qa)} math + {len(know_qa)} knowledge = {len(all_qa)} total")
        save_qa_data(all_qa, str(data_file))

        print(f"\n  Building template pipeline...")
        init_templates("templates")
        template = get_template(template_name)
        examples = parse_examples([str(data_file)], template, use_cache=False)
        print(f"    {len(examples)} structured examples")

        print(f"  Rendering to windows...")
        windows = render_examples_to_windows(examples, template, tokenizer, seq_len=seq_len)
        print(f"    {len(windows)} training windows")

        if len(windows) < 4:
            print("    Too few windows, padding...")
            windows = windows * (4 // len(windows) + 1)

        X = torch.tensor([w[0] for w in windows], dtype=torch.long)
        Y = torch.tensor([w[1] for w in windows], dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X, Y)
        n_val = max(1, int(len(dataset) * 0.1))
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    total_batches = len(train_loader)
    print(f"    {len(train_loader.dataset)} train / {len(val_loader.dataset)} val "
          f"({total_batches} batches/epoch)")

    # ── Test BEFORE fine-tuning ──────────────────────────────────────────────
    print(f"\n{sep}")
    print("  BEFORE fine-tuning:")
    test_prompts = [
        "Query: What is 12 + 15?\nReasoning:",
        "Query: A rectangle has length 5 and width 3.\nReasoning:",
        "Query: What is the capital of France?\nReasoning:",
    ]
    for p in test_prompts:
        test_inference(model, tokenizer, p, device=device)

    # ── AMP setup ─────────────────────────────────────────────────────────
    use_amp = use_amp and device.startswith("cuda")
    scaler = GradScaler("cuda", enabled=use_amp) if _new_amp else GradScaler(enabled=use_amp)
    if use_amp:
        print("  Using AMP (mixed precision) — ~2x speedup on T4")

    # ── Fine-tune ────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  Fine-tuning ({epochs} epochs)...")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            ac = autocast("cuda", enabled=use_amp) if _new_amp else autocast(enabled=use_amp)
            with ac:
                logits = model(bx)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), by.view(-1), ignore_index=-100)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item()
        sched.step()

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                ac = autocast("cuda", enabled=use_amp) if _new_amp else autocast(enabled=use_amp)
                with ac:
                    logits = model(bx)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), by.view(-1), ignore_index=-100)
                vloss += loss.item()
        avg_t = total_loss / max(total_batches, 1)
        avg_v = vloss / max(len(val_loader), 1)
        print(f"    Epoch {epoch:2d}/{epochs}  train={avg_t:.4f}  val={avg_v:.4f}  ({time.time()-t0:.0f}s)")

    # ── Test AFTER fine-tuning ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("  AFTER fine-tuning:")
    for p in test_prompts:
        test_inference(model, tokenizer, p, device=device)

    # ── Save fine-tuned model ────────────────────────────────────────────────
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = FINETUNE_DIR / "finetuned.pt"
    tok_path = FINETUNE_DIR / "finetuned_tokenizer.json"
    actual_layers = len(model.backbone.blocks)
    actual_dim = model.dim
    actual_heads = model.backbone.blocks[0].num_heads if actual_layers > 0 else heads
    cfg = dict(vocab_size=tokenizer.vocab_size_real, hidden_dim=actual_dim,
               num_layers=actual_layers, num_heads=actual_heads,
               n_kv_heads=max(1, actual_heads // 4),
               max_seq=seq_len * 4, dropout=0.05, seq_len=seq_len,
               epochs=epochs, template=template_name, prompt_ratio=0.3)
    save_lm(model, tokenizer, config=cfg, epoch=epochs, loss=avg_v, path=str(model_path))
    tokenizer.save(str(tok_path))

    manifest = dict(
        name="Fine-tuned Model", base="foundation/base_model.pt",
        template=template_name, epochs=epochs,
        loss=round(float(avg_v), 4), examples=len(examples) if not data_dir else 0,
        files=str(data_file) if not data_dir else data_dir,
        weights_file=str(model_path), tokenizer_file=str(tok_path),
    )
    (FINETUNE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total_time = time.time() - t0
    print(f"\n{sep}")
    print(f"  Done in {total_time:.0f}s — saved to {model_path}")
    print(f"{sep}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto fine-tune foundation model on reasoning data")
    parser.add_argument("--data-dir", default="", help="Directory of structured JSON/JSONL data")
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--template", default="step_by_step", help="Template name")
    parser.add_argument("--generate", type=int, default=5000, help="Q&A examples to generate")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    auto_finetune(
        data_dir=args.data_dir or None,
        epochs=args.epochs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        template_name=args.template,
        generate_count=args.generate,
        device=args.device,
        use_amp=not args.no_amp,
    )
