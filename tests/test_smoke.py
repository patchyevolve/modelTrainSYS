import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from data.advanced_tokenizer import AdvancedTokenizer, train_tokenizer
from data.text_dataset import build_text_loaders, read_text_files
from core.implementations import HMTLanguageModel, GenConfig
from core.text_model import lm_val_loss, save_lm, load_lm
from training.unified_trainer import UnifiedTrainer, TrainConfig

CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning is transforming artificial intelligence. "
    "Transformers are a neural network architecture for sequence modeling. "
    "Attention mechanisms focus on relevant parts of the input. "
    "Large language models use decoder-only architectures with GQA and RoPE. "
) * 5

def test_tokenizer():
    tok = train_tokenizer([CORPUS], vocab_size=256, save_path="/tmp/smoke_tok.json")
    assert tok.vocab_size_real > 10
    ids = tok.encode("hello world")
    assert len(ids) > 0
    decoded = tok.decode(ids)
    assert len(decoded) > 0
    print(f"  tokenizer: vocab={tok.vocab_size_real}, encode/decode OK")
    return tok

def test_train_model(tok):
    config = TrainConfig(
        epochs=2,
        batch_size=2,
        lr=0.001,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        seq_len=32,
        use_amp=False,
        grad_accum_steps=1,
    )
    with open("/tmp/smoke_corpus.txt", "w") as f:
        f.write(CORPUS)

    trainer = UnifiedTrainer(
        config, files=["/tmp/smoke_corpus.txt"],
        progress_callback=lambda **kw: None,
        log_callback=lambda msg, lvl="info": print(f"    {lvl}: {msg}"),
    )
    result = trainer.run()
    assert result is not None
    assert result.model is not None
    assert len(result.metrics["train_loss"]) == config.epochs
    print(f"  trainer: {config.epochs} epochs, final loss={result.metrics['train_loss'][-1]:.4f}")
    return result

def test_generate(result, tok):
    model = result.model
    model.eval()
    prompt_ids = tok.encode("machine learning")
    cfg = GenConfig(temperature=0.8, top_k=20, max_new_tokens=10, eos_id=3)
    out = model.generate(prompt_ids, cfg=cfg, device="cpu")
    assert len(out) > 0
    text = tok.decode(out, skip_special=True)
    print(f"  generate: prompt='machine learning' → '{text[:60]}' ({len(out)} tokens)")
    return out

def test_save_load(result, tok):
    cfg = {
        "vocab_size": result.info.get("vocab_size", 434),
        "hidden_dim": 64, "num_layers": 2, "num_heads": 4,
        "seq_len": 32, "dim": 64,
    }
    save_lm(result.model, tok, cfg, path="/tmp/smoke_model.pt")
    assert os.path.exists("/tmp/smoke_model.pt")
    model2, tok2 = load_lm("/tmp/smoke_model.pt", device="cpu")
    assert model2 is not None
    assert tok2 is not None
    n = sum(p.numel() for p in model2.parameters())
    assert n > 0
    print(f"  save/load: checkpoint OK ({n} params)")
    os.remove("/tmp/smoke_model.pt")
    for ext in (".tokenizer", ".tokenizer.json", ".tok", ".json"):
        p = f"/tmp/smoke_model{ext}"
        if os.path.exists(p):
            os.remove(p)

def test_val_loss(result, tok):
    loader = build_text_loaders(
        ["/tmp/smoke_corpus.txt"], seq_len=32, batch_size=2, tokenizer=tok
    )[1]
    loss = lm_val_loss(result.model, loader)
    assert loss > 0
    print(f"  val_loss: {loss:.4f}")

if __name__ == "__main__":
    print("Smoke test suite")
    print("-" * 50)
    tok = test_tokenizer()
    result = test_train_model(tok)
    test_generate(result, tok)
    test_save_load(result, tok)
    test_val_loss(result, tok)
    print("-" * 50)
    print("ALL SMOKE TESTS PASSED")
    os.remove("/tmp/smoke_corpus.txt")
