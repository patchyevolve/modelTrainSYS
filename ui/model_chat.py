import os
import sys
import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List

from core.implementations import GenConfig


R  = "\033[0m"
B  = "\033[1m"
C  = "\033[96m"
G  = "\033[92m"
Y  = "\033[93m"
M  = "\033[95m"
DIM = "\033[2m"

def _c(text, code): return f"{code}{text}{R}"
def cyan(t):    return _c(t, C)
def green(t):   return _c(t, G)
def yellow(t):  return _c(t, Y)
def magenta(t): return _c(t, M)
def dim(t):     return _c(t, DIM)
def bold(t):    return _c(t, B)


def _find_model(name_or_path: Optional[str]) -> Path:
    if name_or_path:
        p = Path(name_or_path)
        if p.exists():
            return p
        for ext in ("", ".pt"):
            candidates = list(Path("trained_models").glob(f"{name_or_path}*{ext}"))
            pts = [c for c in candidates if c.suffix == ".pt"]
            if pts:
                return pts[0]
        raise FileNotFoundError(
            f"Model '{name_or_path}' not found.\n"
            f"Run: python start.py --list-models")
    pts = sorted(Path("trained_models").glob("*.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    if not pts:
        raise FileNotFoundError(
            "No trained models found in trained_models/\n"
            "Train one first: python start.py --ui")
    return pts[0]


def _load_meta(pt_path: Path) -> Dict:
    meta_path = pt_path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


class TextGenSession:
    def __init__(self, pt_path: Path, ckpt: Dict, meta: Dict):
        from core.text_model import load_lm

        self.model, self.tokenizer = load_lm(str(pt_path))
        lm_cfg = ckpt.get("model_config") or ckpt.get("config") or {}
        self.seq_len = lm_cfg.get("seq_len", lm_cfg.get("max_seq", 128))

        print(cyan(f"\n  Model   : {pt_path.name}"))
        print(cyan(f"  Type    : Text Generation (Decoder-Only Transformer)"))
        print(cyan(f"  Vocab   : {self.tokenizer.vocab_size_real} tokens (BPE)"))
        print(cyan(f"  Params  : {sum(p.numel() for p in self.model.parameters()):,}"))
        print(cyan(f"  Context : {self.seq_len} tokens"))
        print()
        print(dim("  Controls:"))
        print(dim("    /temp 0.8       — temperature (0.1=focused, 1.5=creative)"))
        print(dim("    /topk 40        — top-k sampling"))
        print(dim("    /minp 0.05      — min-p sampling (0=off)"))
        print(dim("    /typical 0.9    — typical sampling (0=off)"))
        print(dim("    /rep 1.1        — repetition penalty"))
        print(dim("    /len 300        — max tokens to generate"))
        print(dim("    /reason on/off  — reasoning mode"))
        print(dim("    /quit           — exit"))
        print()

        self.gen_cfg = GenConfig(
            temperature=0.8, top_k=40, top_p=0.9, min_p=0.05,
            repetition_penalty=1.1, max_new_tokens=300, eos_id=3)
        self.reasoning = True

    def run(self):
        print(bold("  Type a prompt and press Enter. The model will continue writing.\n"))
        while True:
            try:
                user = input(magenta("  You › ")).strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  Goodbye.")
                break

            if not user:
                continue

            if user.startswith("/quit"):
                print("  Goodbye.")
                break
            if user.startswith("/temp"):
                try:
                    self.gen_cfg.temperature = float(user.split()[1])
                    print(dim(f"  temperature = {self.gen_cfg.temperature}"))
                except Exception:
                    print(yellow("  Usage: /temp 0.8"))
                continue
            if user.startswith("/topk"):
                try:
                    self.gen_cfg.top_k = int(user.split()[1])
                    print(dim(f"  top_k = {self.gen_cfg.top_k}"))
                except Exception:
                    print(yellow("  Usage: /topk 40"))
                continue
            if user.startswith("/minp"):
                try:
                    self.gen_cfg.min_p = float(user.split()[1])
                    print(dim(f"  min_p = {self.gen_cfg.min_p}"))
                except Exception:
                    print(yellow("  Usage: /minp 0.05"))
                continue
            if user.startswith("/typical"):
                try:
                    self.gen_cfg.typical_p = float(user.split()[1])
                    print(dim(f"  typical_p = {self.gen_cfg.typical_p}"))
                except Exception:
                    print(yellow("  Usage: /typical 0.9"))
                continue
            if user.startswith("/rep"):
                try:
                    self.gen_cfg.repetition_penalty = float(user.split()[1])
                    print(dim(f"  repetition_penalty = {self.gen_cfg.repetition_penalty}"))
                except Exception:
                    print(yellow("  Usage: /rep 1.1"))
                continue
            if user.startswith("/len"):
                try:
                    self.gen_cfg.max_new_tokens = int(user.split()[1])
                    print(dim(f"  max_new = {self.gen_cfg.max_new_tokens}"))
                except Exception:
                    print(yellow("  Usage: /len 300"))
                continue
            if user.startswith("/reason"):
                try:
                    val = user.split()[1].lower()
                    self.reasoning = (val == "on" or val == "true")
                    print(dim(f"  reasoning = {self.reasoning}"))
                except Exception:
                    print(yellow("  Usage: /reason on/off"))
                continue

            full_prompt = user
            if self.reasoning:
                if "Thought:" not in user:
                    full_prompt = f"Question: {user}\nThought:"
                print(dim("  Reasoning…"))

            print(green("  Model › "), end="", flush=True)
            prompt_ids = self.tokenizer.encode(full_prompt)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            output_ids = self.model.generate(
                prompt_ids, cfg=self.gen_cfg, device=device)

            full_text = self.tokenizer.decode(output_ids, skip_special=True)
            print(green(full_text), end="", flush=True)
            print("\n")


def start_chat(model_name: Optional[str] = None):
    if sys.platform == "win32":
        os.system("")

    print()
    print(bold(cyan("  ╔══════════════════════════════════════════╗")))
    print(bold(cyan("  ║     ML SYSTEM — TEXT GENERATION CHAT    ║")))
    print(bold(cyan("  ╚══════════════════════════════════════════╝")))
    print()

    try:
        pt_path = _find_model(model_name)
    except FileNotFoundError as e:
        print(yellow(f"  {e}"))
        return

    print(dim(f"  Loading {pt_path.name}…"))
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    meta = _load_meta(pt_path)

    session = TextGenSession(pt_path, ckpt, meta)
    session.run()
