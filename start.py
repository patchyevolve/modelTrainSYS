import sys
import os
import json
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    os.system("")

def _c(t, code): return f"\033[{code}m{t}\033[0m"
def cyan(t):   return _c(t, "96")
def green(t):  return _c(t, "92")
def yellow(t): return _c(t, "93")
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")


def print_models():
    save_dir = Path("trained_models")
    if not save_dir.exists():
        print(yellow("  No trained models found."))
        return
    models = []
    for f in sorted(save_dir.glob("*.json"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(f) as fp:
                m = json.load(fp)
                models.append(m)
        except Exception:
            pass
    if not models:
        print(yellow("  No trained models found."))
        return
    print(f"\n{'─'*70}")
    print(f"  {bold('NAME'):<48} {bold('TYPE'):<20}")
    print(f"{'─'*70}")
    for m in models:
        name  = m.get("name", "—")[:46]
        mtype = m.get("model_type", "—")[:18]
        wf    = m.get("weights_file")
        tag   = green("✓ .pt") if wf and Path(wf).exists() else yellow("meta only")
        print(f"  {name:<48} {mtype:<20} [{tag}]")
    print(f"{'─'*70}")
    print(f"  {len(models)} model(s)\n")


def interactive_menu():
    while True:
        print()
        print(bold(cyan("  +----------------------------------------------------+")))
        print(bold(cyan("  |         ML TRAINING SYSTEM                        |")))
        print(bold(cyan("  |       Decoder-Only · GQA · YaRN · BPE Tokenizer   |")))
        print(bold(cyan("  +----------------------------------------------------+")))
        print()
        print(f"  {bold('1')}  {cyan('Training GUI')}          - train reasoning model on ANY text file")
        print(f"  {bold('2')}  {cyan('Chat with model')}       - text generation with saved model")
        print(f"  {bold('3')}  {cyan('Train BPE tokenizer')}   - train tokenizer on text files")
        print(f"  {bold('4')}  {cyan('Evaluate model')}        - perplexity + accuracy")
        print(f"  {bold('5')}  {cyan('List trained models')}   - see all saved models")
        print(f"  {bold('6')}  {cyan('Dry-run check')}         - validate all imports")
        print(f"  {bold('7')}  {cyan('Install dependencies')}  - pip install required packages")
        print(f"  {bold('8')}  {dim('Exit')}")
        print()
        try:
            choice = input("  Enter choice [1-8]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye.")
            break

        if choice == "1":
            _cmd_ui()
        elif choice == "2":
            _cmd_chat(None)
        elif choice == "3":
            _cmd_train_tokenizer()
        elif choice == "4":
            _cmd_eval()
        elif choice == "5":
            print_models()
            input(dim("  Press Enter to continue…"))
        elif choice == "6":
            _cmd_dry_run()
            input(dim("  Press Enter to continue…"))
        elif choice == "7":
            _cmd_install()
        elif choice == "8":
            print("  Goodbye.")
            break
        else:
            print(yellow("  Invalid choice."))


def _cmd_ui():
    from ui import training_ui as ui
    app = ui.TrainingApp()
    app.mainloop()


def _cmd_chat(model_name):
    from ui.model_chat import start_chat
    if model_name is None:
        print_models()
        try:
            model_name = input(
                dim("  Model name (Enter = latest): ")).strip() or None
        except (KeyboardInterrupt, EOFError):
            return
    start_chat(model_name)


def _cmd_train_tokenizer():
    import re
    files = []
    vocab_size = 8192
    save_path = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--") and arg not in ("--train-tokenizer",):
            break
        if arg == "--vocab-size" and i + 1 < len(sys.argv):
            vocab_size = int(sys.argv[i + 1])
            i += 2
        elif arg == "--save" and i + 1 < len(sys.argv):
            save_path = sys.argv[i + 1]
            i += 2
        else:
            p = Path(arg)
            if p.exists() and p.suffix.lower() in (".txt", ".jsonl", ".json", ".csv"):
                files.append(arg)
            i += 1

    if not files:
        inp = input(dim("  Text files (space-separated): ")).strip()
        files = [f.strip() for f in inp.split() if Path(f.strip()).exists()]
        if not files:
            print(yellow("  No valid text files provided."))
            return

    print(cyan(f"\n  Training BPE tokenizer (vocab_size={vocab_size}) on {len(files)} file(s)..."))

    from data.advanced_tokenizer import train_tokenizer
    from data.text_dataset import read_text_files

    corpus = read_text_files(files)
    if len(corpus) < 100:
        print(yellow(f"  Corpus too small ({len(corpus)} chars). Need at least 100."))
        return

    tok = train_tokenizer([corpus], vocab_size=vocab_size, save_path=save_path)
    print(green(f"  Tokenizer trained: vocab={tok.vocab_size_real} tokens"))
    if save_path:
        print(green(f"  Saved to: {save_path}"))
    else:
        out = "tokenizer.bpe"
        tok.save(out)
        print(green(f"  Saved to: {out}"))

    sample = corpus[:200]
    ids = tok.encode(sample)
    decoded = tok.decode(ids)
    print(dim(f"\n  Sample encode/decode:"))
    print(dim(f"    Input:   {sample[:80]}..."))
    print(dim(f"    Tokens:  {ids[:20]}... ({len(ids)} total)"))
    print(dim(f"    Decoded: {decoded[:80]}..."))
    print(dim(f"    Compression: {len(sample)/max(len(ids),1):.1f} chars/token"))


def _cmd_eval():
    print_models()
    try:
        model_name = input(dim("  Model name (Enter = latest): ")).strip() or None
        data_file  = input(dim("  Eval text file (optional): ")).strip() or None
        gsm8k_file = input(dim("  GSM8K test file (optional): ")).strip() or None
    except (KeyboardInterrupt, EOFError):
        return

    from eval.harness import evaluate_model, load_gsm8k, save_eval_report

    if model_name:
        pt_path = Path("trained_models") / f"{model_name}.pt"
        if not pt_path.exists():
            pts = list(Path("trained_models").glob(f"{model_name}*.pt"))
            pt_path = pts[0] if pts else pt_path
        model_path = str(pt_path)
    else:
        pts = sorted(Path("trained_models").glob("*.pt"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
        if not pts:
            print(yellow("  No trained models found."))
            return
        model_path = str(pts[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import torch

    eval_texts = None
    if data_file and Path(data_file).exists():
        from data.text_dataset import read_text_files
        corpus = read_text_files([data_file])
        eval_texts = [corpus[i:i+1000] for i in range(0, len(corpus), 1000)][:20]

    eval_questions = eval_answers = None
    if gsm8k_file and Path(gsm8k_file).exists():
        eval_questions, eval_answers = load_gsm8k(gsm8k_file)

    print(cyan(f"\n  Evaluating {Path(model_path).name} on {device}..."))
    results = evaluate_model(model_path, eval_texts, eval_questions, eval_answers, device=device)

    if "perplexity" in results:
        p = results["perplexity"]
        print(green(f"  Perplexity: {p['perplexity']:.2f}  (loss={p['avg_loss']:.4f}, tokens={p['tokens']})"))
    if "reasoning" in results:
        r = results["reasoning"]
        print(green(f"  Reasoning: {r['accuracy']*100:.1f}% ({r['correct']}/{r['correct']+r['incorrect']})"))

    report_path = f"eval_report_{Path(model_path).stem}.json"
    save_eval_report(results, report_path)


def _cmd_install():
    import subprocess
    pkgs = ["torch", "torchvision", "numpy", "pandas", "Pillow", "tokenizers"]
    print(cyan(f"\n  Installing: {', '.join(pkgs)}\n"))
    subprocess.run([sys.executable, "-m", "pip", "install"] + pkgs)
    print(green("\n  Done. Optional GPU support:"))
    print(dim("  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"))


def _cmd_dry_run():
    import importlib
    modules = [
        ("core", ["implementations", "transformer", "text_model",
                    "device_manager", "memory_monitor", "fast_logit_processors"]),
        ("data", ["advanced_tokenizer", "text_dataset", "chat_dataset"]),
        ("training", ["unified_trainer"]),
        ("eval", ["harness"]),
    ]
    ok = fail = 0
    for pkg, subs in modules:
        for sub in subs:
            name = f"{pkg}.{sub}"
            try:
                importlib.import_module(name)
                ok += 1
            except ImportError as e:
                print(dim(f"  ✗ {name}: {e}"))
                fail += 1
            else:
                print(dim(f"  ✓ {name}"))

    tokenizer_ok = False
    try:
        from data.advanced_tokenizer import AdvancedTokenizer
        t = AdvancedTokenizer(vocab_size=128)
        t.build(["test corpus"])
        tokenizer_ok = t.vocab_size_real > 4
    except Exception:
        pass

    torch_avail = False
    try:
        import torch
        torch_avail = torch.__version__
    except ImportError:
        pass

    print()
    print(green(f"  Modules: {ok} loaded, {fail} failed"))
    print(green(f"  Tokenizer: {'✓' if tokenizer_ok else '✗'}"))
    print(green(f"  Torch: {'✓ ' + torch_avail if torch_avail else '✗ not installed'}"))
    if fail > 0:
        print(yellow("  Some modules require torch to be installed."))
    print()


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        interactive_menu()
        sys.exit(0)

    flag = args[0].lstrip("-").lower()

    if flag in ("ui", "gui"):
        _cmd_ui()
    elif flag == "chat":
        model_name = args[1] if len(args) > 1 and not args[1].startswith("-") else None
        _cmd_chat(model_name)
    elif flag in ("train-tokenizer", "train-tok"):
        _cmd_train_tokenizer()
    elif flag in ("eval", "evaluate"):
        _cmd_eval()
    elif flag in ("list", "list-models", "models"):
        print_models()
    elif flag in ("dry-run", "check"):
        _cmd_dry_run()
    elif flag == "install":
        _cmd_install()
    else:
        print(yellow(f"Unknown flag: {args[0]}"))
        print(dim("  Valid: --ui  --chat  --train-tokenizer  --eval  --list  --dry-run  --install"))
        sys.exit(1)
