"""
ML Training System — unified entry point.

  python start.py              → interactive menu (same as run.bat)
  python start.py --ui         → Training GUI
  python start.py --chat       → Chat with latest trained model
  python start.py --chat NAME  → Chat with specific model
  python start.py --inference  → Run inference (CLI)
  python start.py --upgrade    → Auto-Upgrade window
  python start.py --list       → List all trained models
  python start.py --check      → Health check
  python start.py --install    → Install dependencies

  Or double-click run.bat for the same menu in a Windows terminal.
"""

import sys
import os
import json
import importlib
import importlib.util
import types
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Enable ANSI colours on Windows ───────────────────────────────────────────
if sys.platform == "win32":
    os.system("")

# ── Colour helpers ────────────────────────────────────────────────────────────
def _c(t, code): return f"\033[{code}m{t}\033[0m"
def cyan(t):   return _c(t, "96")
def green(t):  return _c(t, "92")
def yellow(t): return _c(t, "93")
def bold(t):   return _c(t, "1")
def dim(t):    return _c(t, "2")

# ── Register package namespaces so relative imports resolve ───────────────────
for _pkg in ["mlsystem", "mlsystem.core", "mlsystem.cybersec", "mlsystem.interface"]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = types.ModuleType(_pkg)


def _load(*names, filepath):
    primary = names[0]
    spec = importlib.util.spec_from_file_location(primary, filepath)
    mod  = importlib.util.module_from_spec(spec)
    for n in names:
        sys.modules[n] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Load all modules (lazy — only when needed) ────────────────────────────────
_loaded = {}

def _ensure_loaded():
    """Load all core modules into sys.modules. Called once before any use."""
    if _loaded:
        return
    _loaded["arch"]   = _load("mlsystem.core.architecture",
                               "mlsystem.cybersec.architecture",
                               "mlsystem.interface.architecture",
                               filepath=ROOT / "architecture.py")
    _loaded["impls"]  = _load("mlsystem.core.implementations",
                               filepath=ROOT / "implementations.py")
    _loaded["refl"]   = _load("mlsystem.core.reflector_trainer",
                               "mlsystem.cybersec.reflector_trainer",
                               filepath=ROOT / "reflector_trainer.py")
    _loaded["upg"]    = _load("mlsystem.core.auto_upgrade",
                               filepath=ROOT / "auto_upgrade.py")
    _loaded["csec"]   = _load("mlsystem.cybersec.trainer",
                               filepath=ROOT / "trainer.py")
    _loaded["chat_m"] = _load("mlsystem.interface.chat",
                               filepath=ROOT / "chat.py")


# ── Public aliases (available after _ensure_loaded) ───────────────────────────
def _arch():   _ensure_loaded(); return _loaded["arch"]
def _impls():  _ensure_loaded(); return _loaded["impls"]
def _refl():   _ensure_loaded(); return _loaded["refl"]
def _upg():    _ensure_loaded(); return _loaded["upg"]


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM FACTORY
# ═════════════════════════════════════════════════════════════════════════════

def build_default_system(with_upgrade: bool = True):
    """Build and return a fully wired MLSystemOrchestrator."""
    import torch.nn as nn
    a = _arch(); i = _impls(); r = _refl(); u = _upg()

    MLSystemOrchestrator = a.MLSystemOrchestrator
    ModuleConfig         = a.ModuleConfig
    ComponentType        = a.ComponentType
    DataType             = a.DataType

    system = MLSystemOrchestrator()
    system.register_module(i.ImageFeeder(ModuleConfig(
        "image_feeder", ComponentType.FEEDER, input_types=[DataType.IMAGE])))
    system.register_module(i.HierarchicalMambaEncoder(ModuleConfig(
        "encoder", ComponentType.ENCODER,
        params={"input_dim": 256, "hidden_dim": 512,
                "num_layers": 4, "num_heads": 8})))
    system.register_module(i.TransformerDecoder(ModuleConfig(
        "decoder", ComponentType.DECODER,
        params={"latent_dim": 512, "output_dim": 256,
                "num_heads": 8, "num_layers": 4})))
    system.register_module(r.LLMReflector(ModuleConfig(
        "reflector", ComponentType.REFLECTOR,
        params={"input_dim": 256, "hidden_dim": 128})))
    system.set_pipeline(["image_feeder", "encoder", "decoder", "reflector"])

    if with_upgrade:
        probe = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128))
        upg_sys = u.AutoUpgradeSystem(a.ModuleConfig(
            "auto_upgrade", a.ComponentType.INFERENCE,
            params={"model": probe, "training_history": {}}))
        upg_sys.initialize()
        system.modules["auto_upgrade"] = upg_sys
        system._upgrade_system = upg_sys

    return system


# ═════════════════════════════════════════════════════════════════════════════
# MODEL UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def list_models() -> list:
    save_dir = Path("trained_models")
    if not save_dir.exists():
        return []
    models = []
    for f in sorted(save_dir.glob("*.json"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(f) as fp:
                m = json.load(fp)
                m["_meta_path"] = str(f)
                models.append(m)
        except Exception:
            pass
    return models


def print_models():
    models = list_models()
    if not models:
        print(yellow("  No trained models found in trained_models/"))
        print(dim("  Train one first: python start.py --ui"))
        return
    print(f"\n{'─'*82}")
    print(f"  {bold('NAME'):<48} {bold('TYPE'):<22} {bold('ACC'):<9} {bold('STATUS')}")
    print(f"{'─'*82}")
    for m in models:
        name   = m.get("name", "—")[:46]
        mtype  = m.get("model_type", "—")[:20]
        acc    = str(m.get("accuracy", "—"))[:7]
        status = m.get("status", "—")
        wf     = m.get("weights_file")
        tag    = green("✓ .pt") if wf and Path(wf).exists() else yellow("meta only")
        print(f"  {name:<48} {mtype:<22} {acc:<9} {status}  [{tag}]")
    print(f"{'─'*82}")
    print(f"  {len(models)} model(s)\n")


def load_model_for_inference(name_or_path: str):
    """
    Load any trained model by name or .pt path.
    Uses HMTClassifier for classifiers, load_lm for language models.
    Returns (model, config, data_info).
    """
    import torch
    from implementations import HMTClassifier

    pt_path = Path(name_or_path)
    if not pt_path.exists():
        candidates = sorted(
            Path("trained_models").glob(f"{name_or_path}*.pt"),
            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(
                f"No .pt file for '{name_or_path}'. "
                f"Run: python start.py --list")
        pt_path = candidates[0]

    ckpt      = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    cfg       = ckpt.get("config", {})
    data_info = ckpt.get("data_info", {})

    # Language model checkpoint
    if ckpt.get("model_config"):
        from text_model import load_lm
        model, tok = load_lm(str(pt_path))
        print(green(f"✓ LM loaded: {pt_path.name}  "
                    f"({model.count_parameters():,} params)"))
        return model, cfg, data_info

    # Classifier checkpoint — rebuild with HMTClassifier
    feat_dim  = data_info.get("feature_dim", 16)
    hidden    = cfg.get("hidden_dim", 128)
    layers    = cfg.get("num_layers", 3)
    num_heads = max(1, min(8, hidden // 64))
    hidden    = (hidden // num_heads) * num_heads

    model = HMTClassifier(
        input_dim=feat_dim, num_classes=1,
        dim=hidden, num_layers=layers,
        num_heads=num_heads, num_scales=3)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(green(f"✓ Loaded: {pt_path.name}  "
                f"({sum(p.numel() for p in model.parameters()):,} params)"))
    return model, cfg, data_info


# ═════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═════════════════════════════════════════════════════════════════════════════

REQUIRED_FILES = [
    ("architecture.py",      "Core architecture & base classes"),
    ("implementations.py",   "HMT backbone — Mamba + Transformer"),
    ("reflector_trainer.py", "Reflector + LLM reflector + trainer"),
    ("auto_upgrade.py",      "Self-upgrade system + SQLite DB"),
    ("project_context.py",    "Project file context storage"),
    ("smart_upgrade.py",     "Smart upgrade with full project analysis"),
    ("trainer.py",           "Cybersecurity adversarial trainer"),
    ("chat.py",              "System CLI interface"),
    ("training_ui.py",       "Visual training GUI"),
    ("upgrade_window.py",    "Auto-upgrade window"),
    ("data_loader.py",       "CSV / NPY data loader"),
    ("inference.py",         "Classifier inference engine"),
    ("text_dataset.py",      "Text corpus loader + tokenizer"),
    ("text_model.py",        "HMT language model shim"),
    ("model_chat.py",        "Model chat / inference session"),
    ("run.bat",              "Windows batch launcher"),
    ("mamba_kernel.cpp",     "C++ Mamba kernel (optional)"),
]


def health_check(verbose: bool = True) -> bool:
    if verbose:
        print(bold(cyan("\n── Codebase Health ──────────────────────────────────")))
    missing = 0
    for fname, desc in REQUIRED_FILES:
        ok  = Path(fname).exists()
        opt = fname.endswith(".cpp")
        if verbose:
            icon = green("✓") if ok else (yellow("⚠ opt") if opt else yellow("✗ MISSING"))
            print(f"  {icon}  {fname:<28} {dim(desc)}")
        if not ok and not opt:
            missing += 1
    if verbose:
        print()
        if missing == 0:
            print(green("  All required files present.\n"))
        else:
            print(yellow(f"  {missing} file(s) missing.\n"))
    return missing == 0


# ═════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MENU  (shown when no args given)
# ═════════════════════════════════════════════════════════════════════════════

def interactive_menu():
    while True:
        print()
        print(bold(cyan("  ╔══════════════════════════════════════════════════════╗")))
        print(bold(cyan("  ║         ML TRAINING SYSTEM                          ║")))
        print(bold(cyan("  ║   Hierarchical Mamba + Transformer  │  Groq LLM     ║")))
        print(bold(cyan("  ╚══════════════════════════════════════════════════════╝")))
        print()
        print(f"  {bold('1')}  {cyan('Training GUI')}          — drag-and-drop training window")
        print(f"  {bold('2')}  {cyan('Chat with model')}       — talk to a trained model")
        print(f"  {bold('3')}  {cyan('Run inference')}         — classify / generate on data")
        print(f"  {bold('4')}  {cyan('List trained models')}   — see all saved models")
        print(f"  {bold('5')}  {cyan('Auto-Upgrade window')}   — LLM-powered model upgrader")
        print(f"  {bold('6')}  {cyan('Health check')}          — verify all files present")
        print(f"  {bold('7')}  {cyan('Install dependencies')}  — pip install required packages")
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
            _cmd_inference()
        elif choice == "4":
            print_models()
            input(dim("  Press Enter to continue…"))
        elif choice == "5":
            _cmd_upgrade()
        elif choice == "6":
            health_check()
            input(dim("  Press Enter to continue…"))
        elif choice == "7":
            _cmd_install()
        elif choice == "8":
            print("  Goodbye.")
            break
        else:
            print(yellow("  Invalid choice."))


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════

def _cmd_ui():
    health_check(verbose=False)
    system     = build_default_system(with_upgrade=True)
    upgrade_sys = system.modules.get("auto_upgrade")

    try:
        from tkinterdnd2 import TkinterDnD
        import training_ui as ui
        class App(ui.TrainingApp, TkinterDnD.Tk):
            def __init__(self):
                TkinterDnD.Tk.__init__(self)
                ui.TrainingApp.__init__(self)
                self._upgrade_system = upgrade_sys
        App().mainloop()
    except ImportError:
        import training_ui as ui
        app = ui.TrainingApp()
        app._upgrade_system = upgrade_sys
        app.mainloop()


def _cmd_chat(model_name):
    from model_chat import start_chat
    if model_name is None:
        print_models()
        try:
            model_name = input(
                dim("  Model name (Enter = latest): ")).strip() or None
        except (KeyboardInterrupt, EOFError):
            return
    start_chat(model_name)


def _cmd_inference():
    print_models()
    try:
        model_name = input(dim("  Model name (Enter = latest): ")).strip() or None
        data_file  = input(dim("  Data file  (Enter = randomDATA/): ")).strip() or None
    except (KeyboardInterrupt, EOFError):
        return

    args = ["inference.py", "--save"]
    if model_name:
        pt = Path("trained_models") / f"{model_name}.pt"
        if not pt.exists():
            pts = list(Path("trained_models").glob(f"{model_name}*.pt"))
            pt  = pts[0] if pts else pt
        args += ["--model", str(pt)]
    if data_file:
        args += ["--data", data_file]

    import subprocess
    subprocess.run([sys.executable] + args)


def _cmd_upgrade():
    health_check(verbose=False)
    system      = build_default_system(with_upgrade=True)
    upgrade_sys = system.modules.get("auto_upgrade")
    import tkinter as tk
    from upgrade_window import AutoUpgradeWindow
    root = tk.Tk()
    root.withdraw()
    win  = AutoUpgradeWindow(root, upgrade_sys)
    win.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


def _cmd_smart_upgrade():
    from smart_upgrade import quick_upgrade
    print(cyan("\n  Running Smart Upgrade..."))
    result = quick_upgrade()
    print(green(f"\n  Applied {result.get('applied_count', 0)} upgrades"))
    stats = result.get('groq_stats', {})
    print(dim(f"  API calls: {stats.get('total_calls', 0)}, Cache hits: {stats.get('cache_hits', 0)}\n"))


def _cmd_install():
    import subprocess
    pkgs = ["torch", "torchvision", "numpy", "pandas", "Pillow", "tkinterdnd2"]
    print(cyan(f"\n  Installing: {', '.join(pkgs)}\n"))
    subprocess.run([sys.executable, "-m", "pip", "install"] + pkgs)
    print(green("\n  Done. Optional GPU support:"))
    print(dim("  pip install torch --index-url "
              "https://download.pytorch.org/whl/cu121\n"))


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        interactive_menu()
        sys.exit(0)

    flag = args[0].lstrip("-").lower()

    if flag == "ui":
        _cmd_ui()

    elif flag == "chat":
        model_name = args[1] if len(args) > 1 and not args[1].startswith("-") else None
        _cmd_chat(model_name)

    elif flag in ("inference", "infer"):
        _cmd_inference()

    elif flag in ("upgrade", "upg"):
        _cmd_upgrade()

    elif flag in ("smart-upgrade", "smart"):
        _cmd_smart_upgrade()

    elif flag in ("list", "list-models", "models"):
        print_models()

    elif flag == "check":
        health_check()

    elif flag == "install":
        _cmd_install()

    elif flag == "load":
        if len(args) < 2:
            print(yellow("Usage: python start.py --load <name>"))
            sys.exit(1)
        try:
            model, cfg, info = load_model_for_inference(args[1])
            print(f"\nConfig:\n{json.dumps(cfg, indent=2)}")
            print(f"\nData info:\n{json.dumps(info, indent=2)}")
        except Exception as e:
            print(yellow(f"Error: {e}"))

    else:
        print(yellow(f"Unknown flag: {args[0]}"))
        print(dim("  Valid: --ui  --chat  --inference  --upgrade  "
                  "--list  --check  --install  --load <name>"))
        sys.exit(1)
