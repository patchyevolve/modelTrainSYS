"""
Interactive chat/inference session for any trained model.
Detects model type and provides the right interaction mode:

  Text Generation  → you type a prompt, model continues writing
  Cybersecurity    → you describe a network session, model says attack/benign
  Classifier       → you enter feature values, model predicts class
  Image            → you give an image path, model classifies it

Run via:
  python start.py --chat                     # picks latest model
  python start.py --chat <model_name_or_pt>  # specific model
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Any


# ─── ANSI colours (work on Windows 10+ terminal) ─────────────────────────────
R  = "\033[0m"       # reset
B  = "\033[1m"       # bold
C  = "\033[96m"      # cyan   — system messages
G  = "\033[92m"      # green  — model output
Y  = "\033[93m"      # yellow — warnings / labels
M  = "\033[95m"      # magenta — prompts
DIM = "\033[2m"      # dim    — hints

def _c(text, code): return f"{code}{text}{R}"
def cyan(t):    return _c(t, C)
def green(t):   return _c(t, G)
def yellow(t):  return _c(t, Y)
def magenta(t): return _c(t, M)
def dim(t):     return _c(t, DIM)
def bold(t):    return _c(t, B)


# ─── Model loader ─────────────────────────────────────────────────────────────

def _find_model(name_or_path: Optional[str]) -> Path:
    """Resolve model name → .pt path."""
    if name_or_path:
        p = Path(name_or_path)
        if p.exists():
            return p
        # Try trained_models/
        for ext in ("", ".pt"):
            candidates = list(Path("trained_models").glob(f"{name_or_path}*{ext}"))
            pts = [c for c in candidates if c.suffix == ".pt"]
            if pts:
                return pts[0]
        raise FileNotFoundError(
            f"Model '{name_or_path}' not found.\n"
            f"Run: python start.py --list-models")
    # Latest model
    pts = sorted(Path("trained_models").glob("*.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    if not pts:
        raise FileNotFoundError(
            "No trained models found in trained_models/\n"
            "Train one first: python start.py --ui")
    return pts[0]


def _load_meta(pt_path: Path) -> Dict:
    """Load .json metadata alongside the .pt file."""
    meta_path = pt_path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def _detect_task(ckpt: Dict, meta: Dict) -> str:
    """Detect what kind of model this is."""
    if meta.get("task") == "language_model":
        return "text_generation"
    if meta.get("task") == "image_classification":
        return "image_classification"
    if ckpt.get("model_config"):
        return "text_generation"
    if ckpt.get("model_arch", {}).get("type") == "HMTImageClassifier":
        return "image_classification"
    mtype = meta.get("model_type", "") or ckpt.get("config", {}).get("model_type", "")
    if "Text Generation" in mtype or "TextGen" in mtype:
        return "text_generation"
    if "Cybersecurity" in mtype:
        return "cybersecurity"
    if "Image" in mtype:
        return "image_classification"
    return "classifier"


# ─── Rebuild classifier model ─────────────────────────────────────────────────

def _rebuild_classifier(cfg: Dict, data_info: Dict) -> nn.Module:
    feat_dim = data_info.get("feature_dim", 16)
    hidden   = cfg.get("hidden_dim", 128)
    layers   = cfg.get("num_layers", 3)
    mtype    = cfg.get("model_type", "")

    if "Cybersecurity" in mtype:
        h2 = hidden * 2
        return nn.Sequential(
            nn.Linear(feat_dim, h2),   nn.BatchNorm1d(h2),     nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h2, h2),         nn.BatchNorm1d(h2),     nn.ReLU(),
            nn.Linear(h2, hidden),     nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )
    layer_list = [nn.Linear(feat_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU()]
    for _ in range(max(1, layers - 1)):
        layer_list += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1)]
    layer_list.append(nn.Linear(hidden, 1))
    return nn.Sequential(*layer_list)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class TextGenSession:
    """
    Chat with a trained language model.
    You type a prompt, the model continues writing from it.
    """

    def __init__(self, pt_path: Path, ckpt: Dict, meta: Dict):
        from text_model import MambaLM, load_lm
        from text_dataset import CharTokenizer

        self.model, self.tokenizer = load_lm(str(pt_path))
        lm_cfg = ckpt.get("model_config", {})
        self.seq_len = lm_cfg.get("seq_len", 128)

        print(cyan(f"\n  Model   : {pt_path.name}"))
        print(cyan(f"  Type    : Text Generation (MambaLM)"))
        print(cyan(f"  Vocab   : {self.tokenizer.vocab_size} characters"))
        print(cyan(f"  Params  : {sum(p.numel() for p in self.model.parameters()):,}"))
        print(cyan(f"  Context : {self.seq_len} characters"))
        print()
        print(dim("  Controls:"))
        print(dim("    /temp 0.8     — set temperature (0.1=focused, 1.5=creative)"))
        print(dim("    /topk 40      — set top-k sampling"))
        print(dim("    /len 300      — set max characters to generate"))
        print(dim("    /quit         — exit"))
        print()

        self.temperature = 0.8
        self.top_k       = 40
        self.max_new     = 300

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

            # Commands
            if user.startswith("/quit"):
                print("  Goodbye.")
                break
            if user.startswith("/temp"):
                try:
                    self.temperature = float(user.split()[1])
                    print(dim(f"  temperature = {self.temperature}"))
                except Exception:
                    print(yellow("  Usage: /temp 0.8"))
                continue
            if user.startswith("/topk"):
                try:
                    self.top_k = int(user.split()[1])
                    print(dim(f"  top_k = {self.top_k}"))
                except Exception:
                    print(yellow("  Usage: /topk 40"))
                continue
            if user.startswith("/len"):
                try:
                    self.max_new = int(user.split()[1])
                    print(dim(f"  max_new = {self.max_new}"))
                except Exception:
                    print(yellow("  Usage: /len 300"))
                continue

            # Generate
            print(green("  Model › "), end="", flush=True)
            prompt_ids = self.tokenizer.encode(user)
            new_ids    = self.model.generate(
                prompt_ids,
                max_new     = self.max_new,
                temperature = self.temperature,
                top_k       = self.top_k,
            )
            generated = self.tokenizer.decode(new_ids)
            print(green(generated))
            print()


class CybersecuritySession:
    """
    Chat with a cybersecurity classifier.
    Describe a network session in plain English or enter CSV values.
    The model tells you if it looks like an attack.
    """

    # Feature order from cybersecurity_intrusion_data.csv after preprocessing
    FEATURE_NAMES = [
        "network_packet_size", "login_attempts", "session_duration",
        "ip_reputation_score", "failed_logins", "unusual_time_access",
        # one-hot: protocol_type (ICMP, TCP, UDP)
        "protocol_ICMP", "protocol_TCP", "protocol_UDP",
        # one-hot: encryption_used (AES, DES)
        "encryption_AES", "encryption_DES",
        # one-hot: browser_type (Chrome, Edge, Firefox, Safari, Unknown)
        "browser_Chrome", "browser_Edge", "browser_Firefox",
        "browser_Safari", "browser_Unknown",
    ]

    ATTACK_TYPES = {
        "sql": "SQL Injection",
        "xss": "Cross-Site Scripting",
        "ddos": "DDoS / Flood",
        "malware": "Malware",
        "overflow": "Buffer Overflow",
        "priv": "Privilege Escalation",
        "cred": "Credential Stuffing",
        "zero": "Zero-Day",
    }

    def __init__(self, pt_path: Path, ckpt: Dict, meta: Dict):
        cfg       = ckpt.get("config", meta.get("config", {}))
        data_info = ckpt.get("data_info", {})

        self.model = _rebuild_classifier(cfg, data_info)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.eval()
        self.feat_dim = data_info.get("feature_dim", 16)
        self.cfg      = cfg

        print(cyan(f"\n  Model   : {pt_path.name}"))
        print(cyan(f"  Type    : Cybersecurity Classifier"))
        print(cyan(f"  Accuracy: {meta.get('accuracy', '—')}"))
        print(cyan(f"  Params  : {sum(p.numel() for p in self.model.parameters()):,}"))
        print()
        print(dim("  Modes:"))
        print(dim("    /describe  — describe a session in plain text (uses Groq to parse)"))
        print(dim("    /values    — enter raw feature values manually"))
        print(dim("    /example   — show an example attack session"))
        print(dim("    /quit      — exit"))
        print()

    def run(self):
        print(bold("  Describe a network session or use /values to enter features.\n"))
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
            if user.startswith("/example"):
                self._show_example()
                continue
            if user.startswith("/values"):
                self._manual_values()
                continue
            if user.startswith("/describe"):
                self._describe_mode()
                continue

            # Default: treat as plain-text description → parse with Groq
            self._classify_description(user)

    def _classify_description(self, description: str):
        """Parse a plain-text session description using Groq, then classify."""
        print(dim("  Parsing description with Groq…"))
        features = self._groq_parse(description)
        if features is None:
            print(yellow("  Could not parse. Try /values to enter features manually."))
            return
        self._run_and_show(features, description)

    def _groq_parse(self, description: str):
        """Ask Groq to extract feature values from a plain-text description."""
        try:
            from reflector_trainer import _groq_chat
            prompt = f"""Extract network session features from this description and return ONLY a JSON object.
Description: {description}

Return exactly these keys with numeric values (use 0 or 1 for binary/one-hot):
network_packet_size (integer, bytes), login_attempts (integer), session_duration (float, seconds),
ip_reputation_score (float 0-1, higher=worse), failed_logins (integer), unusual_time_access (0 or 1),
protocol_ICMP (0/1), protocol_TCP (0/1), protocol_UDP (0/1),
encryption_AES (0/1), encryption_DES (0/1),
browser_Chrome (0/1), browser_Edge (0/1), browser_Firefox (0/1), browser_Safari (0/1), browser_Unknown (0/1).

Only one protocol and one browser should be 1. Return ONLY the JSON, no explanation."""

            raw = _groq_chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=300)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            features = [float(parsed.get(k, 0)) for k in self.FEATURE_NAMES]
            return features
        except Exception as e:
            print(yellow(f"  Groq parse failed: {e}"))
            return None

    def _manual_values(self):
        """Let user enter feature values one by one."""
        print(dim(f"  Enter {self.feat_dim} feature values (press Enter for 0):"))
        features = []
        for i, name in enumerate(self.FEATURE_NAMES[:self.feat_dim]):
            try:
                val = input(dim(f"    {name} › ")).strip()
                features.append(float(val) if val else 0.0)
            except ValueError:
                features.append(0.0)
        # Pad if needed
        while len(features) < self.feat_dim:
            features.append(0.0)
        self._run_and_show(features[:self.feat_dim])

    def _show_example(self):
        """Show an example attack session."""
        example = {
            "description": "Large packet size, 5 failed logins, high IP reputation score, "
                           "TCP protocol, no encryption, Chrome browser, unusual time",
            "features": [1200, 5, 45.0, 0.9, 5, 1,
                         0, 1, 0,   # TCP
                         0, 0,      # no encryption
                         1, 0, 0, 0, 0]  # Chrome
        }
        print(dim(f"\n  Example: {example['description']}"))
        self._run_and_show(example["features"], example["description"])

    def _run_and_show(self, features, description: str = ""):
        """Run the model and display result."""
        # Normalise to feat_dim
        feat = features[:self.feat_dim]
        while len(feat) < self.feat_dim:
            feat.append(0.0)

        x      = torch.tensor([feat], dtype=torch.float32)
        with torch.no_grad():
            logit = self.model(x)
            prob  = torch.sigmoid(logit).item()

        is_attack  = prob >= 0.5
        confidence = prob if is_attack else 1 - prob
        label      = "ATTACK" if is_attack else "BENIGN"
        color      = yellow if is_attack else green

        print()
        print(color(f"  ┌─ VERDICT: {label} ({'%.1f' % (confidence*100)}% confidence) ─┐"))
        print(color(f"  │  Attack probability : {prob:.4f}"))
        print(color(f"  │  Threshold          : 0.50"))
        if is_attack:
            print(color(f"  │  Recommendation     : Block / Investigate"))
        else:
            print(color(f"  │  Recommendation     : Allow"))
        print(color(f"  └{'─'*45}┘"))
        print()

    def _describe_mode(self):
        print(dim("  Describe the session (e.g. 'large packets, 3 failed logins, TCP, unusual time'):"))
        try:
            desc = input(magenta("  Session › ")).strip()
            if desc:
                self._classify_description(desc)
        except (KeyboardInterrupt, EOFError):
            pass


class ClassifierSession:
    """
    Generic classifier chat.
    Shows what features the model expects and lets you enter values.
    """

    def __init__(self, pt_path: Path, ckpt: Dict, meta: Dict):
        cfg       = ckpt.get("config", meta.get("config", {}))
        data_info = ckpt.get("data_info", {})

        self.model = _rebuild_classifier(cfg, data_info)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.eval()
        self.feat_dim  = data_info.get("feature_dim", 16)
        self.n_classes = data_info.get("num_classes", 2)
        self.is_binary = data_info.get("is_binary", True)
        self.meta      = meta

        print(cyan(f"\n  Model     : {pt_path.name}"))
        print(cyan(f"  Type      : {meta.get('model_type', 'Classifier')}"))
        print(cyan(f"  Accuracy  : {meta.get('accuracy', '—')}"))
        print(cyan(f"  Features  : {self.feat_dim}"))
        print(cyan(f"  Classes   : {self.n_classes}"))
        print()
        print(dim("  Commands:"))
        print(dim("    /values   — enter feature values manually"))
        print(dim("    /csv      — paste a CSV row"))
        print(dim("    /info     — show model info"))
        print(dim("    /quit     — exit"))
        print()

    def run(self):
        print(bold("  Enter feature values to get a prediction.\n"))
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
            if user.startswith("/info"):
                self._show_info()
                continue
            if user.startswith("/values"):
                self._manual_values()
                continue
            if user.startswith("/csv"):
                self._csv_row()
                continue

            # Try to parse as comma-separated numbers
            try:
                vals = [float(v.strip()) for v in user.split(",")]
                self._predict(vals)
            except ValueError:
                print(yellow("  Enter comma-separated numbers, or use /values for guided input."))

    def _manual_values(self):
        print(dim(f"  Enter {self.feat_dim} values (press Enter for 0):"))
        vals = []
        for i in range(self.feat_dim):
            try:
                v = input(dim(f"    feature_{i} › ")).strip()
                vals.append(float(v) if v else 0.0)
            except ValueError:
                vals.append(0.0)
        self._predict(vals)

    def _csv_row(self):
        print(dim("  Paste a CSV row (comma-separated values):"))
        try:
            row = input(magenta("  Row › ")).strip()
            vals = [float(v.strip()) for v in row.split(",")]
            self._predict(vals)
        except (ValueError, KeyboardInterrupt):
            print(yellow("  Could not parse row."))

    def _predict(self, values):
        feat = list(values)[:self.feat_dim]
        while len(feat) < self.feat_dim:
            feat.append(0.0)

        x = torch.tensor([feat], dtype=torch.float32)
        with torch.no_grad():
            logit = self.model(x)
            prob  = torch.sigmoid(logit).item()

        label = "Class 1" if prob >= 0.5 else "Class 0"
        conf  = prob if prob >= 0.5 else 1 - prob
        color = yellow if prob >= 0.5 else green

        print()
        print(color(f"  Prediction : {label}  ({conf*100:.1f}% confidence)"))
        print(color(f"  Probability: {prob:.4f}"))
        print()

    def _show_info(self):
        cfg = self.meta.get("config", {})
        print(cyan(f"\n  Name       : {self.meta.get('name','—')}"))
        print(cyan(f"  Type       : {self.meta.get('model_type','—')}"))
        print(cyan(f"  Trained on : {self.meta.get('train_rows','—')} rows"))
        print(cyan(f"  Features   : {self.feat_dim}"))
        print(cyan(f"  Hidden dim : {cfg.get('hidden_dim','—')}"))
        print(cyan(f"  Epochs     : {cfg.get('epochs','—')}"))
        print(cyan(f"  Accuracy   : {self.meta.get('accuracy','—')}"))
        print()


class ImageSession:
    """
    Classify images with a trained HMTImageClassifier.
    You give an image path (or folder), model returns class + confidence.
    """

    def __init__(self, pt_path: Path, ckpt: Dict, meta: Dict):
        from implementations import HMTImageClassifier

        arch        = ckpt.get("model_arch", {})
        class_names = ckpt.get("class_names",
                               meta.get("class_names", ["class_0"]))
        n_classes   = len(class_names)
        dim         = arch.get("dim", meta.get("config", {}).get("hidden_dim", 128))
        num_layers  = arch.get("num_layers",
                               meta.get("config", {}).get("num_layers", 2))
        patch_size  = arch.get("patch_size", 16)
        img_size    = arch.get("img_size",
                               meta.get("img_size", 64))
        num_heads   = max(1, min(8, dim // 64))
        dim         = (dim // num_heads) * num_heads

        self.model = HMTImageClassifier(
            num_classes=n_classes, dim=dim,
            patch_size=patch_size, num_layers=num_layers,
            num_heads=num_heads, num_scales=3)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.eval()

        self.class_names = class_names
        self.img_size    = img_size
        self.n_classes   = n_classes

        print(cyan(f"\n  Model      : {pt_path.name}"))
        print(cyan(f"  Type       : Image Classification"))
        print(cyan(f"  Classes    : {class_names}"))
        print(cyan(f"  Image size : {img_size}×{img_size}"))
        print(cyan(f"  Accuracy   : {meta.get('accuracy', '—')}"))
        print()
        print(dim("  Commands:"))
        print(dim("    /classify <path>  — classify a single image"))
        print(dim("    /folder <path>    — classify all images in a folder"))
        print(dim("    /quit             — exit"))
        print()

    def run(self):
        print(bold("  Enter an image path to classify.\n"))
        while True:
            try:
                user = input(magenta("  Image path › ")).strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  Goodbye.")
                break

            if not user:
                continue
            if user.startswith("/quit"):
                print("  Goodbye.")
                break
            if user.startswith("/folder"):
                parts = user.split(maxsplit=1)
                folder = parts[1].strip() if len(parts) > 1 else "."
                self._classify_folder(folder)
                continue
            if user.startswith("/classify"):
                parts = user.split(maxsplit=1)
                path  = parts[1].strip() if len(parts) > 1 else ""
                if path:
                    self._classify_one(path)
                continue

            # Default: treat as image path
            self._classify_one(user)

    def _preprocess(self, img_path: str):
        """Load and preprocess one image."""
        import torch
        from PIL import Image
        from torchvision import transforms

        tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(img_path).convert("RGB")
        return tf(img).unsqueeze(0)   # (1, 3, H, W)

    def _classify_one(self, img_path: str):
        import torch, torch.nn.functional as F
        p = Path(img_path)
        if not p.exists():
            print(yellow(f"  File not found: {img_path}"))
            return

        try:
            x      = self._preprocess(str(p))
            with torch.no_grad():
                logits = self.model(x)                 # (1, n_classes)
                probs  = F.softmax(logits, dim=-1)[0]  # (n_classes,)

            top_idx  = probs.argmax().item()
            top_prob = probs[top_idx].item()
            label    = self.class_names[top_idx] if top_idx < len(self.class_names) \
                       else f"class_{top_idx}"

            print()
            print(green(f"  ┌─ {p.name} ─────────────────────────────────┐"))
            print(green(f"  │  Prediction : {label}"))
            print(green(f"  │  Confidence : {top_prob*100:.1f}%"))
            if self.n_classes <= 10:
                print(green(f"  │  All classes:"))
                sorted_probs = sorted(enumerate(probs.tolist()),
                                      key=lambda x: x[1], reverse=True)
                for idx, prob in sorted_probs[:5]:
                    cls = self.class_names[idx] if idx < len(self.class_names) \
                          else f"class_{idx}"
                    bar = "█" * int(prob * 20)
                    print(green(f"  │    {cls:<15} {prob*100:5.1f}%  {bar}"))
            print(green(f"  └{'─'*46}┘"))
            print()
        except Exception as e:
            print(yellow(f"  Error: {e}"))

    def _classify_folder(self, folder_path: str):
        from pathlib import Path as P
        IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        folder   = P(folder_path)
        if not folder.exists():
            print(yellow(f"  Folder not found: {folder_path}"))
            return
        imgs = [p for p in folder.iterdir()
                if p.suffix.lower() in IMG_EXTS]
        if not imgs:
            print(yellow("  No images found in folder."))
            return
        print(cyan(f"\n  Classifying {len(imgs)} images in {folder.name}/\n"))
        for img in sorted(imgs)[:20]:   # cap at 20
            self._classify_one(str(img))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def start_chat(model_name: Optional[str] = None):
    """
    Main entry point. Detects model type and starts the right session.
    """
    # Enable ANSI on Windows
    if sys.platform == "win32":
        os.system("")

    print()
    print(bold(cyan("  ╔══════════════════════════════════════════╗")))
    print(bold(cyan("  ║     ML SYSTEM — MODEL CHAT / INFERENCE   ║")))
    print(bold(cyan("  ╚══════════════════════════════════════════╝")))
    print()

    # Find and load model
    try:
        pt_path = _find_model(model_name)
    except FileNotFoundError as e:
        print(yellow(f"  {e}"))
        return

    print(dim(f"  Loading {pt_path.name}…"))
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    meta = _load_meta(pt_path)
    task = _detect_task(ckpt, meta)

    print(dim(f"  Task detected: {task}"))

    # Route to correct session
    if task == "text_generation":
        session = TextGenSession(pt_path, ckpt, meta)
    elif task == "cybersecurity":
        session = CybersecuritySession(pt_path, ckpt, meta)
    elif task == "image_classification":
        session = ImageSession(pt_path, ckpt, meta)
    else:
        session = ClassifierSession(pt_path, ckpt, meta)

    session.run()


import os  # needed for ANSI enable
