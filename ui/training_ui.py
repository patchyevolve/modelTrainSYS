"""
Training System Visual Interface
Hierarchical Mamba + Transformer ML System
Drag-and-drop data loading, live training metrics, model manager
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import math
import os
import time
import random
from pathlib import Path
from datetime import datetime

# ─── Model type registry — what each option does ─────────────────────────────
MODEL_REGISTRY = {
    "Hierarchical Mamba": {
        "desc": (
            "Multi-scale Mamba encoder → classifier head.\n"
            "Best for: tabular/CSV data, cybersecurity logs.\n"
            "Output: binary or multi-class label (a number, not text)."
        ),
        "input_dim_auto": True,
        "task": "binary_classification",
    },
    "Transformer Only": {
        "desc": (
            "Residual MLP with skip connections.\n"
            "Best for: clean tabular data, fast baseline.\n"
            "Output: binary or multi-class label."
        ),
        "input_dim_auto": True,
        "task": "binary_classification",
    },
    "Mamba+Transformer": {
        "desc": (
            "Hierarchical Mamba with deeper fusion layers.\n"
            "Best for: large tabular datasets (10k+ rows).\n"
            "Output: binary or multi-class label."
        ),
        "input_dim_auto": True,
        "task": "binary_classification",
    },
    "Cybersecurity": {
        "desc": (
            "Adversarial trainer for attack detection.\n"
            "Trains on: SQL injection, XSS, DDoS, malware, zero-day.\n"
            "Best for: cybersecurity_intrusion_data.csv\n"
            "Output: attack probability (0.0–1.0)."
        ),
        "input_dim_auto": True,
        "task": "binary_classification",
    },
    "Image Classification": {
        "desc": (
            "HMT Vision Transformer — patch-based image classifier.\n"
            "Train on: folders of images (one subfolder per class).\n"
            "  root/cats/img1.jpg  root/dogs/img2.jpg  ...\n"
            "Or flat folder (single class). Supports JPG/PNG/BMP/TIFF.\n"
            "Output: class label + confidence."
        ),
        "input_dim_auto": False,
        "task": "image_classification",
    },
    "Text Generation": {
        "desc": (
            "Hierarchical Mamba Language Model — learns to generate text.\n"
            "Train on: .txt, .csv, .json, .jsonl files OR HuggingFace datasets.\n"
            "Enter dataset name (e.g., wikitext, ianncity/General-Distillation-Prompts-1M)\n"
            "to train on online datasets. Enable reasoning for logical tasks.\n"
            "After training: type a prompt → model continues writing."
        ),
        "input_dim_auto": False,
        "task": "language_model",
    },
}

# ─── Color Palette ────────────────────────────────────────────────────────────
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_CARD    = "#1c2128"
BG_INPUT   = "#21262d"
ACCENT     = "#238636"
ACCENT_HOV = "#2ea043"
ACCENT2    = "#1f6feb"
BORDER     = "#30363d"
TEXT_PRI   = "#e6edf3"
TEXT_SEC   = "#8b949e"
TEXT_WARN  = "#d29922"
TEXT_ERR   = "#f85149"
TEXT_OK    = "#3fb950"
DRAG_OVER  = "#1f3a5f"

SUPPORTED_EXTS = {
    "Images":  [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"],
    "Text":    [".txt", ".csv", ".json", ".jsonl", ".xml"],
    "Stats":   [".csv", ".npy", ".npz", ".parquet"],
    "Archive": [".zip", ".tar", ".gz"],
}
ALL_EXTS = [e for exts in SUPPORTED_EXTS.values() for e in exts]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def styled_frame(parent, bg=BG_PANEL, bd=1, **kw):
    f = tk.Frame(parent, bg=bg, highlightbackground=BORDER,
                 highlightthickness=bd, **kw)
    return f

def label(parent, text, fg=TEXT_PRI, bg=BG_PANEL, font=("Segoe UI", 10), **kw):
    return tk.Label(parent, text=text, fg=fg, bg=bg, font=font, **kw)

def section_title(parent, text, bg=BG_PANEL):
    return label(parent, text, fg=TEXT_SEC, bg=bg,
                 font=("Segoe UI", 8, "bold"))

def accent_btn(parent, text, cmd, color=ACCENT, width=14):
    btn = tk.Button(parent, text=text, command=cmd,
                    bg=color, fg=TEXT_PRI, activebackground=ACCENT_HOV,
                    activeforeground=TEXT_PRI, relief="flat", cursor="hand2",
                    font=("Segoe UI", 9, "bold"), width=width, pady=5)
    btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT_HOV))
    btn.bind("<Leave>", lambda e: btn.config(bg=color))
    return btn

def ghost_btn(parent, text, cmd, width=12):
    btn = tk.Button(parent, text=text, command=cmd,
                    bg=BG_INPUT, fg=TEXT_SEC, activebackground=BORDER,
                    activeforeground=TEXT_PRI, relief="flat", cursor="hand2",
                    font=("Segoe UI", 9), width=width, pady=4)
    return btn

def separator(parent, bg=BG_PANEL):
    return tk.Frame(parent, bg=BORDER, height=1)

def fmt_size(b):
    for u in ["B","KB","MB","GB"]:
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} TB"

# ─── Drag-and-Drop Zone ───────────────────────────────────────────────────────

class DropZone(tk.Frame):
    def __init__(self, parent, on_files_added, **kw):
        super().__init__(parent, bg=BG_INPUT, highlightbackground=BORDER,
                         highlightthickness=1, **kw)
        self.on_files_added = on_files_added
        self._build()
        self._try_dnd()

    def _build(self):
        self.icon_lbl = tk.Label(self, text="⬆", font=("Segoe UI", 28),
                                 fg=TEXT_SEC, bg=BG_INPUT)
        self.icon_lbl.pack(pady=(18, 4))
        self.main_lbl = tk.Label(self, text="Drop files or folders here",
                                 font=("Segoe UI", 11, "bold"),
                                 fg=TEXT_PRI, bg=BG_INPUT)
        self.main_lbl.pack()
        self.sub_lbl = tk.Label(self,
            text="Images · Text · CSV · JSON · NPY · ZIP",
            font=("Segoe UI", 8), fg=TEXT_SEC, bg=BG_INPUT)
        self.sub_lbl.pack(pady=(2, 8))
        self.browse_btn = ghost_btn(self, "Browse Files", self._browse, width=16)
        self.browse_btn.pack(pady=(0, 6))
        self.browse_dir_btn = ghost_btn(self, "Browse Folder", self._browse_dir, width=16)
        self.browse_dir_btn.pack(pady=(0, 14))

    def _try_dnd(self):
        try:
            self.drop_target_register("DND_Files")
            self.dnd_bind("<<Drop>>", self._on_drop)
            self.dnd_bind("<<DragEnter>>", lambda e: self._highlight(True))
            self.dnd_bind("<<DragLeave>>", lambda e: self._highlight(False))
        except Exception:
            pass  # tkinterdnd2 not installed – browse-only mode

    def _highlight(self, on):
        self.config(bg=DRAG_OVER if on else BG_INPUT)
        for w in (self.icon_lbl, self.main_lbl, self.sub_lbl):
            w.config(bg=DRAG_OVER if on else BG_INPUT)

    def _on_drop(self, event):
        self._highlight(False)
        paths = self.tk.splitlist(event.data)
        self.on_files_added(list(paths))

    def _browse(self):
        paths = filedialog.askopenfilenames(
            title="Select training files",
            filetypes=[("Supported", " ".join(f"*{e}" for e in ALL_EXTS)),
                       ("All files", "*.*")])
        if paths:
            self.on_files_added(list(paths))

    def _browse_dir(self):
        d = filedialog.askdirectory(title="Select training folder")
        if d:
            self.on_files_added([d])

# ─── Data Panel (left column) ─────────────────────────────────────────────────

class DataPanel(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG_PANEL, **kw)
        self.files = []
        self._build()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=BG_PANEL)
        hdr.pack(fill="x", padx=12, pady=(12, 6))
        label(hdr, "DATA", fg=TEXT_SEC, bg=BG_PANEL,
              font=("Segoe UI", 8, "bold")).pack(side="left")
        self.count_lbl = label(hdr, "0 files", fg=ACCENT2, bg=BG_PANEL,
                               font=("Segoe UI", 8))
        self.count_lbl.pack(side="right")

        # Drop zone
        self.drop = DropZone(self, self._add_files)
        self.drop.pack(fill="x", padx=12, pady=(0, 8))

        # Type breakdown
        type_frame = styled_frame(self, bg=BG_CARD)
        type_frame.pack(fill="x", padx=12, pady=(0, 8))
        section_title(type_frame, "  TYPE BREAKDOWN", bg=BG_CARD).pack(
            anchor="w", pady=(6, 4))
        self.type_vars = {}
        for dtype in ["Images", "Text", "Stats", "Archive", "Other"]:
            row = tk.Frame(type_frame, bg=BG_CARD)
            row.pack(fill="x", padx=8, pady=1)
            dot_color = {
                "Images": "#58a6ff", "Text": "#3fb950",
                "Stats": "#d29922", "Archive": "#bc8cff", "Other": TEXT_SEC
            }[dtype]
            tk.Label(row, text="●", fg=dot_color, bg=BG_CARD,
                     font=("Segoe UI", 8)).pack(side="left")
            tk.Label(row, text=f" {dtype}", fg=TEXT_PRI, bg=BG_CARD,
                     font=("Segoe UI", 9)).pack(side="left")
            v = tk.StringVar(value="0")
            self.type_vars[dtype] = v
            tk.Label(row, textvariable=v, fg=TEXT_SEC, bg=BG_CARD,
                     font=("Segoe UI", 9)).pack(side="right", padx=6)
        tk.Frame(type_frame, bg=BG_CARD, height=6).pack()

        # File list
        list_hdr = tk.Frame(self, bg=BG_PANEL)
        list_hdr.pack(fill="x", padx=12, pady=(0, 4))
        section_title(list_hdr, "FILES", bg=BG_PANEL).pack(side="left")
        ghost_btn(list_hdr, "Clear", self._clear, width=6).pack(side="right")

        list_frame = styled_frame(self, bg=BG_CARD)
        list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.file_list = tk.Listbox(list_frame, bg=BG_CARD, fg=TEXT_PRI,
                                    selectbackground=ACCENT2, relief="flat",
                                    font=("Segoe UI", 8), borderwidth=0,
                                    highlightthickness=0, activestyle="none")
        sb = ttk.Scrollbar(list_frame, orient="vertical",
                           command=self.file_list.yview)
        self.file_list.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.file_list.pack(fill="both", expand=True, padx=4, pady=4)

    def _add_files(self, paths):
        added = 0
        for p in paths:
            p = Path(p)
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and f.suffix.lower() in ALL_EXTS:
                        if str(f) not in self.files:
                            self.files.append(str(f))
                            added += 1
            elif p.is_file():
                if str(p) not in self.files:
                    self.files.append(str(p))
                    added += 1
        self._refresh()
        return added

    def _refresh(self):
        self.file_list.delete(0, "end")
        counts = {k: 0 for k in self.type_vars}
        for f in self.files:
            ext = Path(f).suffix.lower()
            name = Path(f).name
            if ext in SUPPORTED_EXTS["Images"]:
                counts["Images"] += 1; tag = "🖼"
            elif ext in SUPPORTED_EXTS["Text"]:
                counts["Text"] += 1; tag = "📄"
            elif ext in SUPPORTED_EXTS["Stats"]:
                counts["Stats"] += 1; tag = "📊"
            elif ext in SUPPORTED_EXTS["Archive"]:
                counts["Archive"] += 1; tag = "📦"
            else:
                counts["Other"] += 1; tag = "📁"
            self.file_list.insert("end", f"  {tag} {name}")
        for k, v in self.type_vars.items():
            v.set(str(counts[k]))
        self.count_lbl.config(text=f"{len(self.files)} files")

    def _clear(self):
        self.files.clear()
        self._refresh()

    def get_files(self):
        return self.files

# ─── Mini Canvas Chart ────────────────────────────────────────────────────────

class LineChart(tk.Canvas):
    def __init__(self, parent, label="Loss", color=TEXT_ERR, **kw):
        super().__init__(parent, bg=BG_CARD, highlightthickness=0, **kw)
        self.label = label
        self.color = color
        self.data = []
        self.bind("<Configure>", lambda e: self._draw())

    def push(self, val):
        self.data.append(val)
        if len(self.data) > 200:
            self.data = self.data[-200:]
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10 or not self.data:
            return
        pad = 6
        mn, mx = min(self.data), max(self.data)
        rng = mx - mn or 1
        pts = []
        for i, v in enumerate(self.data):
            x = pad + (i / max(len(self.data)-1, 1)) * (w - 2*pad)
            y = h - pad - ((v - mn) / rng) * (h - 2*pad)
            pts.append((x, y))
        # Grid lines
        for frac in [0.25, 0.5, 0.75]:
            yg = pad + frac * (h - 2*pad)
            self.create_line(pad, yg, w-pad, yg, fill=BORDER, dash=(2,4))
        # Line
        if len(pts) > 1:
            flat = [c for p in pts for c in p]
            self.create_line(*flat, fill=self.color, width=2, smooth=True)
        # Last value dot
        if pts:
            lx, ly = pts[-1]
            self.create_oval(lx-3, ly-3, lx+3, ly+3, fill=self.color, outline="")
        # Label + last value
        last = f"{self.data[-1]:.4f}" if self.data else ""
        self.create_text(pad+2, pad+2, text=self.label, fill=TEXT_SEC,
                         font=("Segoe UI", 7), anchor="nw")
        self.create_text(w-pad-2, pad+2, text=last, fill=self.color,
                         font=("Segoe UI", 7, "bold"), anchor="ne")

# ─── Training Config + Status Panel (center) ─────────────────────────────────

class TrainingPanel(tk.Frame):
    def __init__(self, parent, get_files_cb, **kw):
        super().__init__(parent, bg=BG_PANEL, **kw)
        self.get_files = get_files_cb
        self.running = False
        self._thread = None
        self._stop_flag = threading.Event()
        self.log_lines = []
        self._build()

    def _ui(self, fn, *args):
        """Thread-safe UI update helper."""
        try:
            if self.winfo_exists():
                self.after(0, fn, *args)
        except Exception:
            pass

    def _build(self):
        # ── Config section ──
        cfg_frame = styled_frame(self, bg=BG_CARD)
        cfg_frame.pack(fill="x", padx=12, pady=(12, 6))

        section_title(cfg_frame, "  MODEL CONFIGURATION", bg=BG_CARD).pack(
            anchor="w", pady=(8, 6), padx=8)

        params = [
            ("Model Type",    list(MODEL_REGISTRY.keys())),
            ("Optimizer",     ["Adam", "AdamW", "SGD"]),
            ("Scheduler",     ["CosineAnnealing", "StepLR", "None"]),
        ]
        self.combos = {}
        for lbl_text, opts in params:
            row = tk.Frame(cfg_frame, bg=BG_CARD)
            row.pack(fill="x", padx=8, pady=3)
            tk.Label(row, text=lbl_text, fg=TEXT_SEC, bg=BG_CARD,
                     font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
            cb = ttk.Combobox(row, values=opts, state="readonly",
                              font=("Segoe UI", 9), width=20)
            cb.set(opts[0])
            cb.pack(side="left", padx=4)
            self.combos[lbl_text] = cb
            # Info button for Model Type
            if lbl_text == "Model Type":
                info_btn = tk.Button(row, text="?", bg=ACCENT2, fg=TEXT_PRI,
                                     relief="flat", cursor="hand2",
                                     font=("Segoe UI", 8, "bold"), width=2,
                                     command=self._show_model_info)
                info_btn.pack(side="left", padx=2)
                cb.bind("<<ComboboxSelected>>", lambda e: self._update_model_desc())

        spinners = [
            ("Epochs",      "epochs",    20,   1,   500),
            ("Batch Size",  "batch",     64,   1,   512),
            ("Hidden Dim",  "hidden",    256,  64,  2048),
            ("Num Layers",  "layers",    4,    1,   12),
            ("Num Heads",   "heads",     12,   1,   32),
        ]
        self.spins = {}
        for lbl_text, key, default, lo, hi in spinners:
            row = tk.Frame(cfg_frame, bg=BG_CARD)
            row.pack(fill="x", padx=8, pady=3)
            tk.Label(row, text=lbl_text, fg=TEXT_SEC, bg=BG_CARD,
                     font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
            sv = tk.StringVar(value=str(default))
            sp = tk.Spinbox(row, from_=lo, to=hi, textvariable=sv,
                            bg=BG_INPUT, fg=TEXT_PRI, insertbackground=TEXT_PRI,
                            relief="flat", font=("Segoe UI", 9), width=8,
                            buttonbackground=BG_INPUT)
            sp.pack(side="left", padx=4)
            self.spins[key] = sv

        lr_row = tk.Frame(cfg_frame, bg=BG_CARD)
        lr_row.pack(fill="x", padx=8, pady=3)
        tk.Label(lr_row, text="Learning Rate", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        self.lr_var = tk.StringVar(value="0.001")
        tk.Entry(lr_row, textvariable=self.lr_var, bg=BG_INPUT, fg=TEXT_PRI,
                 insertbackground=TEXT_PRI, relief="flat",
                 font=("Segoe UI", 9), width=10).pack(side="left", padx=4)

        # Reflector toggle
        ref_row = tk.Frame(cfg_frame, bg=BG_CARD)
        ref_row.pack(fill="x", padx=8, pady=1)
        tk.Label(ref_row, text="Reflector", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        self.reflector_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ref_row, variable=self.reflector_var, bg=BG_CARD,
                       fg=TEXT_PRI, selectcolor=BG_INPUT,
                       activebackground=BG_CARD).pack(side="left")
        tk.Label(ref_row, text="Enable auto-correction", fg=TEXT_SEC,
                 bg=BG_CARD, font=("Segoe UI", 8)).pack(side="left")

        # Reasoning Mode toggle (new)
        reason_row = tk.Frame(cfg_frame, bg=BG_CARD)
        reason_row.pack(fill="x", padx=8, pady=(1, 4))
        tk.Label(reason_row, text="Reasoning Mode", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        self.reasoning_var = tk.BooleanVar(value=False)
        tk.Checkbutton(reason_row, variable=self.reasoning_var, bg=BG_CARD,
                       fg=TEXT_PRI, selectcolor=BG_INPUT,
                       activebackground=BG_CARD).pack(side="left")
        tk.Label(reason_row, text="Train on logic only (Text Gen)", fg=TEXT_SEC,
                 bg=BG_CARD, font=("Segoe UI", 8)).pack(side="left")

        # Seq Len (context length) - critical for text generation
        seq_row = tk.Frame(cfg_frame, bg=BG_CARD)
        seq_row.pack(fill="x", padx=8, pady=(1, 4))
        tk.Label(seq_row, text="Seq Length", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        self.seq_len_var = tk.StringVar(value="128")
        tk.Entry(seq_row, textvariable=self.seq_len_var, bg=BG_INPUT, fg=TEXT_PRI,
                 insertbackground=TEXT_PRI, relief="flat",
                 font=("Segoe UI", 9), width=6).pack(side="left", padx=4)
        tk.Label(seq_row, text=" chars context (0=auto)", fg=TEXT_SEC,
                 bg=BG_CARD, font=("Segoe UI", 8)).pack(side="left")

        # Dataset Name (for HuggingFace datasets)
        ds_row = tk.Frame(cfg_frame, bg=BG_CARD)
        ds_row.pack(fill="x", padx=8, pady=(1, 4))
        tk.Label(ds_row, text="Dataset Name", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        self.dataset_var = tk.StringVar(value="")
        tk.Entry(ds_row, textvariable=self.dataset_var, bg=BG_INPUT, fg=TEXT_PRI,
                 insertbackground=TEXT_PRI, relief="flat",
                 font=("Segoe UI", 9), width=28).pack(side="left", padx=4)
        tk.Label(ds_row, text=" (e.g. wikitext, ianncity/...)", fg=TEXT_SEC,
                 bg=BG_CARD, font=("Segoe UI", 8)).pack(side="left")

        # Reasoning Settings (for Text Generation)
        tk.Label(cfg_frame, text="── Text/Reasoning Settings ──", fg=ACCENT2, bg=BG_CARD,
                 font=("Segoe UI", 8, "bold")).pack(fill="x", padx=8, pady=(8, 4))
        
        # Reasoning Weight
        reason_weight_row = tk.Frame(cfg_frame, bg=BG_CARD)
        reason_weight_row.pack(fill="x", padx=8, pady=(1, 4))
        tk.Label(reason_weight_row, text="Reasoning Weight", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        self.reasoning_weight_var = tk.StringVar(value="3.0")
        tk.Entry(reason_weight_row, textvariable=self.reasoning_weight_var, bg=BG_INPUT, fg=TEXT_PRI,
                 insertbackground=TEXT_PRI, relief="flat",
                 font=("Segoe UI", 9), width=6).pack(side="left", padx=4)
        tk.Label(reason_weight_row, text=" (boost logic tokens)", fg=TEXT_SEC,
                 bg=BG_CARD, font=("Segoe UI", 8)).pack(side="left")
        
        # Logic Weight
        logic_weight_row = tk.Frame(cfg_frame, bg=BG_CARD)
        logic_weight_row.pack(fill="x", padx=8, pady=(1, 4))
        tk.Label(logic_weight_row, text="Logic Weight", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        self.logic_weight_var = tk.StringVar(value="2.5")
        tk.Entry(logic_weight_row, textvariable=self.logic_weight_var, bg=BG_INPUT, fg=TEXT_PRI,
                 insertbackground=TEXT_PRI, relief="flat",
                 font=("Segoe UI", 9), width=6).pack(side="left", padx=4)
        tk.Label(logic_weight_row, text=" (connectors)", fg=TEXT_SEC,
                 bg=BG_CARD, font=("Segoe UI", 8)).pack(side="left")
        
        # Curriculum Learning toggle
        curriculum_row = tk.Frame(cfg_frame, bg=BG_CARD)
        curriculum_row.pack(fill="x", padx=8, pady=(1, 4))
        tk.Label(curriculum_row, text="Curriculum", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        self.curriculum_var = tk.BooleanVar(value=True)
        tk.Checkbutton(curriculum_row, variable=self.curriculum_var, bg=BG_CARD,
                       fg=TEXT_PRI, selectcolor=BG_INPUT,
                       activebackground=BG_CARD).pack(side="left")
        tk.Label(curriculum_row, text="(simple→complex)", fg=TEXT_SEC,
                 bg=BG_CARD, font=("Segoe UI", 8)).pack(side="left")
        
        # Focal Loss toggle
        focal_row = tk.Frame(cfg_frame, bg=BG_CARD)
        focal_row.pack(fill="x", padx=8, pady=(1, 4))
        tk.Label(focal_row, text="Focal Loss", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        self.focal_loss_var = tk.BooleanVar(value=False)
        tk.Checkbutton(focal_row, variable=self.focal_loss_var, bg=BG_CARD,
                       fg=TEXT_PRI, selectcolor=BG_INPUT,
                       activebackground=BG_CARD).pack(side="left")
        tk.Label(focal_row, text="(hard example mining)", fg=TEXT_SEC,
                 bg=BG_CARD, font=("Segoe UI", 8)).pack(side="left")

        # Model description box
        self.model_desc_lbl = tk.Label(
            cfg_frame, text="", fg=TEXT_SEC, bg=BG_CARD,
            font=("Segoe UI", 8), justify="left", anchor="w",
            wraplength=340, padx=8, pady=4)
        self.model_desc_lbl.pack(fill="x", padx=8, pady=(0, 8))
        self._update_model_desc()

        # ── Action buttons ──
        btn_row = tk.Frame(self, bg=BG_PANEL)
        btn_row.pack(fill="x", padx=12, pady=(0, 8))
        self.start_btn = accent_btn(btn_row, "▶  Start Training",
                                    self._start, width=18)
        self.start_btn.pack(side="left", padx=(0, 6))
        self.stop_btn = accent_btn(btn_row, "■  Stop",
                                   self._stop, color="#b91c1c", width=10)
        self.stop_btn.pack(side="left")
        self.stop_btn.config(state="disabled")
        ghost_btn(btn_row, "Save Config", self._save_config, width=12).pack(
            side="right")

        # ── Status cards ──
        status_row = tk.Frame(self, bg=BG_PANEL)
        status_row.pack(fill="x", padx=12, pady=(0, 8))
        self.stat_vars = {}
        cards = [
            ("Epoch",    "epoch",    "—"),
            ("Loss",     "loss",     "—"),
            ("Accuracy", "acc",      "—"),
            ("LR",       "lr",       "—"),
            ("ETA",      "eta",      "—"),
            ("Reflector","refl",     "—"),
        ]
        for i, (title, key, init) in enumerate(cards):
            card = styled_frame(status_row, bg=BG_CARD)
            card.grid(row=0, column=i, padx=3, sticky="ew")
            status_row.columnconfigure(i, weight=1)
            tk.Label(card, text=title, fg=TEXT_SEC, bg=BG_CARD,
                     font=("Segoe UI", 7, "bold")).pack(pady=(6, 0))
            v = tk.StringVar(value=init)
            self.stat_vars[key] = v
            tk.Label(card, textvariable=v, fg=TEXT_PRI, bg=BG_CARD,
                     font=("Segoe UI", 11, "bold")).pack(pady=(0, 6))

        # ── Progress bar ──
        prog_frame = tk.Frame(self, bg=BG_PANEL)
        prog_frame.pack(fill="x", padx=12, pady=(0, 6))
        self.progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Green.Horizontal.TProgressbar",
                        troughcolor=BG_INPUT, background=ACCENT,
                        bordercolor=BG_INPUT, lightcolor=ACCENT,
                        darkcolor=ACCENT)
        self.pbar = ttk.Progressbar(prog_frame, variable=self.progress_var,
                                    maximum=100, length=400,
                                    style="Green.Horizontal.TProgressbar")
        self.pbar.pack(fill="x")
        self.prog_lbl = label(prog_frame, "Ready", fg=TEXT_SEC, bg=BG_PANEL,
                              font=("Segoe UI", 8))
        self.prog_lbl.pack(anchor="e")

        # ── Charts ──
        chart_row = tk.Frame(self, bg=BG_PANEL)
        chart_row.pack(fill="x", padx=12, pady=(0, 6))
        self.loss_chart = LineChart(chart_row, "Train Loss", TEXT_ERR,
                                    height=90)
        self.loss_chart.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.acc_chart = LineChart(chart_row, "Accuracy", TEXT_OK, height=90)
        self.acc_chart.pack(side="left", fill="x", expand=True)

        # ── Log ──
        log_hdr = tk.Frame(self, bg=BG_PANEL)
        log_hdr.pack(fill="x", padx=12, pady=(0, 4))
        section_title(log_hdr, "TRAINING LOG", bg=BG_PANEL).pack(side="left")
        ghost_btn(log_hdr, "Clear", self._clear_log, width=6).pack(side="right")

        log_frame = styled_frame(self, bg=BG_CARD)
        log_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.log_text = tk.Text(log_frame, bg=BG_CARD, fg=TEXT_PRI,
                                font=("Consolas", 8), relief="flat",
                                state="disabled", wrap="word",
                                insertbackground=TEXT_PRI)
        log_sb = ttk.Scrollbar(log_frame, orient="vertical",
                               command=self.log_text.yview)
        self.log_text.config(yscrollcommand=log_sb.set)
        log_sb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_text.tag_config("ok",   foreground=TEXT_OK)
        self.log_text.tag_config("warn", foreground=TEXT_WARN)
        self.log_text.tag_config("err",  foreground=TEXT_ERR)
        self.log_text.tag_config("info", foreground=ACCENT2)

    # ── Model info helpers ────────────────────────────────────────────────────
    def _update_model_desc(self, _=None):
        mtype = self.combos["Model Type"].get()
        info  = MODEL_REGISTRY.get(mtype, {})
        desc  = info.get("desc", "")
        # Show first line only in the inline label
        first_line = desc.split("\n")[0] if desc else ""
        self.model_desc_lbl.config(text=f"ℹ  {first_line}")

    def _show_model_info(self):
        mtype = self.combos["Model Type"].get()
        info  = MODEL_REGISTRY.get(mtype, {})
        desc  = info.get("desc", "No description available.")
        win = tk.Toplevel(self)
        win.title(f"Model: {mtype}")
        win.configure(bg=BG_DARK)
        win.geometry("420x220")
        win.resizable(False, False)
        tk.Label(win, text=mtype, fg=TEXT_PRI, bg=BG_DARK,
                 font=("Segoe UI", 12, "bold")).pack(pady=(16, 8), padx=20, anchor="w")
        tk.Label(win, text=desc, fg=TEXT_SEC, bg=BG_DARK,
                 font=("Segoe UI", 9), justify="left", wraplength=380,
                 anchor="w").pack(padx=20, anchor="w")
        tk.Button(win, text="Close", command=win.destroy,
                  bg=BG_INPUT, fg=TEXT_SEC, relief="flat",
                  font=("Segoe UI", 9), pady=4, width=10).pack(pady=16)

    # ── Log helpers ──────────────────────────────────────────────────────────
    def _log(self, msg, tag=""):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_text.config(state="normal")
        self.log_text.insert("end", line, tag)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    # ── Config save ──────────────────────────────────────────────────────────
    def _save_config(self):
        cfg = self._get_config()
        path = filedialog.asksaveasfilename(defaultextension=".json",
            filetypes=[("JSON", "*.json")])
        if path:
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
            self._log(f"Config saved → {path}", "ok")

    def _get_config(self):
        seq_len_val = int(self.seq_len_var.get())
        return {
            "model_type":     self.combos["Model Type"].get(),
            "optimizer":      self.combos["Optimizer"].get(),
            "scheduler":      self.combos["Scheduler"].get(),
            "epochs":         int(self.spins["epochs"].get()),
            "batch_size":     int(self.spins["batch"].get()),
            "hidden_dim":     int(self.spins["hidden"].get()),
            "num_layers":     int(self.spins["layers"].get()),
            "num_heads":      int(self.spins["heads"].get()),
            "lr":             float(self.lr_var.get()),
            "seq_len":        seq_len_val if seq_len_val > 0 else 0,
            "reflector":      self.reflector_var.get(),
            "reasoning_only": self.reasoning_var.get(),
            "dataset_name":   self.dataset_var.get().strip(),
            "reasoning_weight": float(self.reasoning_weight_var.get()),
            "logic_weight":   float(self.logic_weight_var.get()),
            "curriculum":      self.curriculum_var.get(),
            "focal_loss":     self.focal_loss_var.get(),
        }

    # ── Training simulation / real run ───────────────────────────────────────
    def _start(self):
        files = self.get_files()
        cfg = self._get_config()
        task = MODEL_REGISTRY.get(cfg["model_type"], {}).get("task", "binary_classification")
        
        # Allow dataset or reasoning training without local files
        if task not in ("hf_dataset", "reasoning") and not files:
            messagebox.showwarning("No Data",
                "Please add training files before starting.")
            return
        cfg = self._get_config()
        self._stop_flag.clear()
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._log(f"Starting training — {cfg['model_type']} | "
                  f"{len(files)} files | {cfg['epochs']} epochs", "info")
        self._thread = threading.Thread(target=self._run_training,
                                        args=(cfg, files), daemon=True)
        self._thread.start()

    def _stop(self):
        self._stop_flag.set()
        self._log("Stop requested…", "warn")

    def _run_training(self, cfg, files):
        """Real training loop — branches on model task type."""
        import torch, os
        # Set PyTorch to use all available CPU cores for matrix ops
        n_cores = os.cpu_count() or 4
        try:
            torch.set_num_threads(n_cores)
            torch.set_num_interop_threads(max(2, n_cores // 2))
        except RuntimeError:
            pass  # Threads already set

        task = MODEL_REGISTRY.get(cfg["model_type"], {}).get("task", "binary_classification")
        if task == "language_model":
            self._run_lm_training(cfg, files)
        elif task == "image_classification":
            self._run_image_training(cfg, files)
        else:
            self._run_classifier_training(cfg, files)
    def _run_lm_training(self, cfg, files):
        """Language model training — next-token prediction on text files.
        Supports: local files (txt, csv, json) and HuggingFace datasets.
        Optional: reasoning-aware training with token boosting."""
        import torch
        import torch.optim as optim

        epochs     = cfg["epochs"]
        batch_size = cfg["batch_size"]
        lr         = cfg["lr"]
        start_t    = time.time()

        # Reasoning settings (apply to all text training)
        reasoning_weight = cfg.get("reasoning_weight", 1.0)
        use_reasoning   = reasoning_weight > 1.0
        use_curriculum  = cfg.get("curriculum", False)
        use_focal       = cfg.get("focal_loss", False)

        # Check for HuggingFace dataset
        dataset_name = cfg.get("dataset_name", "").strip()
        
        # Auto-size seq_len
        user_seq_len = cfg.get("seq_len", 0)
        seq_len = user_seq_len if user_seq_len > 0 else 128

        # ── 1. Load text data ─────────────────────────────────────────────────
        try:
            if dataset_name:
                # HuggingFace dataset
                self._ui(self._log, f"Loading HuggingFace dataset: {dataset_name}...", "info")
                try:
                    from data.hf_dataset_loader import build_hf_loaders
                    train_loader, val_loader, tokenizer, info = build_hf_loaders(
                        dataset_name, seq_len=seq_len, batch_size=batch_size, val_split=0.1
                    )
                    self._ui(self._log,
                        f"Dataset: {dataset_name} | "
                        f"vocab={info['vocab_size']} | "
                        f"{info['train_batches']} batches/epoch", "ok")
                except ImportError:
                    self._ui(self._log, "datasets library not installed. Run: pip install datasets", "err")
                    self._ui(self._finish_training)
                    return
                except Exception as e:
                    self._ui(self._log, f"Dataset loading failed: {e}", "err")
                    self._ui(self._finish_training)
                    return
                    
            elif files:
                # Local files (txt, csv, json, jsonl)
                from data.text_dataset import build_text_loaders, read_text_files
                
                corpus_size = sum(
                    Path(f).stat().st_size for f in files if Path(f).exists())
                
                if user_seq_len == 0:
                    if corpus_size > 5_000_000:
                        seq_len = 128
                        batch_size = min(batch_size, 64)
                    elif corpus_size > 1_000_000:
                        seq_len = 96
                        batch_size = min(batch_size, 48)
                    elif corpus_size > 200_000:
                        seq_len = 64

                self._ui(self._log, "Loading text corpus…", "info")
                train_loader, val_loader, tokenizer, info = build_text_loaders(
                    files, seq_len=seq_len, batch_size=batch_size)
                
                from data.prefetch_loader import PrefetchLoader
                train_loader = PrefetchLoader(train_loader, buffer_size=3)
                
                self._ui(self._log,
                    f"Corpus: {info['corpus_chars']:,} chars | "
                    f"vocab={info['vocab_size']} | seq_len={seq_len} | "
                    f"{info['train_batches']} batches/epoch", "ok")
            else:
                self._ui(self._log, "No data provided. Provide files or dataset name.", "err")
                self._ui(self._finish_training)
                return

            n_batches = info["train_batches"] if isinstance(info, dict) else len(train_loader)
            
            if use_reasoning:
                self._ui(self._log, f"Reasoning mode: weight={reasoning_weight}", "info")
            if use_curriculum:
                self._ui(self._log, "Curriculum learning: enabled", "info")

        except Exception as e:
            self._ui(self._log, f"Text data loading failed: {e}", "err")
            self._ui(self._finish_training)
            return

        # ── 2. Build language model ───────────────────────────────────────────
        from core.text_model import lm_train_step, lm_val_loss, save_lm
        from core.implementations import HMTLanguageModel
        hidden    = cfg["hidden_dim"]
        num_heads = max(1, min(8, hidden // 64))
        hidden    = (hidden // num_heads) * num_heads
        n_layers  = cfg["num_layers"]
        n_scales  = 3

        model = HMTLanguageModel(
            vocab_size = info["vocab_size"],
            dim        = hidden,
            num_layers = n_layers,
            num_heads  = num_heads,
            num_scales = n_scales,
            max_seq    = seq_len,
            dropout    = 0.1,
        )
        param_count = model.count_parameters()

        # ── Device selection ──────────────────────────────────────────────────
        from core.device_manager import get_best_device, move_batch
        device, device_name = get_best_device(
            model_params=param_count, batch_size=batch_size)
        model = model.to(device)
        self._ui(self._log,
            f"HMT-LM: {param_count:,} params | "
            f"vocab={info['vocab_size']} | dim={hidden} | "
            f"layers={n_layers} scales={n_scales}", "ok")
        self._ui(self._log, f"Device: {device_name}", "info")

        # ── 3. Optimizer ──────────────────────────────────────────────────────
        opt_name = cfg["optimizer"]
        if opt_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        elif opt_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)

        sched_name = cfg["scheduler"]
        if sched_name == "CosineAnnealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs)
        elif sched_name == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, epochs // 3), gamma=0.5)
        else:
            scheduler = None

        # ── 4. Setup reasoning-aware training if enabled ─────────────────────────
        loss_fn = None
        if use_reasoning:
            try:
                from training.reasoning_trainer import ReasoningAwareLoss, CurriculumScheduler
                loss_fn = ReasoningAwareLoss(
                    reasoning_weight=reasoning_weight,
                    logic_weight=cfg.get("logic_weight", 2.5),
                    use_focal=use_focal,
                )
                curriculum = CurriculumScheduler(total_steps=epochs * steps_per_epoch) if use_curriculum else None
                self._ui(self._log, f"Reasoning loss active | Curriculum: {use_curriculum}", "info")
            except ImportError:
                self._ui(self._log, "Reasoning trainer not available, using standard loss", "warn")
                use_reasoning = False

        # ── 5. Training loop ──────────────────────────────────────────────────
        best_val    = float("inf")
        best_state  = None
        total_steps = epochs * n_batches
        UI_INTERVAL = 2.0   # seconds between UI refreshes

        MAX_STEPS_PER_EPOCH = 2000
        steps_per_epoch = min(n_batches, MAX_STEPS_PER_EPOCH)
        if steps_per_epoch < n_batches:
            total_steps = epochs * steps_per_epoch
            self._ui(self._log,
                f"Capping at {steps_per_epoch:,} steps/epoch "
                f"(full epoch = {n_batches:,} batches).", "warn")

        last_ui_t = time.time()

        for epoch in range(1, epochs + 1):
            if self._stop_flag.is_set():
                self._ui(self._log, "Training stopped by user.", "warn")
                break

            epoch_loss = 0.0
            model.train()
            step = 0
            
            # Update curriculum if active
            if curriculum:
                cur_params = curriculum.step((epoch - 1) * steps_per_epoch)
                if loss_fn:
                    loss_fn.reasoning_weight = cur_params["reasoning_weight"]

            for xb, yb in train_loader:
                if self._stop_flag.is_set():
                    break
                if step >= steps_per_epoch:
                    break

                xb, yb = move_batch((xb, yb), device)
                
                # Use reasoning-aware loss or standard
                if loss_fn:
                    logits = model(xb)
                    # Create simple weights (1.0 for all tokens)
                    weights = torch.ones_like(yb, dtype=torch.float).to(device)
                    loss_val, metrics = loss_fn(logits, yb, weights)
                    loss_val = loss_val.item()
                    
                    if torch.isnan(loss_val) or torch.isinf(loss_val):
                        continue
                    
                    optimizer.zero_grad()
                    loss_val.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                else:
                    loss_val = lm_train_step(
                        model, optimizer, xb, yb,
                        reasoning_weight=reasoning_weight,
                        tokenizer=tokenizer
                    )
                
                epoch_loss += loss_val
                step += 1

                # ── Time-based UI update (every 2 seconds) ────────────────
                now = time.time()
                if now - last_ui_t >= UI_INTERVAL or step == steps_per_epoch:
                    last_ui_t  = now
                    avg        = epoch_loss / step
                    ppl        = min(math.exp(avg), 9999.0)
                    cur_lr     = optimizer.param_groups[0]["lr"]
                    elapsed    = now - start_t
                    done       = (epoch - 1) * steps_per_epoch + step
                    eta_s      = int((elapsed / done) * (total_steps - done))
                    eta_str    = f"{eta_s//60}m {eta_s%60}s"
                    pct        = (done / total_steps) * 100

                    stage_info = ""
                    if curriculum:
                        stage_info = f" [{curriculum.stages[curriculum.current_stage]['name']}]"

                    self._ui(self._update_stats,
                             epoch, epochs, avg,
                             max(0.0, 1.0 - avg / 5.0),
                             cur_lr, eta_str, 0.0, pct)

                    self._ui(self._log,
                        f"Ep {epoch}/{epochs}  "
                        f"step {step}/{steps_per_epoch}  "
                        f"loss={avg:.4f}  ppl={ppl:.1f}  "
                        f"eta={eta_str}", "info")

            if scheduler:
                scheduler.step()

            # ── End-of-epoch: validation + text sample ────────────────────
            avg_loss   = epoch_loss / max(step, 1)
            perplexity = min(math.exp(avg_loss), 9999.0)

            self._ui(self._log, "Running validation...", "info")
            val_l = lm_val_loss(model, val_loader, device=device, max_batches=50)
            val_p = min(math.exp(val_l), 9999.0)
            if val_l < best_val:
                best_val   = val_l
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}

            self._ui(self._log,
                f"── Epoch {epoch}/{epochs} done │ "
                f"loss={avg_loss:.4f}  val={val_l:.4f} │ "
                f"ppl={perplexity:.1f}  val_ppl={val_p:.1f}", "ok")

            sample = self._lm_sample(model, tokenizer, seq_len)
            if sample:
                self._ui(self._log, f"Sample › {sample[:120]}", "info")

        else:
            self._ui(self._log, "LM training complete.", "ok")
            # Save with tokenizer embedded
            lm_cfg = {
                "vocab_size":  info["vocab_size"],
                "dim":         hidden,
                "num_layers":  cfg["num_layers"],
                "num_heads":   num_heads,
                "num_scales":  3,
                "max_seq":     seq_len,
                "dropout":     0.1,
            }
            self._ui(self._save_lm_checkpoint, cfg, model, best_state,
                     tokenizer, lm_cfg, info)

        self._ui(self._finish_training)

    def _run_image_training(self, cfg, files):
        """Image classification training using HMTImageClassifier."""
        import torch
        import torch.nn as nn
        import torch.optim as optim

        epochs     = cfg["epochs"]
        batch_size = cfg["batch_size"]
        lr         = cfg["lr"]
        start_t    = time.time()

        # ── 1. Load images ────────────────────────────────────────────────────
        try:
            from data.image_dataset import build_image_loaders
            self._ui(self._log, "Scanning image folders…", "info")

            # Auto img_size from hidden_dim: bigger model → bigger patches ok
            hidden   = cfg["hidden_dim"]
            img_size = 32 if hidden <= 128 else 64 if hidden <= 256 else 96
            # Round to nearest multiple of patch_size (16)
            img_size = max(32, (img_size // 16) * 16)

            train_loader, val_loader, class_names, info = build_image_loaders(
                files, img_size=img_size, batch_size=batch_size)

            from data.prefetch_loader import PrefetchLoader
            train_loader = PrefetchLoader(train_loader, buffer_size=3)

            n_classes  = info["num_classes"]
            n_batches  = info["train_batches"]
            self._ui(self._log,
                f"Images: {info['total_images']:,} | "
                f"classes={n_classes} {class_names[:8]} | "
                f"size={img_size}×{img_size} | "
                f"{n_batches} batches/epoch", "ok")
        except Exception as e:
            self._ui(self._log, f"Image loading failed: {e}", "err")
            self._ui(self._finish_training)
            return

        # ── 2. Build model ────────────────────────────────────────────────────
        try:
            from core.implementations import HMTImageClassifier
            num_heads  = max(1, min(8, hidden // 64))
            hidden     = (hidden // num_heads) * num_heads
            patch_size = 8 if img_size <= 32 else 16

            model = HMTImageClassifier(
                num_classes  = n_classes,
                dim          = hidden,
                patch_size   = patch_size,
                img_channels = 3,
                num_layers   = cfg["num_layers"],
                num_heads    = num_heads,
                num_scales   = 3,
                dropout      = 0.1,
            )
            param_count = sum(p.numel() for p in model.parameters())
            self._ui(self._log,
                f"HMTImageClassifier: {param_count:,} params | "
                f"patch={patch_size}×{patch_size} | "
                f"dim={hidden} | {n_classes} classes", "ok")

            # ── Device selection ──────────────────────────────────────────
            from core.device_manager import get_best_device, move_batch
            device, device_name = get_best_device(
                model_params=param_count, batch_size=batch_size)
            model = model.to(device)
            self._ui(self._log, f"Device: {device_name}", "info")
        except Exception as e:
            self._ui(self._log, f"Model build failed: {e}", "err")
            self._ui(self._finish_training)
            return

        # ── 3. Optimizer + loss ───────────────────────────────────────────────
        opt_name = cfg["optimizer"]
        if opt_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        elif opt_name == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  momentum=0.9, weight_decay=1e-4)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)

        sched_name = cfg["scheduler"]
        if sched_name == "CosineAnnealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs)
        elif sched_name == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, epochs // 3), gamma=0.5)
        else:
            scheduler = None

        loss_fn = nn.CrossEntropyLoss() if n_classes > 2 else nn.BCEWithLogitsLoss()

        # ── 4. Training loop ──────────────────────────────────────────────────
        best_val_acc = 0.0
        best_state   = None
        total_steps  = epochs * n_batches
        UI_INTERVAL  = 2.0
        last_ui_t    = time.time()

        for epoch in range(1, epochs + 1):
            if self._stop_flag.is_set():
                self._ui(self._log, "Training stopped by user.", "warn")
                break

            model.train()
            epoch_loss = correct = total = 0

            for batch_idx, (xb, yb) in enumerate(train_loader, 1):
                if self._stop_flag.is_set():
                    break

                xb, yb = move_batch((xb, yb), device)
                optimizer.zero_grad()
                out = model(xb)                        # (B, n_classes)

                if n_classes > 2:
                    loss  = loss_fn(out, yb)
                    preds = out.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                else:
                    loss  = loss_fn(out.squeeze(1), yb.float())
                    preds = (torch.sigmoid(out.squeeze(1)) > 0.5).long()
                    correct += (preds == yb).sum().item()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                total      += yb.size(0)

                # ── Time-based UI update ──────────────────────────────────
                now = time.time()
                if now - last_ui_t >= UI_INTERVAL or batch_idx == n_batches:
                    last_ui_t  = now
                    avg        = epoch_loss / batch_idx
                    acc        = correct / max(total, 1)
                    cur_lr     = optimizer.param_groups[0]["lr"]
                    elapsed    = now - start_t
                    done       = (epoch - 1) * n_batches + batch_idx
                    eta_s      = int((elapsed / done) * (total_steps - done))
                    eta_str    = f"{eta_s//60}m {eta_s%60}s"
                    pct        = (done / total_steps) * 100

                    self._ui(self._update_stats, epoch, epochs, avg,
                             acc, cur_lr, eta_str, 0.0, pct)

            if scheduler:
                scheduler.step()

            # ── Validation ────────────────────────────────────────────────
            model.eval()
            val_loss = val_correct = val_total = 0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv, yv = move_batch((xv, yv), device)
                    ov = model(xv)
                    if n_classes > 2:
                        val_loss    += loss_fn(ov, yv).item()
                        val_correct += (ov.argmax(1) == yv).sum().item()
                    else:
                        val_loss    += loss_fn(ov.squeeze(1), yv.float()).item()
                        val_correct += (
                            (torch.sigmoid(ov.squeeze(1)) > 0.5).long() == yv
                        ).sum().item()
                    val_total += yv.size(0)

            val_loss /= max(len(val_loader), 1)
            val_acc   = val_correct / max(val_total, 1)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone()
                                for k, v in model.state_dict().items()}

            avg_loss = epoch_loss / n_batches
            acc      = correct / max(total, 1)
            self._ui(self._log,
                f"── Epoch {epoch}/{epochs} │ "
                f"loss={avg_loss:.4f}  val_loss={val_loss:.4f} │ "
                f"acc={acc*100:.1f}%  val_acc={val_acc*100:.1f}%", "ok")

        else:
            self._ui(self._log, "Image training complete.", "ok")
            self._ui(self._save_image_checkpoint, cfg, model,
                     best_state, class_names, info)

        self._ui(self._finish_training)

    def _save_image_checkpoint(self, cfg, model, best_state,
                                class_names, data_info):
        import torch
        save_dir = Path("trained_models")
        save_dir.mkdir(exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"ImageClassifier_{ts}"
        pt_path = save_dir / f"{name}.pt"

        if best_state:
            model.load_state_dict(best_state)

        torch.save({
            "model_state_dict": model.state_dict(),
            "config":           cfg,
            "data_info":        data_info,
            "class_names":      class_names,
            "model_arch": {
                "type":        "HMTImageClassifier",
                "num_classes": len(class_names),
                "dim":         cfg["hidden_dim"],
                "num_layers":  cfg["num_layers"],
                "patch_size":  8 if data_info.get("img_size", 64) <= 32 else 16,
                "img_size":    data_info.get("img_size", 64),
            },
        }, pt_path)

        meta = {
            "name":         name,
            "model_type":   "Image Classification",
            "epochs":       cfg["epochs"],
            "loss":         self.stat_vars["loss"].get(),
            "accuracy":     self.stat_vars["acc"].get(),
            "reflector":    cfg["reflector"],
            "created":      datetime.now().isoformat(),
            "status":       "ready",
            "config":       cfg,
            "weights_file": str(pt_path),
            "num_classes":  len(class_names),
            "class_names":  class_names,
            "img_size":     data_info.get("img_size", 64),
            "task":         "image_classification",
        }
        with open(save_dir / f"{name}.json", "w") as f:
            json.dump(meta, f, indent=2)
        self._log(f"Image model saved → {name}.pt", "ok")

    def _lm_sample(self, model, tokenizer, seq_len: int,
                   prompt: str = "\n", max_new: int = 60) -> str:
        """Quick sample during training to show progress."""
        try:
            from core.text_model import MambaLM
            prompt_ids = tokenizer.encode(prompt)
            new_ids    = model.generate(
                prompt_ids, max_new=max_new,
                temperature=0.8, top_k=40)
            return tokenizer.decode(new_ids)
        except Exception:
            return ""

    def _save_lm_checkpoint(self, cfg, model, best_state,
                             tokenizer, lm_cfg, data_info):
        import torch
        from core.text_model import save_lm
        save_dir = Path("trained_models")
        save_dir.mkdir(exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"TextGen_MambaLM_{ts}"
        pt_path = save_dir / f"{name}.pt"

        if best_state:
            model.load_state_dict(best_state)
        save_lm(model, tokenizer, lm_cfg, str(pt_path))

        meta = {
            "name":         name,
            "model_type":   "Text Generation",
            "epochs":       cfg["epochs"],
            "loss":         self.stat_vars["loss"].get(),
            "accuracy":     "see perplexity",
            "reflector":    cfg["reflector"],
            "created":      datetime.now().isoformat(),
            "status":       "ready",
            "config":       cfg,
            "lm_config":    lm_cfg,
            "weights_file": str(pt_path),
            "vocab_size":   data_info["vocab_size"],
            "corpus_chars": data_info["corpus_chars"],
            "task":         "language_model",
        }
        meta_path = save_dir / f"{name}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        self._log(f"LM checkpoint saved → {name}.pt", "ok")

    def _run_classifier_training(self, cfg, files):
        import torch
        import torch.nn as nn
        import torch.optim as optim

        epochs     = cfg["epochs"]
        batch_size = cfg["batch_size"]
        lr         = cfg["lr"]
        start_t    = time.time()

        # ── 1. Load data ──────────────────────────────────────────────────────
        try:
            from data.data_loader import build_loaders
            self._ui(self._log, "Loading dataset…", "info")
            train_loader, val_loader, data_info = build_loaders(
                files, batch_size=batch_size, val_split=0.15, num_workers=0)

            from data.prefetch_loader import PrefetchLoader
            train_loader = PrefetchLoader(train_loader, buffer_size=3)
            feat_dim = data_info["feature_dim"]
            self._ui(self._log,
                f"Dataset: {data_info['total_rows']} rows | "
                f"{feat_dim} features | "
                f"{data_info['train_batches']} train batches", "ok")
            if data_info["files_skipped"]:
                self._ui(self._log,
                    f"Skipped (unsupported): "
                    f"{[Path(f).name for f in data_info['files_skipped']]}",
                    "warn")
        except Exception as e:
            self._ui(self._log, f"Data loading failed: {e}", "err")
            self._ui(self._finish_training)
            return

        # ── 2. Build model ────────────────────────────────────────────────────
        hidden = cfg["hidden_dim"]
        layers = cfg["num_layers"]
        mtype  = cfg["model_type"]
        n_classes = data_info.get("num_classes", 1)

        try:
            model = self._build_model(mtype, feat_dim, hidden, layers, n_classes)
            param_count = sum(p.numel() for p in model.parameters())
            self._ui(self._log,
                f"Model: {mtype} | {param_count:,} params | "
                f"input_dim={feat_dim} hidden={hidden} layers={layers}", "ok")

            # ── Device selection ──────────────────────────────────────────
            from core.device_manager import get_best_device, move_batch
            device, device_name = get_best_device(
                model_params=param_count, batch_size=batch_size)
            model = model.to(device)
            self._ui(self._log, f"Device: {device_name}", "info")
        except Exception as e:
            self._ui(self._log, f"Model build failed: {e}", "err")
            self._ui(self._finish_training)
            return

        # ── 3. Optimizer + loss ───────────────────────────────────────────────
        opt_name = cfg["optimizer"]
        if opt_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif opt_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        # Scheduler
        sched_name = cfg["scheduler"]
        if sched_name == "CosineAnnealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs)
        elif sched_name == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, epochs // 3), gamma=0.5)
        else:
            scheduler = None

        is_binary = data_info.get("is_binary", True)
        num_classes = data_info.get("num_classes", 1)
        if is_binary:
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()
            self._ui(self._log, f"Multi-class ({num_classes} classes) using CrossEntropyLoss", "info")

        # ── 4. Training loop ──────────────────────────────────────────────────
        best_val_loss = float("inf")
        best_state    = None

        for epoch in range(1, epochs + 1):
            if self._stop_flag.is_set():
                self._ui(self._log, "Training stopped by user.", "warn")
                break

            model.train()
            epoch_loss = 0.0
            correct = total = 0
            refl_conf = 0.0
            n_batches    = len(train_loader)
            update_every = max(5, n_batches // 20)

            for batch_idx, (xb, yb) in enumerate(train_loader, 1):
                if self._stop_flag.is_set():
                    break
                xb, yb = move_batch((xb, yb), device)
                optimizer.zero_grad()
                out  = model(xb)
                if is_binary:
                    loss = loss_fn(out, yb)
                    preds = (torch.sigmoid(out) > 0.5).squeeze()
                    correct += (preds.long() == yb.long().squeeze()).sum().item()
                else:
                    yb_long = yb.long().squeeze() if yb.dim() > 1 else yb.long()
                    loss = loss_fn(out, yb_long)
                    preds = out.argmax(dim=1)
                    correct += (preds == yb_long).sum().item()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                total   += yb.numel()

                # ── Real-time UI update ───────────────────────────────────
                if batch_idx % update_every == 0 or batch_idx == n_batches:
                    avg_so_far = epoch_loss / batch_idx
                    acc_so_far = correct / max(total, 1)
                    cur_lr     = optimizer.param_groups[0]["lr"]
                    elapsed    = time.time() - start_t
                    total_steps = epochs * n_batches
                    done_steps  = (epoch - 1) * n_batches + batch_idx
                    eta_s       = int((elapsed / done_steps) *
                                      (total_steps - done_steps))
                    eta_str     = f"{eta_s//60}m {eta_s%60}s"
                    pct         = (done_steps / total_steps) * 100

                    refl_conf = 0.0
                    if cfg["reflector"]:
                        try:
                            with torch.no_grad():
                                if is_binary:
                                    sig = torch.sigmoid(out)
                                else:
                                    sig = torch.softmax(out, dim=-1)
                                refl_conf = float(
                                    (sig.max(dim=0).values.mean() +
                                     (1 - sig.min(dim=0).values.mean())) / 2)
                        except Exception:
                            pass

                    self._ui(self._update_stats, epoch, epochs,
                             avg_so_far, acc_so_far, cur_lr,
                             eta_str, refl_conf, pct)

            if scheduler:
                scheduler.step()

            avg_loss = epoch_loss / n_batches
            acc      = correct / max(total, 1)
            cur_lr   = optimizer.param_groups[0]["lr"]

            # Validation
            if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                model.eval()
                val_loss = val_correct = val_total = 0
                with torch.no_grad():
                    for xv, yv in val_loader:
                        xv, yv = move_batch((xv, yv), device)
                        ov = model(xv)
                        if is_binary:
                            val_loss += loss_fn(ov, yv).item()
                            pv = (torch.sigmoid(ov) > 0.5).squeeze()
                            val_correct += (pv.long() == yv.long().squeeze()).sum().item()
                        else:
                            yv_long = yv.long().squeeze() if yv.dim() > 1 else yv.long()
                            val_loss += loss_fn(ov, yv_long).item()
                            pv = ov.argmax(dim=1)
                            val_correct += (pv == yv_long).sum().item()
                        val_total   += yv.numel()
                val_loss /= len(val_loader)
                val_acc   = val_correct / max(val_total, 1)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state    = {k: v.clone()
                                     for k, v in model.state_dict().items()}
                self._ui(self._log,
                    f"Epoch {epoch}/{epochs} | "
                    f"loss={avg_loss:.4f} val_loss={val_loss:.4f} | "
                    f"acc={acc*100:.1f}% val_acc={val_acc*100:.1f}%", "ok")
                model.train()

            elapsed = time.time() - start_t
            eta_s   = int((elapsed / epoch) * (epochs - epoch))
            eta_str = f"{eta_s//60}m {eta_s%60}s"
            pct     = (epoch / epochs) * 100

            self._ui(self._update_stats, epoch, epochs, avg_loss, acc,
                     cur_lr, eta_str, refl_conf, pct)

        else:
            self._ui(self._log, "Training complete.", "ok")
            self._ui(self._save_checkpoint, cfg, model, best_state, data_info)

        self._ui(self._finish_training)

    def _finish_training(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _save_checkpoint(self, cfg, model=None, best_state=None,
                          data_info=None):
        import torch
        save_dir = Path("trained_models")
        save_dir.mkdir(exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{cfg['model_type'].replace(' ','_')}_{ts}"

        # Save weights
        weights_path = save_dir / f"{name}.pt"
        if model is not None:
            state = best_state if best_state else model.state_dict()
            torch.save({
                "model_state_dict": state,
                "config":           cfg,
                "data_info":        data_info or {},
            }, weights_path)

        meta = {
            "name":        name,
            "model_type":  cfg["model_type"],
            "epochs":      cfg["epochs"],
            "loss":        self.stat_vars["loss"].get(),
            "accuracy":    self.stat_vars["acc"].get(),
            "reflector":   cfg["reflector"],
            "created":     datetime.now().isoformat(),
            "status":      "ready",
            "config":      cfg,
            "weights_file": str(weights_path) if model is not None else None,
            "feature_dim": (data_info or {}).get("feature_dim"),
            "num_classes": (data_info or {}).get("num_classes"),
            "train_rows":  (data_info or {}).get("train_rows"),
        }
        meta_path = save_dir / f"{name}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        self._log(f"Checkpoint saved → {name}.pt + {name}.json", "ok")

# ─── Model Manager Panel (right column) ──────────────────────────────────────

class ModelManagerPanel(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG_PANEL, **kw)
        self.models = []
        self.selected = None
        self._build()
        self._refresh()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=BG_PANEL)
        hdr.pack(fill="x", padx=12, pady=(12, 6))
        label(hdr, "MODELS", fg=TEXT_SEC, bg=BG_PANEL,
              font=("Segoe UI", 8, "bold")).pack(side="left")
        ghost_btn(hdr, "⟳ Refresh", self._refresh, width=10).pack(side="right")

        # Summary cards
        sum_row = tk.Frame(self, bg=BG_PANEL)
        sum_row.pack(fill="x", padx=12, pady=(0, 8))
        self.sum_vars = {}
        for i, (title, key) in enumerate([("Total","total"),
                                           ("Ready","ready"),
                                           ("Training","training")]):
            card = styled_frame(sum_row, bg=BG_CARD)
            card.grid(row=0, column=i, padx=3, sticky="ew")
            sum_row.columnconfigure(i, weight=1)
            tk.Label(card, text=title, fg=TEXT_SEC, bg=BG_CARD,
                     font=("Segoe UI", 7, "bold")).pack(pady=(6, 0))
            v = tk.StringVar(value="0")
            self.sum_vars[key] = v
            color = {"total": TEXT_PRI, "ready": TEXT_OK,
                     "training": TEXT_WARN}[key]
            tk.Label(card, textvariable=v, fg=color, bg=BG_CARD,
                     font=("Segoe UI", 14, "bold")).pack(pady=(0, 6))

        # Model list
        list_frame = styled_frame(self, bg=BG_CARD)
        list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        cols = ("Name", "Type", "Acc", "Status")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings",
                                 height=10, selectmode="browse")
        style = ttk.Style()
        style.configure("Treeview", background=BG_CARD, foreground=TEXT_PRI,
                        fieldbackground=BG_CARD, borderwidth=0,
                        font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background=BG_INPUT,
                        foreground=TEXT_SEC, font=("Segoe UI", 8, "bold"),
                        relief="flat")
        style.map("Treeview", background=[("selected", ACCENT2)])

        widths = {"Name": 160, "Type": 110, "Acc": 60, "Status": 70}
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=widths[col], anchor="w")

        tsb = ttk.Scrollbar(list_frame, orient="vertical",
                            command=self.tree.yview)
        self.tree.config(yscrollcommand=tsb.set)
        tsb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Detail card
        self.detail_frame = styled_frame(self, bg=BG_CARD)
        self.detail_frame.pack(fill="x", padx=12, pady=(0, 8))
        section_title(self.detail_frame, "  MODEL DETAILS", bg=BG_CARD).pack(
            anchor="w", pady=(6, 4), padx=8)
        self.detail_text = tk.Text(self.detail_frame, bg=BG_CARD, fg=TEXT_PRI,
                                   font=("Consolas", 8), relief="flat",
                                   height=7, state="disabled",
                                   wrap="word")
        self.detail_text.pack(fill="x", padx=8, pady=(0, 8))

        # Action buttons
        act_row = tk.Frame(self, bg=BG_PANEL)
        act_row.pack(fill="x", padx=12, pady=(0, 12))
        accent_btn(act_row, "▶ Run Inference", self._run_inference_ui,
                   color=ACCENT2, width=16).pack(side="left", padx=(0, 6))
        ghost_btn(act_row, "Export", self._export_model, width=10).pack(
            side="left", padx=(0, 6))
        ghost_btn(act_row, "Delete", self._delete_model, width=10).pack(
            side="left")

    def _refresh(self):
        if not self.winfo_exists(): return
        self.models = []
        save_dir = Path("trained_models")
        if save_dir.exists():
            for f in sorted(save_dir.glob("*.json"),
                            key=lambda x: x.stat().st_mtime, reverse=True):
                try:
                    with open(f) as fp:
                        m = json.load(fp)
                        m["_path"] = str(f)
                        self.models.append(m)
                except Exception:
                    pass

        self.tree.delete(*self.tree.get_children())
        ready = training = 0
        for m in self.models:
            status = m.get("status", "unknown")
            acc    = m.get("accuracy", "—")
            tag    = "ready" if status == "ready" else "other"
            self.tree.insert("", "end", iid=m["name"],
                             values=(m["name"], m.get("model_type","—"),
                                     acc, status),
                             tags=(tag,))
            if status == "ready":    ready += 1
            if status == "training": training += 1

        self.tree.tag_configure("ready", foreground=TEXT_OK)
        self.tree.tag_configure("other", foreground=TEXT_WARN)
        self.sum_vars["total"].set(str(len(self.models)))
        self.sum_vars["ready"].set(str(ready))
        self.sum_vars["training"].set(str(training))

    def _on_select(self, _=None):
        if not self.winfo_exists(): return
        sel = self.tree.selection()
        if not sel:
            return
        name = sel[0]
        m = next((x for x in self.models if x["name"] == name), None)
        if not m:
            return
        self.selected = m
        cfg = m.get("config", {})
        wf  = m.get("weights_file") or "—"
        lines = [
            f"Name:        {m.get('name','—')}",
            f"Type:        {m.get('model_type','—')}",
            f"Status:      {m.get('status','—')}",
            f"Accuracy:    {m.get('accuracy','—')}",
            f"Loss:        {m.get('loss','—')}",
            f"Epochs:      {cfg.get('epochs','—')}",
            f"Hidden Dim:  {cfg.get('hidden_dim','—')}",
            f"Layers:      {cfg.get('num_layers','—')}",
            f"Feat Dim:    {m.get('feature_dim','—')}",
            f"Classes:     {m.get('num_classes','—')}",
            f"Train Rows:  {m.get('train_rows','—')}",
            f"Reflector:   {cfg.get('reflector','—')}",
            f"Weights:     {Path(wf).name if wf != '—' else '—'}",
            f"Created:     {m.get('created','—')[:19]}",
        ]
        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("end", "\n".join(lines))
        self.detail_text.config(state="disabled")

    def _run_inference_ui(self):
        if not self.selected:
            messagebox.showinfo("No Selection", "Select a model first.")
            return
        wf = self.selected.get("weights_file")
        if not wf or not Path(wf).exists():
            messagebox.showwarning("No Weights",
                f"Weights file not found:\n{wf}\n\n"
                "Re-train the model to generate a .pt file.")
            return
        InferenceWindow(self, self.selected)

    def _export_model(self):
        if not self.selected:
            messagebox.showinfo("No Selection", "Select a model first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=self.selected["name"],
            filetypes=[("JSON", "*.json")])
        if path:
            import shutil
            shutil.copy(self.selected["_path"], path)
            messagebox.showinfo("Exported", f"Saved to {path}")

    def _delete_model(self):
        if not self.selected:
            messagebox.showinfo("No Selection", "Select a model first.")
            return
        if messagebox.askyesno("Delete",
                f"Delete '{self.selected['name']}'?"):
            try:
                os.remove(self.selected["_path"])
            except Exception:
                pass
            self.selected = None
            self._refresh()

# ─── Inference Window ────────────────────────────────────────────────────────

class InferenceWindow(tk.Toplevel):
    """Full inference UI — pick data file, set threshold, run, see results."""

    def __init__(self, parent, model_meta: dict):
        super().__init__(parent)
        self.model_meta = model_meta
        self.title(f"Inference — {model_meta['name']}")
        self.configure(bg=BG_DARK)
        self.geometry("780x680")
        self.minsize(680, 560)
        self._result = None
        self._build()

    def _ui(self, fn, *args):
        """Thread-safe UI update helper with safety check."""
        try:
            if self.winfo_exists():
                self.after(0, fn, *args)
        except Exception:
            pass

    def _build(self):
        # ── Header ──
        hdr = tk.Frame(self, bg=BG_PANEL, height=46)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        label(hdr, f"▶  Inference", fg=TEXT_PRI, bg=BG_PANEL,
              font=("Segoe UI", 12, "bold")).pack(side="left", padx=14, pady=10)
        label(hdr, self.model_meta["name"], fg=TEXT_SEC, bg=BG_PANEL,
              font=("Segoe UI", 9)).pack(side="left")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── Model info card ──
        info_frame = styled_frame(self, bg=BG_CARD)
        info_frame.pack(fill="x", padx=12, pady=(10, 6))
        cfg = self.model_meta.get("config", {})
        info_lines = [
            ("Type",       self.model_meta.get("model_type", "—")),
            ("Accuracy",   self.model_meta.get("accuracy", "—")),
            ("Loss",       self.model_meta.get("loss", "—")),
            ("Feat Dim",   str(self.model_meta.get("feature_dim", "—"))),
            ("Epochs",     str(cfg.get("epochs", "—"))),
            ("Hidden Dim", str(cfg.get("hidden_dim", "—"))),
        ]
        row_f = tk.Frame(info_frame, bg=BG_CARD)
        row_f.pack(fill="x", padx=10, pady=8)
        for i, (k, v) in enumerate(info_lines):
            col = tk.Frame(row_f, bg=BG_CARD)
            col.grid(row=0, column=i, padx=10)
            row_f.columnconfigure(i, weight=1)
            label(col, k, fg=TEXT_SEC, bg=BG_CARD,
                  font=("Segoe UI", 7, "bold")).pack()
            label(col, v, fg=TEXT_PRI, bg=BG_CARD,
                  font=("Segoe UI", 10, "bold")).pack()

        # ── Data file picker (hidden for LM models) ──
        self._is_lm = self.model_meta.get("task") == "language_model"

        pick_frame = styled_frame(self, bg=BG_CARD)
        pick_frame.pack(fill="x", padx=12, pady=(0, 6))

        if self._is_lm:
            # Language model: show prompt box instead of file picker
            section_title(pick_frame, "  PROMPT", bg=BG_CARD).pack(
                anchor="w", padx=8, pady=(8, 4))
            self._prompt_var = tk.StringVar(value="Once upon a time")
            tk.Entry(pick_frame, textvariable=self._prompt_var,
                     bg=BG_INPUT, fg=TEXT_PRI, insertbackground=TEXT_PRI,
                     relief="flat", font=("Segoe UI", 10)).pack(
                fill="x", padx=8, pady=(0, 8))
        else:
            section_title(pick_frame, "  DATA FILE", bg=BG_CARD).pack(
                anchor="w", padx=8, pady=(8, 4))
            pick_row = tk.Frame(pick_frame, bg=BG_CARD)
            pick_row.pack(fill="x", padx=8, pady=(0, 8))
            self.data_var = tk.StringVar(value=self._default_data())
            tk.Entry(pick_row, textvariable=self.data_var, bg=BG_INPUT,
                     fg=TEXT_PRI, insertbackground=TEXT_PRI, relief="flat",
                     font=("Segoe UI", 9)).pack(side="left", fill="x",
                                                 expand=True, padx=(0, 6))
            ghost_btn(pick_row, "Browse", self._browse_data, width=8).pack(side="left")

        # ── Options ──
        opt_frame = styled_frame(self, bg=BG_CARD)
        opt_frame.pack(fill="x", padx=12, pady=(0, 6))
        section_title(opt_frame, "  OPTIONS", bg=BG_CARD).pack(
            anchor="w", padx=8, pady=(8, 4))
        opt_row = tk.Frame(opt_frame, bg=BG_CARD)
        opt_row.pack(fill="x", padx=8, pady=(0, 8))

        if self._is_lm:
            # LM options: temperature, top-k, tokens to generate
            for lbl, var_name, default, lo, hi, inc in [
                ("Temperature", "_temp_var",  "0.8",  "0.1", "2.0", "0.1"),
                ("Top-K",       "_topk_var",  "40",   "1",   "200", "5"),
                ("Generate N",  "_ngen_var",  "300",  "10",  "2000","50"),
            ]:
                label(opt_row, f"{lbl}:", fg=TEXT_SEC, bg=BG_CARD,
                      font=("Segoe UI", 9)).pack(side="left", padx=(0, 2))
                v = tk.StringVar(value=default)
                setattr(self, var_name, v)
                tk.Spinbox(opt_row, from_=float(lo), to=float(hi),
                           increment=float(inc), textvariable=v,
                           bg=BG_INPUT, fg=TEXT_PRI, insertbackground=TEXT_PRI,
                           relief="flat", font=("Segoe UI", 9), width=6,
                           buttonbackground=BG_INPUT).pack(side="left", padx=(0, 14))
        else:
            label(opt_row, "Threshold:", fg=TEXT_SEC, bg=BG_CARD,
                  font=("Segoe UI", 9)).pack(side="left")
            self.thresh_var = tk.StringVar(value="0.5")
            tk.Spinbox(opt_row, from_=0.1, to=0.9, increment=0.05,
                       textvariable=self.thresh_var, format="%.2f",
                       bg=BG_INPUT, fg=TEXT_PRI, insertbackground=TEXT_PRI,
                       relief="flat", font=("Segoe UI", 9), width=6,
                       buttonbackground=BG_INPUT).pack(side="left", padx=(4, 20))

            self.save_var = tk.BooleanVar(value=True)
            tk.Checkbutton(opt_row, text="Save results to inference_results/",
                           variable=self.save_var, bg=BG_CARD, fg=TEXT_PRI,
                           selectcolor=BG_INPUT, activebackground=BG_CARD,
                           font=("Segoe UI", 9)).pack(side="left")

        # ── Run button ──
        btn_row = tk.Frame(self, bg=BG_DARK)
        btn_row.pack(fill="x", padx=12, pady=(0, 6))
        self.run_btn = accent_btn(btn_row, "▶  Run Inference",
                                  self._run, color=ACCENT2, width=18)
        self.run_btn.pack(side="left")
        self.status_lbl = label(btn_row, "Ready", fg=TEXT_SEC, bg=BG_DARK,
                                font=("Segoe UI", 9))
        self.status_lbl.pack(side="left", padx=12)

        # ── Results ──
        res_frame = styled_frame(self, bg=BG_CARD)
        res_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        section_title(res_frame, "  RESULTS", bg=BG_CARD).pack(
            anchor="w", padx=8, pady=(8, 4))
        self.result_text = tk.Text(
            res_frame, bg=BG_CARD, fg=TEXT_PRI,
            font=("Consolas", 9), relief="flat", state="disabled", wrap="word")
        sb = ttk.Scrollbar(res_frame, orient="vertical",
                           command=self.result_text.yview)
        self.result_text.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.result_text.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        self.result_text.tag_config("header",  foreground=ACCENT2,
                                    font=("Consolas", 9, "bold"))
        self.result_text.tag_config("good",    foreground=TEXT_OK)
        self.result_text.tag_config("warn",    foreground=TEXT_WARN)
        self.result_text.tag_config("metric",  foreground=TEXT_PRI)
        self.result_text.tag_config("section", foreground=TEXT_SEC,
                                    font=("Consolas", 8, "bold"))

    def _default_data(self) -> str:
        for p in Path("randomDATA").glob("*.csv"):
            return str(p)
        return ""

    def _browse_data(self):
        path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[("CSV files", "*.csv"), ("NumPy", "*.npy *.npz"),
                       ("All", "*.*")])
        if path:
            self.data_var.set(path)

    def _run(self):
        data_path = self.data_var.get().strip()
        # Text generation models don't need a data file for inference
        is_lm = self.model_meta.get("task") == "language_model"

        if not is_lm and (not data_path or not Path(data_path).exists()):
            messagebox.showwarning("No Data", "Select a valid data file.")
            return
        try:
            threshold = float(self.thresh_var.get())
        except ValueError:
            threshold = 0.5

        self.run_btn.config(state="disabled")
        self.status_lbl.config(text="Running…", fg=TEXT_WARN)

        if is_lm:
            self._write("Generating text…\n", "section")
            def _worker():
                try:
                    self._run_lm_inference()
                except Exception as e:
                    self.after(0, self._show_error, str(e))
        else:
            self._write("Running inference…\n", "section")
            def _worker():
                try:
                    from utils.inference import load_checkpoint, run_inference, save_results
                    wf = self.model_meta["weights_file"]
                    model, config, data_info = load_checkpoint(wf)
                    results = run_inference(model, data_info, data_path,
                                            threshold=threshold)
                    if self.save_var.get():
                        out = save_results(results, self.model_meta["name"],
                                           data_path)
                    else:
                        out = None
                    self._ui(self._show_results, results, out)
                except Exception as e:
                    self._ui(self._show_error, str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _run_lm_inference(self):
        """Generate text from a trained language model."""
        from core.text_model import load_lm
        wf     = self.model_meta["weights_file"]
        model, tokenizer = load_lm(wf)
        prompt = self._prompt_var.get() if hasattr(self, "_prompt_var") else "\n"
        if not prompt:
            prompt = "\n"

        try:
            temp  = float(self._temp_var.get())
        except Exception:
            temp  = 0.8
        try:
            topk  = int(self._topk_var.get())
        except Exception:
            topk  = 40
        try:
            n_new = int(self._ngen_var.get())
        except Exception:
            n_new = 300

        prompt_ids = tokenizer.encode(prompt)
        new_ids    = model.generate(
            prompt_ids, max_new=n_new,
            temperature=temp, top_k=topk)
        generated  = tokenizer.decode(new_ids)
        self._ui(self._show_lm_output, prompt, generated)

    def _show_lm_output(self, prompt: str, generated: str):
        self.run_btn.config(state="normal")
        self.status_lbl.config(text=f"Generated {len(generated)} chars", fg=TEXT_OK)

        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")

        self.result_text.insert("end", "═" * 52 + "\n", "header")
        self.result_text.insert("end", "  TEXT GENERATION OUTPUT\n", "header")
        self.result_text.insert("end", "═" * 52 + "\n", "header")
        self.result_text.insert("end", "\n  PROMPT\n", "section")
        self.result_text.insert("end", f"  {prompt}\n\n")
        self.result_text.insert("end", "  GENERATED\n", "section")
        self.result_text.insert("end", generated + "\n\n")
        self.result_text.insert("end", "═" * 52 + "\n", "header")
        self.result_text.config(state="disabled")
        self.result_text.see("1.0")

    def _show_results(self, r: dict, saved_path: str):
        self.run_btn.config(state="normal")
        self.status_lbl.config(
            text=f"Done — {r['accuracy']*100:.1f}% accuracy", fg=TEXT_OK)

        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")

        w = lambda text, tag="metric": self.result_text.insert("end", text, tag)

        w("═" * 52 + "\n", "header")
        w("  INFERENCE RESULTS\n", "header")
        w("═" * 52 + "\n", "header")

        w("\n  DATASET\n", "section")
        w(f"  Total samples      {r['total_samples']:>10,}\n")
        w(f"  Threshold          {r['threshold']:>10.2f}\n")

        w("\n  PREDICTIONS\n", "section")
        w(f"  Attacks detected   {r['attack_detected']:>10,}\n")
        w(f"  Benign detected    {r['benign_detected']:>10,}\n")
        w(f"  Avg confidence     {r['avg_confidence']:>10.3f}\n")
        w(f"  High conf (>0.8)   {r['high_confidence']:>10,}\n")
        w(f"  Low conf  (<0.3)   {r['low_confidence']:>10,}\n")

        w("\n  CONFUSION MATRIX\n", "section")
        w(f"  {'':>16}  Pred 0    Pred 1\n")
        w(f"  {'Actual 0 (Benign)':<16}  {r['tn']:>8,}  {r['fp']:>8,}\n")
        w(f"  {'Actual 1 (Attack)':<16}  {r['fn']:>8,}  {r['tp']:>8,}\n")

        w("\n  METRICS\n", "section")
        acc_tag = "good" if r["accuracy"] > 0.85 else "warn"
        w(f"  Accuracy           {r['accuracy']*100:>9.2f}%\n", acc_tag)
        w(f"  Precision          {r['precision']*100:>9.2f}%\n")
        rec_tag = "good" if r["recall"] > 0.80 else "warn"
        w(f"  Recall             {r['recall']*100:>9.2f}%\n", rec_tag)
        f1_tag  = "good" if r["f1_score"] > 0.80 else "warn"
        w(f"  F1 Score           {r['f1_score']:>10.4f}\n", f1_tag)
        fpr_tag = "good" if r["false_positive_rate"] < 0.05 else "warn"
        w(f"  False Positive     {r['false_positive_rate']*100:>9.2f}%\n", fpr_tag)

        f1 = r["f1_score"]
        grade = ("EXCELLENT" if f1 > 0.90 else
                 "GOOD"      if f1 > 0.80 else
                 "FAIR"      if f1 > 0.65 else "NEEDS IMPROVEMENT")
        grade_tag = "good" if f1 > 0.80 else "warn"
        w(f"\n  GRADE: {grade}\n", grade_tag)

        if saved_path:
            w(f"\n  Saved → {saved_path}\n", "section")

        w("═" * 52 + "\n", "header")
        self.result_text.config(state="disabled")
        self.result_text.see("1.0")

    def _show_error(self, err: str):
        self.run_btn.config(state="normal")
        self.status_lbl.config(text="Error", fg=TEXT_ERR)
        self.result_text.config(state="normal")
        self.result_text.insert("end", f"Error: {err}\n", "warn")
        self.result_text.config(state="disabled")

    def _write(self, text: str, tag: str = ""):
        self.result_text.config(state="normal")
        self.result_text.insert("end", text, tag)
        self.result_text.config(state="disabled")


# ─── Codebase Health Check ────────────────────────────────────────────────────

class HealthPanel(tk.Toplevel):
    REQUIRED_STRUCTURE = [
        ("core/",                        "Core module directory"),
        ("core/__init__.py",             "Core module exports"),
        ("core/architecture.py",         "Base classes and orchestrator"),
        ("core/implementations.py",      "Feeders, encoder, decoder"),
        ("core/mamba.py",               "Mamba block implementation"),
        ("core/transformer.py",          "Transformer block implementation"),
        ("core/text_model.py",           "Language model utilities"),
        ("core/device_manager.py",       "GPU/CPU device management"),
        ("training/",                    "Training module directory"),
        ("training/__init__.py",         "Training module exports"),
        ("training/trainer.py",          "Cybersecurity trainer"),
        ("training/reflector_trainer.py","Reflector + integrated trainer"),
        ("data/",                        "Data module directory"),
        ("data/__init__.py",             "Data module exports"),
        ("data/data_loader.py",          "CSV/NPY data loading"),
        ("data/text_dataset.py",         "Text dataset for LM training"),
        ("data/image_dataset.py",        "Image dataset loading"),
        ("data/prefetch_loader.py",      "Async data prefetching"),
        ("ui/",                          "UI module directory"),
        ("ui/__init__.py",               "UI module exports"),
        ("ui/chat.py",                  "Chat interface"),
        ("ui/training_ui.py",           "Training GUI"),
        ("ui/upgrade_window.py",        "Auto-upgrade UI"),
        ("utils/",                       "Utils module directory"),
        ("utils/__init__.py",            "Utils module exports"),
        ("utils/data_classifier.py",    "Universal data classifier"),
        ("utils/inference.py",          "Inference engine"),
        ("utils/project_context.py",    "Project context analysis"),
        ("utils/smart_upgrade.py",      "Smart upgrade system"),
        ("utils/auto_upgrade.py",       "Auto-upgrade system"),
        ("README.md",                    "Documentation"),
    ]
    
    OPTIONAL_FILES = [
        ("mamba_kernel.cpp",      "C++ Mamba kernel (optional)"),
        ("reflector_kernel.cpp",   "C++ reflector kernel (optional)"),
    ]

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Codebase Health Check")
        self.configure(bg=BG_DARK)
        self.geometry("700x580")
        self.resizable(True, True)
        self._build()
        self._run_check()

    def _build(self):
        tk.Label(self, text="Codebase Health Check",
                 fg=TEXT_PRI, bg=BG_DARK,
                 font=("Segoe UI", 13, "bold")).pack(pady=(16, 4))
        tk.Label(self, text="Verifying modular folder structure",
                 fg=TEXT_SEC, bg=BG_DARK,
                 font=("Segoe UI", 9)).pack(pady=(0, 12))

        frame = styled_frame(self, bg=BG_CARD)
        frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        cols = ("Location", "Description", "Status")
        self.tree = ttk.Treeview(frame, columns=cols, show="headings")
        style = ttk.Style()
        style.configure("Health.Treeview", background=BG_CARD,
                        foreground=TEXT_PRI, fieldbackground=BG_CARD,
                        font=("Segoe UI", 9))
        self.tree.configure(style="Health.Treeview")
        for col, w in zip(cols, [180, 320, 80]):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="w")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.tree.tag_configure("ok",      foreground=TEXT_OK)
        self.tree.tag_configure("missing", foreground=TEXT_ERR)
        self.tree.tag_configure("opt",     foreground=TEXT_WARN)
        self.tree.tag_configure("dir",     foreground=ACCENT2)

        self.summary_lbl = tk.Label(self, text="", fg=TEXT_PRI, bg=BG_DARK,
                                    font=("Segoe UI", 10, "bold"))
        self.summary_lbl.pack(pady=(0, 8))
        ghost_btn(self, "Close", self.destroy, width=10).pack(pady=(0, 12))

    def _check_path(self, path: str) -> tuple:
        """Check if path exists. Returns (exists, is_directory, is_optional)"""
        p = Path(path)
        is_dir = path.endswith("/")
        is_opt = path in [f[0] for f in self.OPTIONAL_FILES]
        
        if is_dir:
            exists = p.exists() and p.is_dir()
        else:
            exists = p.exists() and p.is_file()
        
        return exists, is_dir, is_opt

    def _run_check(self):
        ok = missing = optional = dirs_ok = 0
        
        for path, desc in self.REQUIRED_STRUCTURE:
            exists, is_dir, is_opt = self._check_path(path)
            
            if is_dir:
                tag = "dir"
                if exists:
                    status = "✓ Dir"
                    dirs_ok += 1
                else:
                    status = "✗ Missing"
                    missing += 1
            elif exists:
                status = "✓ Found"; tag = "ok"; ok += 1
            elif is_opt:
                status = "⚠ Optional"; tag = "opt"; optional += 1
            else:
                status = "✗ Missing"; tag = "missing"; missing += 1
            
            self.tree.insert("", "end", values=(path, desc, status), tags=(tag,))
        
        # Check optional files
        for path, desc in self.OPTIONAL_FILES:
            exists = Path(path).exists()
            if exists:
                status = "✓ Found"; tag = "ok"; ok += 1
            else:
                status = "⚠ Optional"; tag = "opt"; optional += 1
            self.tree.insert("", "end", values=(path, desc, status), tags=(tag,))

        if missing == 0:
            msg = f"✓ All required files/directories present  ({dirs_ok} dirs, {ok} files, {optional} optional)"
            color = TEXT_OK
        else:
            msg = f"✗ {missing} item(s) missing  ({dirs_ok} dirs, {ok} files, {optional} optional)"
            color = TEXT_ERR
        self.summary_lbl.config(text=msg, fg=color)


# ─── Main Application Window ──────────────────────────────────────────────────

class TrainingApp(tk.Tk):
    def __init__(self, skip_init=False):
        if not skip_init:
            super().__init__()
        self.title("ML Training System — Hierarchical Mamba + Transformer")
        self.configure(bg=BG_DARK)
        self.geometry("1400x860")
        self.minsize(1100, 700)
        self._closing = False
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self._build_titlebar()
        self._build_body()

    def _on_closing(self):
        """Ensure all background threads stop before exiting."""
        if self._closing:
            return
        self._closing = True
        
        if hasattr(self, "train_panel") and self.train_panel.running:
            self.train_panel._stop_flag.set()
            self.train_panel.running = False
            self.after(500, self._force_destroy)
        else:
            self._force_destroy()
    
    def _force_destroy(self):
        """Force destroy the window."""
        try:
            self.destroy()
        except Exception:
            pass

    def _build_titlebar(self):
        bar = tk.Frame(self, bg=BG_PANEL, height=48)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        tk.Label(bar, text="⬡  ML Training System",
                 fg=TEXT_PRI, bg=BG_PANEL,
                 font=("Segoe UI", 12, "bold")).pack(side="left", padx=16,
                                                      pady=12)
        tk.Label(bar, text="Hierarchical Mamba + Transformer  |  Reflector Auto-Correction",
                 fg=TEXT_SEC, bg=BG_PANEL,
                 font=("Segoe UI", 9)).pack(side="left")

        # Right-side toolbar buttons
        ghost_btn(bar, "Health Check",
                  lambda: HealthPanel(self), width=12).pack(side="right",
                                                             padx=8, pady=8)
        ghost_btn(bar, "⚙ Auto-Upgrade",
                  self._open_upgrade_window, width=14).pack(side="right", pady=8)
        ghost_btn(bar, "Open Models Dir",
                  self._open_models_dir, width=14).pack(side="right", pady=8)

        separator(self).pack(fill="x")

    def _build_body(self):
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=8, pady=8)

        # Left: Data panel
        self.data_panel = DataPanel(body)
        self.data_panel.pack(side="left", fill="y", padx=(0, 4))
        self.data_panel.config(width=280)
        self.data_panel.pack_propagate(False)

        separator_v = tk.Frame(body, bg=BORDER, width=1)
        separator_v.pack(side="left", fill="y")

        # Center: Training panel
        self.train_panel = TrainingPanel(body, self.data_panel.get_files)
        self.train_panel.pack(side="left", fill="both", expand=True,
                              padx=4)

        separator_v2 = tk.Frame(body, bg=BORDER, width=1)
        separator_v2.pack(side="left", fill="y")

        # Right: Model manager
        self.model_panel = ModelManagerPanel(body)
        self.model_panel.pack(side="left", fill="y", padx=(4, 0))
        self.model_panel.config(width=340)
        self.model_panel.pack_propagate(False)

        # Wire model panel refresh after training saves a checkpoint
        orig_save = self.train_panel._save_checkpoint
        def patched_save(cfg, model=None, best_state=None, data_info=None):
            orig_save(cfg, model, best_state, data_info)
            self.after(500, self.model_panel._refresh)
        self.train_panel._save_checkpoint = patched_save

    def _open_upgrade_window(self):
        try:
            from ui.upgrade_window import AutoUpgradeWindow
            # Try to get upgrade system from start.py if available
            upgrade_sys = getattr(self, "_upgrade_system", None)
            AutoUpgradeWindow(self, upgrade_sys)
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Upgrade Window Error", str(e))

    def _open_models_dir(self):
        d = Path("trained_models")
        d.mkdir(exist_ok=True)
        os.startfile(str(d.resolve()))


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Try to enable tkinterdnd2 for native drag-and-drop
    try:
        from tkinterdnd2 import TkinterDnD
        class TrainingAppDnD(TrainingApp, TkinterDnD.Tk):
            def __init__(self):
                TkinterDnD.Tk.__init__(self)
                TrainingApp.__init__(self, skip_init=True)
        app = TrainingAppDnD()
    except ImportError:
        app = TrainingApp()

    app.mainloop()
