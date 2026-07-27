import tkinter as tk
from tkinter import ttk
from pathlib import Path
from .theme import (
    BG_DARK, BG_CARD, TEXT_PRI, TEXT_SEC, TEXT_OK, TEXT_ERR, TEXT_WARN, ACCENT2,
    styled_frame, ghost_btn
)

class HealthPanel(tk.Toplevel):
    REQUIRED_STRUCTURE = [
        ("core/",                        "Core module directory"),
        ("core/__init__.py",             "Core module exports"),
        ("core/implementations.py",      "LMHead, GenConfig, DecoderOnlyTransformer, HMTLanguageModel"),
        ("core/transformer.py",          "RotaryEmbedding (YaRN), TransformerBlock (GQA, SwiGLU)"),
        ("core/text_model.py",           "lm_val_loss, save_lm, load_lm"),
        ("core/device_manager.py",       "GPU/CPU device selection, move_batch"),
        ("training/",                    "Training module directory"),
        ("training/__init__.py",         "Training module exports"),
        ("training/unified_trainer.py",  "UnifiedTrainer, TrainConfig, train_model"),
        ("data/",                        "Data module directory"),
        ("data/__init__.py",             "Data module exports"),
        ("data/advanced_tokenizer.py",   "BPE tokenizer (HF / fallback)"),
        ("data/text_dataset.py",         "TextLMDataset, build_text_loaders"),
        ("data/chat_dataset.py",         "ChatDataset, ReasoningDataset, build_chat_loaders"),
        ("ui/",                          "UI module directory"),
        ("ui/__init__.py",               "UI module exports"),
        ("ui/training_ui.py",            "Tkinter training GUI"),
        ("ui/training_controller.py",    "TrainingController, save_checkpoint"),
        ("ui/health_window.py",          "Health check panel"),
        ("ui/model_chat.py",            "Terminal chat with GenConfig controls"),
        ("ui/components.py",            "DropZone, DataPanel, LineChart"),
        ("ui/theme.py",                 "Color palette, styled widget factories"),
        ("ui/inference_window.py",      "Inference popup"),
        ("ui/health_window.py",         "This health check"),
        ("eval/",                        "Evaluation module directory"),
        ("eval/__init__.py",             "Evaluation module exports"),
        ("eval/harness.py",             "Perplexity, reasoning accuracy"),
        ("start.py",                    "Entry point"),
        ("requirements.txt",            "Dependencies"),
    ]

    OPTIONAL_FILES = []

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Codebase Health Check")
        self.configure(bg=BG_DARK)
        self.geometry("700x580")
        self._build()
        self._run_check()

    def _build(self):
        tk.Label(self, text="Codebase Health Check", fg=TEXT_PRI, bg=BG_DARK,
                 font=("Segoe UI", 13, "bold")).pack(pady=(16, 4))
        frame = styled_frame(self, bg=BG_CARD)
        frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        cols = ("Location", "Description", "Status")
        self.tree = ttk.Treeview(frame, columns=cols, show="headings")
        for col, w in zip(cols, [180, 320, 80]):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="w")
        
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True, padx=4, pady=4)

        self.tree.tag_configure("ok", foreground=TEXT_OK)
        self.tree.tag_configure("missing", foreground=TEXT_ERR)
        self.tree.tag_configure("opt", foreground=TEXT_WARN)
        self.tree.tag_configure("dir", foreground=ACCENT2)

        self.summary_lbl = tk.Label(self, text="", fg=TEXT_PRI, bg=BG_DARK,
                                    font=("Segoe UI", 10, "bold"))
        self.summary_lbl.pack(pady=(0, 8))
        ghost_btn(self, "Close", self.destroy, width=10).pack(pady=(0, 12))

    def _run_check(self):
        ok = missing = optional = dirs_ok = 0
        for path, desc in self.REQUIRED_STRUCTURE:
            p = Path(path)
            is_dir = path.endswith("/")
            exists = (p.exists() and p.is_dir()) if is_dir else (p.exists() and p.is_file())
            
            if is_dir:
                tag, status = "dir", "✓ Dir" if exists else "✗ Missing"
                if exists: dirs_ok += 1
                else: missing += 1
            elif exists:
                tag, status = "ok", "✓ Found"
                ok += 1
            else:
                tag, status = "missing", "✗ Missing"
                missing += 1
            self.tree.insert("", "end", values=(path, desc, status), tags=(tag,))

        for path, desc in self.OPTIONAL_FILES:
            exists = Path(path).exists()
            tag, status = ("ok", "✓ Found") if exists else ("opt", "⚠ Optional")
            if exists: ok += 1
            else: optional += 1
            self.tree.insert("", "end", values=(path, desc, status), tags=(tag,))

        if missing == 0:
            msg = f"✓ All required items present ({dirs_ok} dirs, {ok} files)"
            self.summary_lbl.config(text=msg, fg=TEXT_OK)
        else:
            self.summary_lbl.config(text=f"✗ {missing} items missing", fg=TEXT_ERR)