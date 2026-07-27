"""
Training System Visual Interface
Decoder-only Transformer (GQA, YaRN RoPE, BPE Tokenizer)
Drag-and-drop data loading, live training metrics, model manager
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import os
import math
from pathlib import Path
from datetime import datetime

from .theme import (
    BG_DARK, BG_PANEL, BG_CARD, BG_INPUT, ACCENT, ACCENT_HOV, ACCENT2,
    BORDER, TEXT_PRI, TEXT_SEC, TEXT_WARN, TEXT_ERR, TEXT_OK, DRAG_OVER,
    SUPPORTED_EXTS, ALL_EXTS, styled_frame, label, section_title,
    accent_btn, ghost_btn, separator, fmt_size, setup_ttk_styles
)
from .training_controller import TrainingController
from .components import DropZone, DataPanel, LineChart
from .inference_window import InferenceWindow
from .health_window import HealthPanel
from training.unified_trainer import (
    UnifiedTrainer, TrainConfig, TrainingRuntimeError
)
HF_DATASETS_AVAILABLE = False

# ─── Training Config + Status Panel (center) ─────────────────────────────────

class TrainingPanel(tk.Frame):
    def __init__(self, parent, get_files_cb, **kw):
        super().__init__(parent, bg=BG_PANEL, **kw)
        self.get_files = get_files_cb
        self.on_checkpoint_saved = None
        self.running = False
        self._thread = None
        self._stop_flag = threading.Event()
        self.log_lines = []
        self.controller = TrainingController()
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

        # Seq Len (context length)
        seq_row = tk.Frame(cfg_frame, bg=BG_CARD)
        seq_row.pack(fill="x", padx=8, pady=(1, 4))
        tk.Label(seq_row, text="Seq Length", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        self.seq_len_var = tk.StringVar(value="128")
        tk.Entry(seq_row, textvariable=self.seq_len_var, bg=BG_INPUT, fg=TEXT_PRI,
                 insertbackground=TEXT_PRI, relief="flat",
                 font=("Segoe UI", 9), width=6).pack(side="left", padx=4)
        tk.Label(seq_row, text=" tokens context", fg=TEXT_SEC,
                 bg=BG_CARD, font=("Segoe UI", 8)).pack(side="left")

        # Advanced settings
        adv_frame = tk.Frame(cfg_frame, bg=BG_CARD)
        adv_frame.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(adv_frame, text="Max RAM (MB)", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        self.max_ram_var = tk.StringVar(value="0")
        tk.Spinbox(adv_frame, from_=0, to=65536, increment=512,
                   textvariable=self.max_ram_var, bg=BG_INPUT, fg=TEXT_PRI,
                   font=("Segoe UI", 9), width=7, buttonbackground=BG_INPUT).pack(side="left", padx=4)
        tk.Label(adv_frame, text="CPU threads", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9)).pack(side="left", padx=(10, 2))
        self.cpu_threads_var = tk.StringVar(value="0")
        tk.Spinbox(adv_frame, from_=0, to=64, increment=1,
                   textvariable=self.cpu_threads_var, bg=BG_INPUT, fg=TEXT_PRI,
                   font=("Segoe UI", 9), width=4, buttonbackground=BG_INPUT).pack(side="left", padx=2)

        adv_row2 = tk.Frame(cfg_frame, bg=BG_CARD)
        adv_row2.pack(fill="x", padx=8, pady=(1, 4))
        tk.Label(adv_row2, text="Data workers", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        self.num_workers_var = tk.StringVar(value="0")
        tk.Spinbox(adv_row2, from_=0, to=16, increment=1,
                   textvariable=self.num_workers_var, bg=BG_INPUT, fg=TEXT_PRI,
                   font=("Segoe UI", 9), width=4, buttonbackground=BG_INPUT).pack(side="left", padx=4)
        self.subprocess_var = tk.BooleanVar(value=False)
        tk.Checkbutton(adv_row2, variable=self.subprocess_var, bg=BG_CARD,
                       fg=TEXT_PRI, selectcolor=BG_INPUT,
                       activebackground=BG_CARD).pack(side="left", padx=(4, 2))
        tk.Label(adv_row2, text="Subprocess (no UI lag)", fg=TEXT_SEC,
                 bg=BG_CARD, font=("Segoe UI", 8)).pack(side="left")

        tk.Label(cfg_frame, text="Decoder-only Transformer · GQA · YaRN RoPE · Any text file",
                 fg=TEXT_SEC, bg=BG_CARD, font=("Segoe UI", 8),
                 wraplength=340, padx=8, pady=4, anchor="w", justify="left").pack(fill="x", padx=8, pady=(0, 4))

        # ── Expandable Template section ──
        tmpl_sec = tk.Frame(cfg_frame, bg=BG_CARD)
        tmpl_sec.pack(fill="x", padx=0, pady=(0, 8))
        self._tmpl_expanded = tk.BooleanVar(value=False)
        tmpl_header = tk.Frame(tmpl_sec, bg=BG_CARD)
        tmpl_header.pack(fill="x", padx=8)
        tmpl_toggle = tk.Label(tmpl_header, text="▶  TEMPLATE", fg=ACCENT,
                               bg=BG_CARD, font=("Segoe UI", 9, "bold"),
                               cursor="hand2")
        tmpl_toggle.pack(side="left")
        tmpl_body = tk.Frame(tmpl_sec, bg=BG_CARD)
        def _toggle_tmpl(_=None):
            if self._tmpl_expanded.get():
                tmpl_body.pack_forget()
                tmpl_toggle.config(text="▶  TEMPLATE")
                self._tmpl_expanded.set(False)
            else:
                tmpl_body.pack(fill="x", padx=16, pady=(4, 4))
                tmpl_toggle.config(text="▼  TEMPLATE")
                self._tmpl_expanded.set(True)
        tmpl_toggle.bind("<Button-1>", _toggle_tmpl)

        self._refresh_templates()
        tmpl_row = tk.Frame(tmpl_body, bg=BG_CARD)
        tmpl_row.pack(fill="x", pady=2)
        tk.Label(tmpl_row, text="Template", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9), width=12, anchor="w").pack(side="left")
        self.tmpl_combo = ttk.Combobox(tmpl_row, values=self._template_list,
                                       state="readonly", font=("Segoe UI", 9), width=22)
        self.tmpl_combo.set("step_by_step")
        self.tmpl_combo.pack(side="left", padx=4)

        # Field mapping: key → value pairs for template segment fields
        mapping_row = tk.Frame(tmpl_body, bg=BG_CARD)
        mapping_row.pack(fill="x", pady=2)
        tk.Label(mapping_row, text="Field map", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 9), width=12, anchor="w").pack(side="left")
        self.tmpl_fields_var = tk.StringVar(value="")
        tk.Entry(mapping_row, textvariable=self.tmpl_fields_var, bg=BG_INPUT, fg=TEXT_PRI,
                 insertbackground=TEXT_PRI, relief="flat",
                 font=("Segoe UI", 9), width=22).pack(side="left", padx=4)
        tk.Label(mapping_row, text="key=val,key=val", fg=TEXT_SEC, bg=BG_CARD,
                 font=("Segoe UI", 8)).pack(side="left")

    def _parse_template_fields(self):
        raw = self.tmpl_fields_var.get().strip() if hasattr(self, 'tmpl_fields_var') else ""
        if not raw:
            return None
        fields = {}
        for pair in raw.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                fields[k.strip()] = v.strip()
        return fields or None

    def _refresh_templates(self):
        try:
            from data.templates import init_templates, list_templates
            init_templates("templates")
            self._template_list = list_templates()
        except Exception:
            self._template_list = ["step_by_step", "qa", "analysis", "chat", "raw"]

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
            ("LR",       "lr",       "—"),
            ("ETA",      "eta",      "—"),
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
        setup_ttk_styles()
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
            "model_type":     "text_generation",
            "optimizer":      self.combos["Optimizer"].get(),
            "scheduler":      self.combos["Scheduler"].get(),
            "epochs":         int(self.spins["epochs"].get()),
            "batch_size":     int(self.spins["batch"].get()),
            "hidden_dim":     int(self.spins["hidden"].get()),
            "num_layers":     int(self.spins["layers"].get()),
            "num_heads":      int(self.spins["heads"].get()),
            "lr":             float(self.lr_var.get()),
            "seq_len":        max(seq_len_val, 8),
            "max_ram_mb":     int(self.max_ram_var.get()),
            "cpu_threads":    int(self.cpu_threads_var.get()),
            "num_workers":    int(self.num_workers_var.get()),
            "use_subprocess": self.subprocess_var.get(),
            "prompt_ratio":   0.3,
            "template_name":  self.tmpl_combo.get() if hasattr(self, 'tmpl_combo') else "step_by_step",
            "template_fields": self._parse_template_fields(),
            "resume_path":    None,
        }

    # ── Training simulation / real run ───────────────────────────────────────
    def _start(self):
        files = self.get_files()
        cfg = self._get_config()

        if not files:
            messagebox.showwarning(
                "No Data",
                "Add training files (.txt, .csv, .json, .jsonl) via the Data panel first."
            )
            return
        ok, reason = self.controller.validate_runtime_config(cfg)
        if not ok:
            messagebox.showwarning("Unsafe Runtime Configuration", reason)
            self._log(f"[TRN-CONFIG-001] {reason}", "warn")
            return
        cfg = self._get_config()
        cfg["resume_path"] = None
        self._begin_training(cfg, files)

    def _start_with_resume(self, resume_path, new_files, append_mode):
        cfg = self._get_config()
        files = self.get_files() if append_mode else []
        files = list(set(files + list(new_files)))
        cfg["resume_path"] = resume_path
        cfg["epochs"] = max(cfg["epochs"], 1)
        if not files:
            messagebox.showwarning("No Data", "No training files available.")
            return
        self._log(f"Resuming from {Path(resume_path).name} | "
                  f"{len(files)} files ({'append' if append_mode else 'replace'})", "info")
        self._begin_training(cfg, files)

    def _begin_training(self, cfg, files):
        self._stop_flag.clear()
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._log(f"Starting training — {cfg['template_name']} | "
                  f"{len(files)} files | {cfg['epochs']} epochs", "info")
        self._thread = threading.Thread(target=self._run_training,
                                        args=(cfg, files), daemon=True)
        self._thread.start()

    def _stop(self):
        self._stop_flag.set()
        self._log("Stop requested…", "warn")

    def _run_training(self, cfg, files):
        train_config = TrainConfig(
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg.get("num_kv_heads"),
            seq_len=cfg.get("seq_len", 128) or 128,
            optimizer=cfg["optimizer"],
            scheduler=cfg["scheduler"],
            use_amp=cfg.get("use_amp", True),
            grad_accum_steps=cfg.get("grad_accum_steps", 1),
            max_grad_norm=cfg.get("max_grad_norm", 1.0),
            use_gradient_checkpointing=cfg.get("use_gradient_checkpointing", False),
            ema_decay=cfg.get("ema_decay", 0.0),
            weight_decay=cfg.get("weight_decay", 0.01),
            warmup_steps=cfg.get("warmup_steps", 100),
            max_ram_mb=cfg.get("max_ram_mb", 0),
            num_workers=cfg.get("num_workers", 0),
            pin_memory=cfg.get("pin_memory", False),
            cpu_threads=cfg.get("cpu_threads", 0),
            use_subprocess=cfg.get("use_subprocess", False),
            prompt_ratio=cfg.get("prompt_ratio", 0.3),
            template_name=cfg.get("template_name", "step_by_step"),
            template_fields=cfg.get("template_fields"),
            resume_path=cfg.get("resume_path"),
            resume_freeze_layers=cfg.get("resume_freeze_layers", 0),
            resume_freeze_embeddings=cfg.get("resume_freeze_embeddings", False),
        )
        
        # Create trainer
        trainer = UnifiedTrainer(
            config=train_config,
            files=files,
            progress_callback=self._on_training_progress,
            log_callback=self._on_training_log,
        )
        trainer.set_stop_flag(self._stop_flag)
        
        # Run training
        try:
            result = trainer.run()
            if result:
                final_stats = {
                    "loss": self.stat_vars["loss"].get(),
                }
                checkpoint_name = self.controller.save_checkpoint(
                    cfg, result.model, result.best_state, result.info, final_stats,
                    tokenizer=result.tokenizer,
                    files_used=getattr(trainer, '_saved_files', []) + files,
                    resume_count=getattr(trainer, '_resume_count', 0),
                )
                self._ui(self._log, f"Checkpoint saved → {checkpoint_name}.pt", "ok")
                if self.on_checkpoint_saved:
                    self._ui(self.on_checkpoint_saved)
            else:
                self._ui(self._log, "Training ended without a result (check log above).", "warn")
        except TrainingRuntimeError as e:
            self._ui(self._log, f"Training error: [{e.code}] {e.message}", "err")
        except Exception as e:
            self._ui(self._log, f"Training error: [TRN-UNKWN-001] {e}", "err")
        finally:
            self._ui(self._finish_training)

    def _on_training_progress(self, epoch=None, epochs=None, loss=None, accuracy=None,
                             lr=None, eta=None, stage="", pct=None, ram_info=""):
        if epoch is not None and epochs is not None:
            self._ui(self._update_stats,
                    epoch, epochs, loss or 0,
                    accuracy or 0, lr or 0, eta or "", 0.0, pct or 0, ram_info or "")

    def _update_stats(self, epoch, epochs, loss, accuracy, lr, eta, val_loss, pct, ram_info=""):
        if hasattr(self, "progress_var"):
            try:
                p = float(pct or 0)
                if p <= 1.0:
                    p *= 100.0
                self.progress_var.set(min(100.0, max(0.0, p)))
            except (TypeError, ValueError):
                self.progress_var.set(0.0)
        if hasattr(self, "stat_vars"):
            self.stat_vars["epoch"].set(f"{epoch}/{epochs}")
            try:
                self.stat_vars["loss"].set(f"{float(loss):.4f}")
            except (TypeError, ValueError):
                self.stat_vars["loss"].set(str(loss))
            try:
                self.stat_vars["lr"].set(f"{float(lr):.6g}")
            except (TypeError, ValueError):
                self.stat_vars["lr"].set(str(lr))
            self.stat_vars["eta"].set(str(eta) if eta else "—")
        if hasattr(self, 'prog_lbl'):
            eta_str = f" · ETA {eta}" if eta else ""
            ram_str = f" · {ram_info}" if ram_info else ""
            self.prog_lbl.config(text=f"Epoch {epoch}/{epochs}{eta_str}{ram_str}")
        if hasattr(self, 'loss_chart') and loss:
            self.loss_chart.push(loss)
    
    def _on_training_log(self, msg, level="info"):
        """Handle training log messages."""
        self._ui(self._log, msg, level)

    def _finish_training(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

# ─── Model Manager Panel (right column) ──────────────────────────────────────

class ModelManagerPanel(tk.Frame):
    def __init__(self, parent, resume_callback=None, **kw):
        super().__init__(parent, bg=BG_PANEL, **kw)
        self.models = []
        self.selected = None
        self.resume_callback = resume_callback
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
        accent_btn(act_row, "↻ Resume Training", self._resume_training,
                   color="#2563eb", width=16).pack(side="left", padx=(0, 6))
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
            f"Loss:        {m.get('loss','—')}",
            f"Epochs:      {cfg.get('epochs','—')}",
            f"Hidden Dim:  {cfg.get('hidden_dim','—')}",
            f"Layers:      {cfg.get('num_layers','—')}",
            f"Heads:       {cfg.get('num_heads','—')}",
            f"Seq Len:     {cfg.get('seq_len','—')}",
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

    def _resume_training(self):
        if not self.selected:
            messagebox.showinfo("No Selection", "Select a model first.")
            return
        wf = self.selected.get("weights_file")
        if not wf or not Path(wf).exists():
            messagebox.showwarning("No Weights",
                f"Weights file not found:\n{wf}")
            return

        # Pick new data files to train on
        paths = filedialog.askopenfilenames(
            title="Select additional training data",
            filetypes=[("Text/Data", "*.txt *.csv *.json *.jsonl")])
        if not paths:
            return

        # Radio dialog for Append vs Replace
        d = tk.Toplevel(self)
        d.title("Resume Mode")
        d.geometry("320x180")
        d.configure(bg=BG_CARD)
        d.transient(self)
        d.grab_set()
        mode_var = tk.StringVar(value="append")
        tk.Label(d, text="Data mode for resumed training:", fg=TEXT_PRI,
                 bg=BG_CARD, font=("Segoe UI", 10)).pack(pady=(16, 8))
        tk.Radiobutton(d, text="Append — train on old + new data",
                       variable=mode_var, value="append",
                       bg=BG_CARD, fg=TEXT_PRI, selectcolor=BG_INPUT,
                       font=("Segoe UI", 9)).pack(anchor="w", padx=32, pady=2)
        tk.Radiobutton(d, text="Replace — train on only new data",
                       variable=mode_var, value="replace",
                       bg=BG_CARD, fg=TEXT_PRI, selectcolor=BG_INPUT,
                       font=("Segoe UI", 9)).pack(anchor="w", padx=32, pady=2)
        btn_row = tk.Frame(d, bg=BG_CARD)
        btn_row.pack(fill="x", padx=32, pady=(12, 0))
        def _confirm():
            d.destroy()
            if self.resume_callback:
                self.resume_callback(wf, list(paths), mode_var.get() == "append")
        tk.Button(btn_row, text="Start Resume", command=_confirm,
                  bg=ACCENT, fg=TEXT_PRI, relief="flat",
                  font=("Segoe UI", 9)).pack(side="left", padx=(0, 8))
        tk.Button(btn_row, text="Cancel", command=d.destroy,
                  bg="#444", fg=TEXT_PRI, relief="flat",
                  font=("Segoe UI", 9)).pack(side="left")

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
# ─── Main Application Window ──────────────────────────────────────────────────

class TrainingApp(tk.Tk):
    def __init__(self, skip_init=False):
        if not skip_init:
            super().__init__()
        self.title("ML Training System — Decoder-Only Transformer (GQA · YaRN · BPE)")
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
        tk.Label(bar, text="Decoder-Only Transformer · GQA · YaRN RoPE · BPE Tokenizer",
                 fg=TEXT_SEC, bg=BG_PANEL,
                 font=("Segoe UI", 9)).pack(side="left")

        # Right-side toolbar buttons
        ghost_btn(bar, "Health Check",
                  lambda: HealthPanel(self), width=12).pack(side="right",
                                                             padx=8, pady=8)
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
        self.model_panel = ModelManagerPanel(body, resume_callback=self._on_resume)
        self.model_panel.pack(side="left", fill="y", padx=(4, 0))
        self.model_panel.config(width=340)
        self.model_panel.pack_propagate(False)

        # Wire model panel refresh after training
        self.train_panel.on_checkpoint_saved = lambda: self.after(500, self.model_panel._refresh)

    def _on_resume(self, resume_path, new_files, append_mode):
        """Handle resume training from ModelManagerPanel."""
        if not self.train_panel.running:
            self.train_panel._start_with_resume(resume_path, new_files, append_mode)

    def _open_models_dir(self):
        d = Path("trained_models")
        d.mkdir(exist_ok=True)
        import subprocess, sys
        p = str(d.resolve())
        if sys.platform == "win32":
            os.startfile(p)
        elif sys.platform == "darwin":
            subprocess.run(["open", p])
        else:
            subprocess.run(["xdg-open", p])


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
