import tkinter as tk
from tkinter import ttk, messagebox
import threading
from pathlib import Path
from .theme import (
    BG_DARK, BG_PANEL, BG_CARD, BG_INPUT, ACCENT2, BORDER,
    TEXT_PRI, TEXT_SEC, TEXT_WARN, TEXT_ERR, TEXT_OK,
    styled_frame, label, section_title, ghost_btn, accent_btn
)
from core.implementations import GenConfig


class InferenceWindow(tk.Toplevel):
    """Text generation inference UI."""

    def __init__(self, parent, model_meta: dict):
        super().__init__(parent)
        self.model_meta = model_meta
        self.title(f"Inference — {model_meta['name']}")
        self.configure(bg=BG_DARK)
        self.geometry("780x680")
        self.minsize(680, 560)
        self._build()

    def _ui(self, fn, *args):
        try:
            if self.winfo_exists():
                self.after(0, fn, *args)
        except Exception:
            pass

    def _build(self):
        hdr = tk.Frame(self, bg=BG_PANEL, height=46)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        label(hdr, "▶  Inference", fg=TEXT_PRI, bg=BG_PANEL,
              font=("Segoe UI", 12, "bold")).pack(side="left", padx=14, pady=10)
        label(hdr, self.model_meta["name"], fg=TEXT_SEC, bg=BG_PANEL,
              font=("Segoe UI", 9)).pack(side="left")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        info_frame = styled_frame(self, bg=BG_CARD)
        info_frame.pack(fill="x", padx=12, pady=(10, 6))
        cfg = self.model_meta.get("config", {})
        info_lines = [
            ("Type",       self.model_meta.get("model_type", "—")),
            ("Loss",       self.model_meta.get("loss", "—")),
            ("Epochs",     str(cfg.get("epochs", "—"))),
            ("Hidden Dim", str(cfg.get("hidden_dim", "—"))),
            ("Layers",     str(cfg.get("num_layers", "—"))),
            ("Heads",      str(cfg.get("num_heads", "—"))),
        ]
        row_f = tk.Frame(info_frame, bg=BG_CARD)
        row_f.pack(fill="x", padx=10, pady=8)
        for i, (k, v) in enumerate(info_lines):
            col = tk.Frame(row_f, bg=BG_CARD)
            col.grid(row=0, column=i, padx=10)
            row_f.columnconfigure(i, weight=1)
            label(col, k, fg=TEXT_SEC, bg=BG_CARD, font=("Segoe UI", 7, "bold")).pack()
            label(col, v, fg=TEXT_PRI, bg=BG_CARD, font=("Segoe UI", 10, "bold")).pack()

        pick_frame = styled_frame(self, bg=BG_CARD)
        pick_frame.pack(fill="x", padx=12, pady=(0, 6))
        section_title(pick_frame, "  PROMPT", bg=BG_CARD).pack(anchor="w", padx=8, pady=(8, 4))
        self._prompt_var = tk.StringVar(value="Once upon a time")
        tk.Entry(pick_frame, textvariable=self._prompt_var,
                 bg=BG_INPUT, fg=TEXT_PRI, insertbackground=TEXT_PRI,
                 relief="flat", font=("Segoe UI", 10)).pack(fill="x", padx=8, pady=(0, 8))

        opt_frame = styled_frame(self, bg=BG_CARD)
        opt_frame.pack(fill="x", padx=12, pady=(0, 6))
        section_title(opt_frame, "  OPTIONS", bg=BG_CARD).pack(anchor="w", padx=8, pady=(8, 4))
        opt_row = tk.Frame(opt_frame, bg=BG_CARD)
        opt_row.pack(fill="x", padx=8, pady=(0, 8))

        for lbl, var_name, default, lo, hi, inc in [
            ("Temperature", "_temp_var",  "0.8",  "0.1", "2.0", "0.1"),
            ("Top-K",       "_topk_var",  "40",   "1",   "200", "5"),
            ("Generate N",  "_ngen_var",  "300",  "10",  "2000","50"),
        ]:
            label(opt_row, f"{lbl}:", fg=TEXT_SEC, bg=BG_CARD, font=("Segoe UI", 9)).pack(side="left", padx=(0, 2))
            v = tk.StringVar(value=default)
            setattr(self, var_name, v)
            tk.Spinbox(opt_row, from_=float(lo), to=float(hi), increment=float(inc),
                       textvariable=v, bg=BG_INPUT, fg=TEXT_PRI, font=("Segoe UI", 9),
                       width=6, buttonbackground=BG_INPUT).pack(side="left", padx=(0, 14))

        btn_row = tk.Frame(self, bg=BG_DARK)
        btn_row.pack(fill="x", padx=12, pady=(0, 6))
        self.run_btn = accent_btn(btn_row, "▶  Generate", self._run, color=ACCENT2, width=18)
        self.run_btn.pack(side="left")
        self.status_lbl = label(btn_row, "Ready", fg=TEXT_SEC, bg=BG_DARK, font=("Segoe UI", 9))
        self.status_lbl.pack(side="left", padx=12)

        res_frame = styled_frame(self, bg=BG_CARD)
        res_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        section_title(res_frame, "  OUTPUT", bg=BG_CARD).pack(anchor="w", padx=8, pady=(8, 4))
        self.result_text = tk.Text(res_frame, bg=BG_CARD, fg=TEXT_PRI, font=("Consolas", 9),
                                   relief="flat", state="disabled", wrap="word")
        sb = ttk.Scrollbar(res_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.result_text.pack(fill="both", expand=True, padx=6, pady=(0, 8))

    def _run(self):
        self.run_btn.config(state="disabled")
        self.status_lbl.config(text="Generating…", fg=TEXT_WARN)
        self._write("Generating…\n", "section")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            from core.text_model import load_lm
            model, tokenizer = load_lm(self.model_meta["weights_file"])
            prompt = self._prompt_var.get() or "\n"

            cfg = GenConfig(
                temperature=float(self._temp_var.get()),
                top_k=int(self._topk_var.get()),
                max_new_tokens=int(self._ngen_var.get()),
                eos_id=3,
            )
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            new_ids = model.generate(tokenizer.encode(prompt), cfg=cfg, device=device)
            output = tokenizer.decode(new_ids, skip_special=True)
            self._ui(self._show_output, prompt, output)
        except Exception as e:
            self._ui(self._show_error, str(e))

    def _show_output(self, prompt, generated):
        self.run_btn.config(state="normal")
        self.status_lbl.config(text="Done", fg=TEXT_OK)
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", f"PROMPT:\n{prompt}\n\nGENERATED:\n{generated}")
        self.result_text.config(state="disabled")

    def _show_error(self, err):
        self.run_btn.config(state="normal")
        self.status_lbl.config(text="Error", fg=TEXT_ERR)
        self._write(f"\nError: {err}\n", "warn")

    def _write(self, text, tag=""):
        self.result_text.config(state="normal")
        self.result_text.insert("end", text, tag)
        self.result_text.config(state="disabled")