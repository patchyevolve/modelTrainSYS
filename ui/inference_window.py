import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
from ui.theme import (
    BG_DARK, BG_PANEL, BG_CARD, BG_INPUT, ACCENT2, BORDER, 
    TEXT_PRI, TEXT_SEC, TEXT_WARN, TEXT_ERR, TEXT_OK,
    styled_frame, label, section_title, ghost_btn, accent_btn
)

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
        try:
            if self.winfo_exists():
                self.after(0, fn, *args)
        except Exception:
            pass

    def _build(self):
        hdr = tk.Frame(self, bg=BG_PANEL, height=46)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        label(hdr, f"▶  Inference", fg=TEXT_PRI, bg=BG_PANEL,
              font=("Segoe UI", 12, "bold")).pack(side="left", padx=14, pady=10)
        label(hdr, self.model_meta["name"], fg=TEXT_SEC, bg=BG_PANEL,
              font=("Segoe UI", 9)).pack(side="left")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

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
            label(col, k, fg=TEXT_SEC, bg=BG_CARD, font=("Segoe UI", 7, "bold")).pack()
            label(col, v, fg=TEXT_PRI, bg=BG_CARD, font=("Segoe UI", 10, "bold")).pack()

        self._is_lm = self.model_meta.get("task") == "language_model"
        pick_frame = styled_frame(self, bg=BG_CARD)
        pick_frame.pack(fill="x", padx=12, pady=(0, 6))

        if self._is_lm:
            section_title(pick_frame, "  PROMPT", bg=BG_CARD).pack(anchor="w", padx=8, pady=(8, 4))
            self._prompt_var = tk.StringVar(value="Once upon a time")
            tk.Entry(pick_frame, textvariable=self._prompt_var,
                     bg=BG_INPUT, fg=TEXT_PRI, insertbackground=TEXT_PRI,
                     relief="flat", font=("Segoe UI", 10)).pack(fill="x", padx=8, pady=(0, 8))
        else:
            section_title(pick_frame, "  DATA FILE", bg=BG_CARD).pack(anchor="w", padx=8, pady=(8, 4))
            pick_row = tk.Frame(pick_frame, bg=BG_CARD)
            pick_row.pack(fill="x", padx=8, pady=(0, 8))
            self.data_var = tk.StringVar(value=self._default_data())
            tk.Entry(pick_row, textvariable=self.data_var, bg=BG_INPUT,
                     fg=TEXT_PRI, insertbackground=TEXT_PRI, relief="flat",
                     font=("Segoe UI", 9)).pack(side="left", fill="x", expand=True, padx=(0, 6))
            ghost_btn(pick_row, "Browse", self._browse_data, width=8).pack(side="left")

        opt_frame = styled_frame(self, bg=BG_CARD)
        opt_frame.pack(fill="x", padx=12, pady=(0, 6))
        section_title(opt_frame, "  OPTIONS", bg=BG_CARD).pack(anchor="w", padx=8, pady=(8, 4))
        opt_row = tk.Frame(opt_frame, bg=BG_CARD)
        opt_row.pack(fill="x", padx=8, pady=(0, 8))

        if self._is_lm:
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
        else:
            label(opt_row, "Threshold:", fg=TEXT_SEC, bg=BG_CARD, font=("Segoe UI", 9)).pack(side="left")
            self.thresh_var = tk.StringVar(value="0.5")
            tk.Spinbox(opt_row, from_=0.1, to=0.9, increment=0.05, textvariable=self.thresh_var, 
                       format="%.2f", bg=BG_INPUT, fg=TEXT_PRI, font=("Segoe UI", 9), 
                       width=6, buttonbackground=BG_INPUT).pack(side="left", padx=(4, 20))
            self.save_var = tk.BooleanVar(value=True)
            tk.Checkbutton(opt_row, text="Save results", variable=self.save_var, bg=BG_CARD, 
                           fg=TEXT_PRI, selectcolor=BG_INPUT).pack(side="left")

        btn_row = tk.Frame(self, bg=BG_DARK)
        btn_row.pack(fill="x", padx=12, pady=(0, 6))
        self.run_btn = accent_btn(btn_row, "▶  Run Inference", self._run, color=ACCENT2, width=18)
        self.run_btn.pack(side="left")
        self.status_lbl = label(btn_row, "Ready", fg=TEXT_SEC, bg=BG_DARK, font=("Segoe UI", 9))
        self.status_lbl.pack(side="left", padx=12)

        res_frame = styled_frame(self, bg=BG_CARD)
        res_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        section_title(res_frame, "  RESULTS", bg=BG_CARD).pack(anchor="w", padx=8, pady=(8, 4))
        self.result_text = tk.Text(res_frame, bg=BG_CARD, fg=TEXT_PRI, font=("Consolas", 9), 
                                   relief="flat", state="disabled", wrap="word")
        sb = ttk.Scrollbar(res_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.result_text.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        self.result_text.tag_config("header", foreground=ACCENT2, font=("Consolas", 9, "bold"))
        self.result_text.tag_config("good", foreground=TEXT_OK)
        self.result_text.tag_config("warn", foreground=TEXT_ERR)
        self.result_text.tag_config("section", foreground=TEXT_SEC, font=("Consolas", 8, "bold"))

    def _default_data(self):
        for p in Path("randomDATA").glob("*.csv"): return str(p)
        return ""

    def _browse_data(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("NumPy", "*.npy *.npz"), ("All", "*.*")])
        if path: self.data_var.set(path)

    def _run(self):
        data_path = "" if self._is_lm else self.data_var.get().strip()
        if not self._is_lm and (not data_path or not Path(data_path).exists()):
            messagebox.showwarning("No Data", "Select a valid data file.")
            return

        self.run_btn.config(state="disabled")
        self.status_lbl.config(text="Running…", fg=TEXT_WARN)
        self._write("Processing…\n", "section")
        
        threading.Thread(target=self._worker, args=(data_path,), daemon=True).start()

    def _worker(self, data_path):
        try:
            if self._is_lm:
                self._run_lm_inference()
            else:
                from utils.inference import load_checkpoint, run_inference, save_results
                wf = self.model_meta["weights_file"]
                model, config, data_info = load_checkpoint(wf)
                results = run_inference(model, data_info, data_path, threshold=float(self.thresh_var.get()))
                out = save_results(results, self.model_meta["name"], data_path) if self.save_var.get() else None
                self._ui(self._show_results, results, out)
        except Exception as e:
            self._ui(self._show_error, str(e))

    def _run_lm_inference(self):
        from core.text_model import load_lm
        model, tokenizer = load_lm(self.model_meta["weights_file"])
        prompt = self._prompt_var.get() or "\n"
        new_ids = model.generate(tokenizer.encode(prompt), max_new=int(self._ngen_var.get()),
                                 temperature=float(self._temp_var.get()), top_k=int(self._topk_var.get()))
        self._ui(self._show_lm_output, prompt, tokenizer.decode(new_ids))

    def _show_lm_output(self, prompt, generated):
        self.run_btn.config(state="normal")
        self.status_lbl.config(text="Done", fg=TEXT_OK)
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", f"PROMPT: {prompt}\n\nGENERATED:\n{generated}", "metric")
        self.result_text.config(state="disabled")

    def _show_results(self, r, saved_path):
        self.run_btn.config(state="normal")
        self.status_lbl.config(text=f"Done — {r['accuracy']*100:.1f}% acc", fg=TEXT_OK)
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", f"Accuracy: {r['accuracy']*100:.2f}%\n", "good")
        if saved_path: self.result_text.insert("end", f"Saved: {saved_path}", "section")
        self.result_text.config(state="disabled")

    def _show_error(self, err):
        self.run_btn.config(state="normal")
        self.status_lbl.config(text="Error", fg=TEXT_ERR)
        self._write(f"\nError: {err}\n", "warn")

    def _write(self, text, tag=""):
        self.result_text.config(state="normal")
        self.result_text.insert("end", text, tag)
        self.result_text.config(state="disabled")