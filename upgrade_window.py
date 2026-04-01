"""
Auto-Upgrade System Window
Standalone tkinter Toplevel — can be opened from training_ui.py or start.py
Shows: live log, DB upgrade history, model snapshots, LLM conversation, controls
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import json
from datetime import datetime
from pathlib import Path

# ── palette (matches training_ui) ────────────────────────────────────────────
BG_DARK   = "#0d1117"
BG_PANEL  = "#161b22"
BG_CARD   = "#1c2128"
BG_INPUT  = "#21262d"
ACCENT    = "#238636"
ACCENT2   = "#1f6feb"
BORDER    = "#30363d"
TEXT_PRI  = "#e6edf3"
TEXT_SEC  = "#8b949e"
TEXT_WARN = "#d29922"
TEXT_ERR  = "#f85149"
TEXT_OK   = "#3fb950"


def _label(parent, text, fg=TEXT_PRI, bg=BG_PANEL, font=("Segoe UI", 9), **kw):
    return tk.Label(parent, text=text, fg=fg, bg=bg, font=font, **kw)


def _btn(parent, text, cmd, color=BG_INPUT, fg=TEXT_SEC, width=14):
    b = tk.Button(parent, text=text, command=cmd, bg=color, fg=fg,
                  activebackground=BORDER, activeforeground=TEXT_PRI,
                  relief="flat", cursor="hand2",
                  font=("Segoe UI", 9, "bold"), width=width, pady=5)
    return b


def _accent_btn(parent, text, cmd, width=16):
    b = _btn(parent, text, cmd, color=ACCENT, fg=TEXT_PRI, width=width)
    b.bind("<Enter>", lambda e: b.config(bg="#2ea043"))
    b.bind("<Leave>", lambda e: b.config(bg=ACCENT))
    return b


class AutoUpgradeWindow(tk.Toplevel):
    """
    Dedicated window for the Auto-Upgrade System.
    Pass an AutoUpgradeSystem instance (or None for DB-only view).
    """

    def __init__(self, parent, upgrade_system=None):
        super().__init__(parent)
        self.upgrade_system = upgrade_system
        self.title("Auto-Upgrade System")
        self.configure(bg=BG_DARK)
        self.geometry("1100x720")
        self.minsize(900, 600)
        self._build()
        self._refresh_all()

    def _get_db(self):
        """
        Get UpgradeDB — works in all launch modes because auto_upgrade.py
        now handles its own imports without needing package context.
        """
        import sys
        # 1. Prefer the already-loaded module (fastest, avoids re-exec)
        for key in ("mlsystem.core.auto_upgrade", "auto_upgrade",
                    "auto_upgrade_direct"):
            mod = sys.modules.get(key)
            if mod and hasattr(mod, "UpgradeDB"):
                return mod.UpgradeDB()

        # 2. Live system has a db object directly
        if self.upgrade_system and hasattr(self.upgrade_system, "db"):
            return self.upgrade_system.db

        # 3. Direct import — now works because relative imports are guarded
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "auto_upgrade_direct",
            Path(__file__).parent / "auto_upgrade.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["auto_upgrade_direct"] = mod
        spec.loader.exec_module(mod)
        return mod.UpgradeDB()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        # Title bar
        bar = tk.Frame(self, bg=BG_PANEL, height=46)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        _label(bar, "⚙  Auto-Upgrade System", fg=TEXT_PRI, bg=BG_PANEL,
               font=("Segoe UI", 12, "bold")).pack(side="left", padx=14, pady=10)
        _label(bar, "Groq llama-3.3-70b-versatile  ·  SQLite DB",
               fg=TEXT_SEC, bg=BG_PANEL, font=("Segoe UI", 8)).pack(
            side="left", pady=10)
        # Show whether a live system is attached
        sys_status = "● Live system attached" if self.upgrade_system else "○ DB view only — launch via start.py --ui for live upgrades"
        sys_color  = TEXT_OK if self.upgrade_system else TEXT_WARN
        _label(bar, sys_status, fg=sys_color, bg=BG_PANEL,
               font=("Segoe UI", 8)).pack(side="right", padx=14, pady=10)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # Control row
        ctrl = tk.Frame(self, bg=BG_DARK)
        ctrl.pack(fill="x", padx=12, pady=8)

        self.run_btn = _accent_btn(ctrl, "▶  Run Upgrade Cycle", self._run_cycle, width=20)
        self.run_btn.pack(side="left", padx=(0, 8))

        _btn(ctrl, "Analyze Only", self._analyze_only, width=14).pack(side="left", padx=(0, 8))
        _btn(ctrl, "⟳ Refresh DB", self._refresh_all, width=14).pack(side="left", padx=(0, 8))
        _btn(ctrl, "Clear Log", self._clear_log, width=12).pack(side="left")

        self.status_lbl = _label(ctrl, "Ready", fg=TEXT_SEC, bg=BG_DARK,
                                  font=("Segoe UI", 9))
        self.status_lbl.pack(side="right", padx=8)

        # Stat cards
        cards_row = tk.Frame(self, bg=BG_DARK)
        cards_row.pack(fill="x", padx=12, pady=(0, 8))
        self._stat_vars = {}
        for i, (title, key) in enumerate([
            ("Attempted", "attempted"), ("Successful", "successful"),
            ("Success Rate", "rate"), ("DB Records", "db_records"),
        ]):
            card = tk.Frame(cards_row, bg=BG_CARD,
                            highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=0, column=i, padx=4, sticky="ew")
            cards_row.columnconfigure(i, weight=1)
            _label(card, title, fg=TEXT_SEC, bg=BG_CARD,
                   font=("Segoe UI", 7, "bold")).pack(pady=(6, 0))
            v = tk.StringVar(value="—")
            self._stat_vars[key] = v
            color = {
                "attempted": TEXT_PRI, "successful": TEXT_OK,
                "rate": ACCENT2, "db_records": TEXT_WARN,
            }[key]
            _label(card, "", fg=color, bg=BG_CARD,
                   font=("Segoe UI", 13, "bold"),
                   textvariable=v).pack(pady=(0, 6))

        # Notebook tabs
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("Dark.TNotebook.Tab", background=BG_INPUT,
                        foreground=TEXT_SEC, padding=[12, 5],
                        font=("Segoe UI", 9))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", BG_CARD)],
                  foreground=[("selected", TEXT_PRI)])

        nb = ttk.Notebook(self, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self._build_log_tab(nb)
        self._build_history_tab(nb)
        self._build_snapshots_tab(nb)
        self._build_llm_tab(nb)
        self._build_modifications_tab(nb)

    # ── Tab: Live Log ─────────────────────────────────────────────────────────

    def _build_log_tab(self, nb):
        frame = tk.Frame(nb, bg=BG_CARD)
        nb.add(frame, text="  Live Log  ")
        self.log_text = tk.Text(frame, bg=BG_CARD, fg=TEXT_PRI,
                                font=("Consolas", 9), relief="flat",
                                state="disabled", wrap="word")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        self.log_text.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.log_text.tag_config("ok",   foreground=TEXT_OK)
        self.log_text.tag_config("warn", foreground=TEXT_WARN)
        self.log_text.tag_config("err",  foreground=TEXT_ERR)
        self.log_text.tag_config("info", foreground=ACCENT2)

    # ── Tab: Upgrade History ──────────────────────────────────────────────────

    def _build_history_tab(self, nb):
        frame = tk.Frame(nb, bg=BG_CARD)
        nb.add(frame, text="  Upgrade History  ")
        cols = ("ID", "Time", "Type", "Description", "Status", "Score Before", "Score After")
        self.hist_tree = ttk.Treeview(frame, columns=cols, show="headings", height=18)
        style = ttk.Style()
        style.configure("Hist.Treeview", background=BG_CARD, foreground=TEXT_PRI,
                        fieldbackground=BG_CARD, font=("Segoe UI", 8))
        style.configure("Hist.Treeview.Heading", background=BG_INPUT,
                        foreground=TEXT_SEC, font=("Segoe UI", 8, "bold"))
        style.map("Hist.Treeview", background=[("selected", ACCENT2)])
        self.hist_tree.configure(style="Hist.Treeview")
        widths = {"ID": 40, "Time": 130, "Type": 140, "Description": 260,
                  "Status": 70, "Score Before": 90, "Score After": 90}
        for col in cols:
            self.hist_tree.heading(col, text=col)
            self.hist_tree.column(col, width=widths[col], anchor="w")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.hist_tree.yview)
        self.hist_tree.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.hist_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.hist_tree.tag_configure("success", foreground=TEXT_OK)
        self.hist_tree.tag_configure("failed",  foreground=TEXT_ERR)
        self.hist_tree.tag_configure("pending", foreground=TEXT_WARN)

    # ── Tab: Model Snapshots ──────────────────────────────────────────────────

    def _build_snapshots_tab(self, nb):
        frame = tk.Frame(nb, bg=BG_CARD)
        nb.add(frame, text="  Model Snapshots  ")
        cols = ("ID", "Time", "Label", "Params", "Notes")
        self.snap_tree = ttk.Treeview(frame, columns=cols, show="headings", height=18)
        self.snap_tree.configure(style="Hist.Treeview")
        for col, w in zip(cols, [40, 140, 120, 90, 400]):
            self.snap_tree.heading(col, text=col)
            self.snap_tree.column(col, width=w, anchor="w")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.snap_tree.yview)
        self.snap_tree.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.snap_tree.pack(fill="both", expand=True, padx=4, pady=4)

    # ── Tab: LLM Conversation ─────────────────────────────────────────────────

    def _build_llm_tab(self, nb):
        frame = tk.Frame(nb, bg=BG_CARD)
        nb.add(frame, text="  LLM Conversation  ")
        self.llm_text = tk.Text(frame, bg=BG_CARD, fg=TEXT_PRI,
                                font=("Consolas", 8), relief="flat",
                                state="disabled", wrap="word")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.llm_text.yview)
        self.llm_text.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.llm_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.llm_text.tag_config("user",      foreground=ACCENT2)
        self.llm_text.tag_config("assistant", foreground=TEXT_OK)
        self.llm_text.tag_config("ts",        foreground=TEXT_SEC)

    # ── Tab: Self-Modifications ───────────────────────────────────────────────

    def _build_modifications_tab(self, nb):
        frame = tk.Frame(nb, bg=BG_CARD)
        nb.add(frame, text="  Self-Modifications  ")
        cols = ("ID", "Time", "File", "Function", "Applied", "Reason")
        self.mod_tree = ttk.Treeview(frame, columns=cols, show="headings", height=10)
        self.mod_tree.configure(style="Hist.Treeview")
        for col, w in zip(cols, [40, 140, 100, 120, 60, 340]):
            self.mod_tree.heading(col, text=col)
            self.mod_tree.column(col, width=w, anchor="w")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.mod_tree.yview)
        self.mod_tree.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.mod_tree.pack(fill="both", expand=True, padx=4, pady=4)

        # Detail pane
        detail_frame = tk.Frame(frame, bg=BG_CARD)
        detail_frame.pack(fill="x", padx=4, pady=(0, 4))
        _label(detail_frame, "New Code:", fg=TEXT_SEC, bg=BG_CARD,
               font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=4)
        self.mod_code = tk.Text(detail_frame, bg=BG_INPUT, fg=TEXT_PRI,
                                font=("Consolas", 8), relief="flat", height=8)
        self.mod_code.pack(fill="x", padx=4, pady=(0, 4))
        self.mod_tree.bind("<<TreeviewSelect>>", self._on_mod_select)
        self.mod_tree.tag_configure("applied",   foreground=TEXT_OK)
        self.mod_tree.tag_configure("unapplied", foreground=TEXT_WARN)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _run_cycle(self):
        if not self.upgrade_system:
            self._log(
                "No live upgrade system — launch via: python start.py --ui\n"
                "DB history is still viewable in the tabs above.", "warn")
            return
        self.run_btn.config(state="disabled")
        self._set_status("Running upgrade cycle…", TEXT_WARN)
        self.upgrade_system.set_log_callback(self._on_upgrade_log)

        def _worker():
            try:
                summary = self.upgrade_system.run_full_cycle()
                self.after(0, self._on_cycle_done, summary)
            except Exception as e:
                self.after(0, self._on_cycle_error, str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _analyze_only(self):
        if not self.upgrade_system:
            self._log(
                "No live upgrade system — launch via: python start.py --ui", "warn")
            return
        self._set_status("Analyzing…", TEXT_WARN)

        def _worker():
            try:
                report = self.upgrade_system.analyze_performance()
                self.after(0, self._log,
                    f"Analysis complete — score: {report.get('overall_score',0):.1f}/100\n"
                    f"Bottlenecks: {len(report.get('bottlenecks',[]))}\n"
                    f"Opportunities: {len(report.get('opportunities',[]))}\n"
                    f"Params: {report.get('param_count',0):,}", "info")
                self.after(0, self._set_status, "Analysis done", TEXT_OK)
            except Exception as e:
                self.after(0, self._log, f"Analysis error: {e}", "err")
                self.after(0, self._set_status, "Error", TEXT_ERR)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_cycle_done(self, summary):
        self.run_btn.config(state="normal")
        self._set_status(
            f"Done — {summary['applied']}/{summary['fetched']} applied", TEXT_OK)
        self._refresh_all()

    def _on_cycle_error(self, err):
        self.run_btn.config(state="normal")
        self._log(f"Cycle error: {err}", "err")
        self._set_status("Error", TEXT_ERR)

    def _on_upgrade_log(self, msg: str, level: str = "info"):
        self.after(0, self._log, msg, level)

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    # ── Refresh DB views ──────────────────────────────────────────────────────

    def _refresh_all(self):
        try:
            db = self._get_db()
            self._refresh_history(db)
            self._refresh_snapshots(db)
            self._refresh_llm(db)
            self._refresh_modifications(db)
            self._refresh_stats(db)
        except Exception as e:
            self._log(f"DB refresh error: {e}", "err")

    def _refresh_history(self, db):
        self.hist_tree.delete(*self.hist_tree.get_children())
        for row in db.get_recent_upgrades(100):
            ts   = row["ts"][:19] if row["ts"] else ""
            pb   = f"{row['perf_before']:.1f}" if row["perf_before"] else "—"
            pa   = f"{row['perf_after']:.1f}"  if row["perf_after"]  else "—"
            tag  = row["status"] if row["status"] in ("success","failed","pending") else "pending"
            self.hist_tree.insert("", "end", tags=(tag,),
                values=(row["id"], ts, row["upgrade_type"],
                        (row["description"] or "")[:60], row["status"], pb, pa))

    def _refresh_snapshots(self, db):
        self.snap_tree.delete(*self.snap_tree.get_children())
        for row in db.get_snapshots(50):
            ts = row["ts"][:19] if row["ts"] else ""
            self.snap_tree.insert("", "end",
                values=(row["id"], ts, row["label"],
                        f"{row['param_count']:,}" if row["param_count"] else "—",
                        (row["notes"] or "")[:80]))

    def _refresh_llm(self, db):
        self.llm_text.config(state="normal")
        self.llm_text.delete("1.0", "end")
        for row in db.get_conversation(60):
            ts   = row["ts"][:19] if row["ts"] else ""
            role = row["role"]
            tag  = "user" if role == "user" else "assistant"
            self.llm_text.insert("end", f"[{ts}] ", "ts")
            self.llm_text.insert("end", f"{role.upper()}\n", tag)
            self.llm_text.insert("end", row["content"][:600] + "\n\n")
        self.llm_text.see("end")
        self.llm_text.config(state="disabled")

    def _refresh_modifications(self, db):
        self.mod_tree.delete(*self.mod_tree.get_children())
        for row in db.get_modifications():
            ts  = row["ts"][:19] if row["ts"] else ""
            app = "✓" if row["applied"] else "○"
            tag = "applied" if row["applied"] else "unapplied"
            self.mod_tree.insert("", "end", iid=str(row["id"]), tags=(tag,),
                values=(row["id"], ts, row["target_file"],
                        row["function_name"], app,
                        (row["reason"] or "")[:60]))

    def _refresh_stats(self, db):
        rows = db.get_recent_upgrades(1000)
        total    = len(rows)
        success  = sum(1 for r in rows if r["status"] == "success")
        rate     = f"{success/max(1,total)*100:.0f}%"
        self._stat_vars["attempted"].set(str(total))
        self._stat_vars["successful"].set(str(success))
        self._stat_vars["rate"].set(rate)
        self._stat_vars["db_records"].set(str(total))
        if self.upgrade_system:
            s = self.upgrade_system.get_upgrade_status()
            self._stat_vars["attempted"].set(str(s["attempted"]))
            self._stat_vars["successful"].set(str(s["successful"]))
            self._stat_vars["rate"].set(f"{s['success_rate']*100:.0f}%")

    def _on_mod_select(self, _=None):
        sel = self.mod_tree.selection()
        if not sel:
            return
        try:
            db   = self._get_db()
            rows = db.get_modifications()
            row  = next((r for r in rows if str(r["id"]) == sel[0]), None)
            if row:
                self.mod_code.config(state="normal")
                self.mod_code.delete("1.0", "end")
                self.mod_code.insert("end", row["new_code"] or "")
                self.mod_code.config(state="disabled")
        except Exception:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str, level: str = ""):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_text.config(state="normal")
        self.log_text.insert("end", line, level)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _set_status(self, msg: str, color: str = TEXT_SEC):
        self.status_lbl.config(text=msg, fg=color)
