"""
Auto-Upgrade System Window
Standalone tkinter Toplevel with SmartUpgradeSystem integration.
Shows: live log, project files, upgrade history, code diffs, controls.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import json
from datetime import datetime
from pathlib import Path

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
    def __init__(self, parent, upgrade_system=None):
        super().__init__(parent)
        self.smart_upgrade = None
        self.upgrade_system = upgrade_system
        self.title("Smart Upgrade System")
        self.configure(bg=BG_DARK)
        self.geometry("1200x800")
        self.minsize(1000, 600)
        self._build()
        self._refresh_all()

    def _get_smart_system(self):
        if self.smart_upgrade:
            return self.smart_upgrade
        
        from smart_upgrade import SmartUpgradeSystem
        self.smart_upgrade = SmartUpgradeSystem()
        self.smart_upgrade.set_log_callback(self._on_log)
        return self.smart_upgrade

    def _get_db(self):
        try:
            from project_context import ProjectFileDB
            return ProjectFileDB()
        except ImportError:
            return None

    def _build(self):
        bar = tk.Frame(self, bg=BG_PANEL, height=46)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        _label(bar, "Smart Upgrade System", fg=TEXT_PRI, bg=BG_PANEL,
               font=("Segoe UI", 12, "bold")).pack(side="left", padx=14, pady=10)
        _label(bar, "Full Project Analysis  ·  LLM Caching  ·  Per-File Context",
               fg=TEXT_SEC, bg=BG_PANEL, font=("Segoe UI", 8)).pack(side="left", pady=10)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        ctrl = tk.Frame(self, bg=BG_DARK)
        ctrl.pack(fill="x", padx=12, pady=8)

        self.run_btn = _accent_btn(ctrl, "Run Full Upgrade", self._run_cycle, width=18)
        self.run_btn.pack(side="left", padx=(0, 6))

        _btn(ctrl, "Analyze Project", self._analyze_project, width=14).pack(side="left", padx=(0, 6))
        _btn(ctrl, "Query LLM", self._query_llm, width=12).pack(side="left", padx=(0, 6))
        _btn(ctrl, "Refresh", self._refresh_all, width=10).pack(side="left", padx=(0, 6))
        _btn(ctrl, "Clear", self._clear_log, width=8).pack(side="left")

        self.status_lbl = _label(ctrl, "Ready", fg=TEXT_SEC, bg=BG_DARK, font=("Segoe UI", 9))
        self.status_lbl.pack(side="right", padx=8)

        cards_row = tk.Frame(self, bg=BG_DARK)
        cards_row.pack(fill="x", padx=12, pady=(0, 8))
        self._stat_vars = {}
        for i, (title, key) in enumerate([
            ("Files", "files"), ("Upgrades", "upgrades"),
            ("Success", "success"), ("API Calls", "api_calls"),
        ]):
            card = tk.Frame(cards_row, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=0, column=i, padx=4, sticky="ew")
            cards_row.columnconfigure(i, weight=1)
            _label(card, title, fg=TEXT_SEC, bg=BG_CARD, font=("Segoe UI", 7, "bold")).pack(pady=(6, 0))
            v = tk.StringVar(value="—")
            self._stat_vars[key] = v
            colors = {"files": ACCENT2, "upgrades": TEXT_PRI, "success": TEXT_OK, "api_calls": TEXT_WARN}
            _label(card, "", fg=colors[key], bg=BG_CARD, font=("Segoe UI", 13, "bold"), textvariable=v).pack(pady=(0, 6))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("Dark.TNotebook.Tab", background=BG_INPUT, foreground=TEXT_SEC, padding=[12, 5], font=("Segoe UI", 9))
        style.map("Dark.TNotebook.Tab", background=[("selected", BG_CARD)], foreground=[("selected", TEXT_PRI)])

        nb = ttk.Notebook(self, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self._build_log_tab(nb)
        self._build_files_tab(nb)
        self._build_suggestions_tab(nb)
        self._build_history_tab(nb)
        self._build_diff_tab(nb)

    def _build_log_tab(self, nb):
        frame = tk.Frame(nb, bg=BG_CARD)
        nb.add(frame, text="  Live Log  ")
        self.log_text = tk.Text(frame, bg=BG_CARD, fg=TEXT_PRI, font=("Consolas", 9), relief="flat", state="disabled", wrap="word")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        self.log_text.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.log_text.tag_config("ok", foreground=TEXT_OK)
        self.log_text.tag_config("warn", foreground=TEXT_WARN)
        self.log_text.tag_config("err", foreground=TEXT_ERR)
        self.log_text.tag_config("info", foreground=ACCENT2)

    def _build_files_tab(self, nb):
        frame = tk.Frame(nb, bg=BG_CARD)
        nb.add(frame, text="  Project Files  ")
        
        toolbar = tk.Frame(frame, bg=BG_CARD)
        toolbar.pack(fill="x", padx=4, pady=4)
        _btn(toolbar, "Analyze All", self._analyze_project, width=12).pack(side="left", padx=2)
        _btn(toolbar, "View Context", self._view_file_context, width=12).pack(side="left", padx=2)
        
        cols = ("File", "Classes", "Functions", "Lines", "Last Analyzed")
        self.files_tree = ttk.Treeview(frame, columns=cols, show="headings", height=20)
        self.files_tree.configure(style="Hist.Treeview")
        widths = {"File": 250, "Classes": 80, "Functions": 80, "Lines": 60, "Last Analyzed": 150}
        for col in cols:
            self.files_tree.heading(col, text=col)
            self.files_tree.column(col, width=widths[col], anchor="w")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.files_tree.yview)
        self.files_tree.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.files_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.files_tree.bind("<Double-Button-1>", lambda e: self._view_file_context())

    def _build_suggestions_tab(self, nb):
        frame = tk.Frame(nb, bg=BG_CARD)
        nb.add(frame, text="  Suggestions  ")
        
        toolbar = tk.Frame(frame, bg=BG_CARD)
        toolbar.pack(fill="x", padx=4, pady=4)
        self.apply_sel_btn = _btn(toolbar, "Apply Selected", self._apply_selected, width=14)
        self.apply_sel_btn.pack(side="left", padx=2)
        _btn(toolbar, "Apply All", self._apply_all, width=10).pack(side="left", padx=2)
        _btn(toolbar, "Discard All", self._discard_all, width=10).pack(side="left", padx=2)
        
        cols = ("#", "File", "Function", "Issue", "Status")
        self.sugg_tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        self.sugg_tree.configure(style="Hist.Treeview")
        widths = {"#": 30, "File": 150, "Function": 120, "Issue": 350, "Status": 80}
        for col in cols:
            self.sugg_tree.heading(col, text=col)
            self.sugg_tree.column(col, width=widths[col], anchor="w")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.sugg_tree.yview)
        self.sugg_tree.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.sugg_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.sugg_tree.bind("<Double-Button-1>", lambda e: self._view_suggestion_diff())

        detail = tk.Frame(frame, bg=BG_CARD)
        detail.pack(fill="x", padx=4, pady=4)
        _label(detail, "Details:", fg=TEXT_SEC, bg=BG_CARD, font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.sugg_detail = scrolledtext.ScrolledText(detail, bg=BG_INPUT, fg=TEXT_PRI, font=("Consolas", 8), height=10)
        self.sugg_detail.pack(fill="x")
        self.sugg_tree.bind("<<TreeviewSelect>>", self._on_sugg_select)

    def _build_history_tab(self, nb):
        frame = tk.Frame(nb, bg=BG_CARD)
        nb.add(frame, text="  Upgrade History  ")
        cols = ("ID", "Time", "File", "Function", "Status", "Reasoning")
        self.hist_tree = ttk.Treeview(frame, columns=cols, show="headings", height=18)
        self.hist_tree.configure(style="Hist.Treeview")
        widths = {"ID": 40, "Time": 130, "File": 150, "Function": 120, "Status": 80, "Reasoning": 300}
        for col in cols:
            self.hist_tree.heading(col, text=col)
            self.hist_tree.column(col, width=widths[col], anchor="w")
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.hist_tree.yview)
        self.hist_tree.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.hist_tree.pack(fill="both", expand=True, padx=4, pady=4)
        self.hist_tree.tag_configure("applied", foreground=TEXT_OK)
        self.hist_tree.tag_configure("failed", foreground=TEXT_ERR)
        self.hist_tree.tag_configure("pending", foreground=TEXT_WARN)
        self.hist_tree.tag_configure("reverted", foreground=TEXT_SEC)

        btn_row = tk.Frame(frame, bg=BG_CARD)
        btn_row.pack(fill="x", padx=4, pady=4)
        _btn(btn_row, "Revert Selected", self._revert_selected, width=14).pack(side="left", padx=2)

    def _build_diff_tab(self, nb):
        frame = tk.Frame(nb, bg=BG_CARD)
        nb.add(frame, text="  Code Diff  ")
        
        self.diff_text = scrolledtext.ScrolledText(frame, bg=BG_INPUT, fg=TEXT_PRI, font=("Consolas", 9), wrap="none")
        self.diff_text.pack(fill="both", expand=True, padx=4, pady=4)
        self.diff_text.tag_configure("old", foreground=TEXT_ERR, background="#2d1f1f")
        self.diff_text.tag_configure("new", foreground=TEXT_OK, background="#1f2d1f")
        self.diff_text.tag_configure("header", foreground=TEXT_WARN)

    def _run_cycle(self):
        self.run_btn.config(state="disabled")
        self._set_status("Running...", TEXT_WARN)
        self._log("Analyzing project and querying LLM...")

        def _worker():
            try:
                system = self._get_smart_system()
                result = system.analyze_project()
                suggestions = system.query_for_upgrades(max_upgrades=5)
                self.after(0, self._on_cycle_done, suggestions, system)
            except Exception as e:
                self.after(0, self._on_cycle_error, str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _analyze_project(self):
        self._set_status("Analyzing...", TEXT_WARN)
        self._log("Analyzing project files...")

        def _worker():
            try:
                system = self._get_smart_system()
                result = system.analyze_project()
                self.after(0, self._on_analyze_done, result)
            except Exception as e:
                self.after(0, self._on_cycle_error, str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _query_llm(self):
        self._set_status("Querying LLM...", TEXT_WARN)
        self._log("Querying LLM for upgrades...")

        def _worker():
            try:
                system = self._get_smart_system()
                suggestions = system.query_for_upgrades(max_upgrades=5)
                self.after(0, self._on_suggestions, suggestions)
            except Exception as e:
                self.after(0, self._on_cycle_error, str(e))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_cycle_done(self, suggestions, system=None):
        self.run_btn.config(state="normal")
        self._current_suggestions = suggestions
        self._current_system = system
        self._set_status(f"Got {len(suggestions)} suggestions - review and apply", TEXT_OK)
        self._log(f"Got {len(suggestions)} suggestions - use 'Apply Selected' or 'Apply All'")
        self._refresh_suggestions(suggestions)
        self._refresh_all()

    def _on_analyze_done(self, result):
        self._set_status(f"Analyzed: {result['files_analyzed']} files", TEXT_OK)
        self._log(f"Analysis complete: {result['files_analyzed']} files, {result['total_files']} total")
        self._refresh_files()

    def _on_suggestions(self, suggestions):
        self._set_status(f"Got {len(suggestions)} suggestions", TEXT_OK)
        self._log(f"Received {len(suggestions)} upgrade suggestions")
        self._refresh_suggestions(suggestions)

    def _on_cycle_error(self, err):
        self.run_btn.config(state="normal")
        self._log(f"Error: {err}", "err")
        self._set_status("Error", TEXT_ERR)

    def _on_log(self, msg: str, level: str = "info"):
        self.after(0, self._log, msg, level)

    def _log(self, msg: str, level: str = ""):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_text.config(state="normal")
        self.log_text.insert("end", line, level)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _set_status(self, msg: str, color: str = TEXT_SEC):
        self.status_lbl.config(text=msg, fg=color)

    def _refresh_all(self):
        self._refresh_files()
        self._refresh_history()
        self._refresh_stats()

    def _refresh_files(self):
        self.files_tree.delete(*self.files_tree.get_children())
        db = self._get_db()
        if not db:
            return
        
        for ctx in db.get_all_contexts():
            ast_s = json.loads(ctx.get('ast_summary', '{}'))
            ts = ctx.get('last_analyzed', '')[:19] if ctx.get('last_analyzed') else 'Never'
            self.files_tree.insert("", "end", values=(
                ctx.get('file_path', ''),
                len(ast_s.get('classes', [])),
                len(ast_s.get('functions', [])),
                ast_s.get('total_lines', 0),
                ts
            ))

    def _refresh_suggestions(self, suggestions):
        self.sugg_tree.delete(*self.sugg_tree.get_children())
        for i, s in enumerate(suggestions):
            self.sugg_tree.insert("", "end", iid=str(i), values=(
                i + 1,
                s.get('file', ''),
                s.get('function', 'N/A'),
                s.get('issue', '')[:60],
                "Pending"
            ))

    def _refresh_history(self):
        self.hist_tree.delete(*self.hist_tree.get_children())
        db = self._get_db()
        if not db:
            return
        
        for row in db.get_upgrade_history(limit=100):
            ts = row.get('applied_at', '')[:19] if row.get('applied_at') else ''
            status = row.get('status', 'pending')
            reasoning = (row.get('llm_reasoning', '') or '')[:50]
            self.hist_tree.insert("", "end", tags=(status,), values=(
                row.get('id', ''),
                ts,
                row.get('file_path', ''),
                row.get('function_name', ''),
                status,
                reasoning
            ))

    def _refresh_stats(self):
        db = self._get_db()
        system = self._get_smart_system()
        
        if db:
            contexts = db.get_all_contexts()
            history = db.get_upgrade_history()
            self._stat_vars["files"].set(str(len(contexts)))
            self._stat_vars["upgrades"].set(str(len(history)))
            self._stat_vars["success"].set(str(sum(1 for h in history if h.get('status') == 'applied')))
        
        if system:
            stats = system.groq.get_stats()
            self._stat_vars["api_calls"].set(str(stats.get('total_calls', 0)))

    def _view_file_context(self):
        sel = self.files_tree.selection()
        if not sel:
            return
        item = self.files_tree.item(sel[0])
        file_path = item['values'][0]
        
        db = self._get_db()
        if db:
            ctx = db.get_file_context(file_path)
            if ctx:
                self.diff_text.delete("1.0", "end")
                self.diff_text.insert("end", f"# {file_path}\n\n", "header")
                ast_s = json.loads(ctx.get('ast_summary', '{}'))
                
                self.diff_text.insert("end", f"## Classes ({len(ast_s.get('classes', []))})\n")
                for cls in ast_s.get('classes', []):
                    self.diff_text.insert("end", f"  {cls.get('name', '')}\n")
                
                self.diff_text.insert("end", f"\n## Functions ({len(ast_s.get('functions', []))})\n")
                for fn in ast_s.get('functions', [])[:15]:
                    args = ', '.join(fn.get('args', [])[:3])
                    self.diff_text.insert("end", f"  {fn.get('name', '')}({args})\n")
                
                self.diff_text.insert("end", f"\n## Imports\n")
                for imp in json.loads(ctx.get('imports', '[]'))[:20]:
                    self.diff_text.insert("end", f"  {imp}\n")

    def _on_sugg_select(self, _=None):
        sel = self.sugg_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if hasattr(self, '_current_suggestions') and idx < len(self._current_suggestions):
            s = self._current_suggestions[idx]
            self.sugg_detail.delete("1.0", "end")
            self.sugg_detail.insert("end", f"File: {s.get('file', '')}\n")
            self.sugg_detail.insert("end", f"Function: {s.get('function', 'N/A')}\n")
            self.sugg_detail.insert("end", f"Reasoning: {s.get('reasoning', '')}\n\n")
            self.sugg_detail.insert("end", "---\nOLD CODE:\n", "old")
            self.sugg_detail.insert("end", s.get('current_code', 'N/A') + "\n\n")
            self.sugg_detail.insert("end", "---\nNEW CODE:\n", "new")
            self.sugg_detail.insert("end", s.get('new_code', 'N/A'))

    def _view_suggestion_diff(self):
        sel = self.sugg_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if hasattr(self, '_current_suggestions') and idx < len(self._current_suggestions):
            s = self._current_suggestions[idx]
            self.diff_text.delete("1.0", "end")
            self.diff_text.insert("end", f"# {s.get('file', '')} - {s.get('function', 'N/A')}\n\n", "header")
            self.diff_text.insert("end", f"# Issue: {s.get('issue', '')}\n\n", "header")
            self.diff_text.insert("end", "-- OLD --\n", "old")
            self.diff_text.insert("end", s.get('current_code', 'N/A') + "\n\n")
            self.diff_text.insert("end", "-- NEW --\n", "new")
            self.diff_text.insert("end", s.get('new_code', 'N/A'))

    def _apply_selected(self):
        sel = self.sugg_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if hasattr(self, '_current_suggestions') and idx < len(self._current_suggestions):
            s = self._current_suggestions[idx]
            system = self._get_smart_system()
            ok, msg = system.apply_upgrade(s)
            if ok:
                self._log(f"Applied upgrade to {s.get('file', '')}")
                self._refresh_history()
            else:
                self._log(f"Failed: {msg}", "err")

    def _apply_all(self):
        if hasattr(self, '_current_suggestions'):
            system = self._get_smart_system()
            for s in self._current_suggestions:
                ok, msg = system.apply_upgrade(s)
                status = "OK" if ok else f"FAIL: {msg}"
                self._log(f"{s.get('file', '')}: {status}")
            self._refresh_history()

    def _discard_all(self):
        self._current_suggestions = []
        self.sugg_tree.delete(*self.sugg_tree.get_children())
        self._log("Discarded all suggestions")

    def _revert_selected(self):
        sel = self.hist_tree.selection()
        if not sel:
            return
        item = self.hist_tree.item(sel[0])
        hist_id = int(item['values'][0])
        system = self._get_smart_system()
        ok, msg = system.revert_upgrade(hist_id)
        if ok:
            self._log(f"Reverted upgrade #{hist_id}")
            self._refresh_history()
        else:
            self._log(f"Revert failed: {msg}", "err")
