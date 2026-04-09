"""
Reusable UI Components
Contains standalone widgets like DropZone, DataPanel, and LineChart.
"""

import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from ui.theme import (
    BG_PANEL, BG_CARD, BG_INPUT, ACCENT2, BORDER, TEXT_PRI, TEXT_SEC, 
    TEXT_ERR, TEXT_OK, DRAG_OVER, SUPPORTED_EXTS, ALL_EXTS, 
    styled_frame, label, section_title, ghost_btn
)

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
            pass

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

# ─── Data Panel ───────────────────────────────────────────────────────────────

class DataPanel(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG_PANEL, **kw)
        self.files = []
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=BG_PANEL)
        hdr.pack(fill="x", padx=12, pady=(12, 6))
        label(hdr, "DATA", fg=TEXT_SEC, bg=BG_PANEL,
              font=("Segoe UI", 8, "bold")).pack(side="left")
        self.count_lbl = label(hdr, "0 files", fg=ACCENT2, bg=BG_PANEL,
                               font=("Segoe UI", 8))
        self.count_lbl.pack(side="right")

        self.drop = DropZone(self, self._add_files)
        self.drop.pack(fill="x", padx=12, pady=(0, 8))

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
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_list.yview)
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
            tag = "📁"
            if ext in SUPPORTED_EXTS["Images"]: counts["Images"] += 1; tag = "🖼"
            elif ext in SUPPORTED_EXTS["Text"]: counts["Text"] += 1; tag = "📄"
            elif ext in SUPPORTED_EXTS["Stats"]: counts["Stats"] += 1; tag = "📊"
            elif ext in SUPPORTED_EXTS["Archive"]: counts["Archive"] += 1; tag = "📦"
            else: counts["Other"] += 1
            self.file_list.insert("end", f"  {tag} {name}")
        for k, v in self.type_vars.items(): v.set(str(counts[k]))
        self.count_lbl.config(text=f"{len(self.files)} files")

    def _clear(self):
        self.files.clear()
        self._refresh()

    def get_files(self):
        return self.files

# ─── Chart ───────────────────────────────────────────────────────────────────

class LineChart(tk.Canvas):
    def __init__(self, parent, label="Loss", color=TEXT_ERR, **kw):
        super().__init__(parent, bg=BG_CARD, highlightthickness=0, **kw)
        self.label, self.color, self.data = label, color, []
        self.bind("<Configure>", lambda e: self._draw())

    def push(self, val):
        self.data.append(val)
        if len(self.data) > 200: self.data = self.data[-200:]
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10 or not self.data: return
        pad, mn, mx = 6, min(self.data), max(self.data)
        rng = mx - mn or 1
        pts = [(pad + (i/max(len(self.data)-1,1))*(w-2*pad), h-pad-((v-mn)/rng)*(h-2*pad)) 
               for i, v in enumerate(self.data)]
        for f in [0.25, 0.5, 0.75]:
            yg = pad + f*(h-2*pad)
            self.create_line(pad, yg, w-pad, yg, fill=BORDER, dash=(2,4))
        if len(pts) > 1:
            self.create_line(*[c for p in pts for c in p], fill=self.color, width=2, smooth=True)
        if pts:
            lx, ly = pts[-1]
            self.create_oval(lx-3, ly-3, lx+3, ly+3, fill=self.color, outline="")
        self.create_text(pad+2, pad+2, text=self.label, fill=TEXT_SEC, font=("Segoe UI", 7), anchor="nw")
        self.create_text(w-pad-2, pad+2, text=f"{self.data[-1]:.4f}" if self.data else "", 
                         fill=self.color, font=("Segoe UI", 7, "bold"), anchor="ne")