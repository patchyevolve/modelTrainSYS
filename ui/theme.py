"""
Theme and UI Utilities
Defines the visual language and shared widget factories for the ML System UI.
"""

import tkinter as tk
from tkinter import ttk

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

# ─── File Extension Groups ───────────────────────────────────────────────────
SUPPORTED_EXTS = {
    "Images":  [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"],
    "Text":    [".txt", ".csv", ".json", ".jsonl", ".xml"],
    "Stats":   [".csv", ".npy", ".npz", ".parquet"],
    "Archive": [".zip", ".tar", ".gz"],
}
ALL_EXTS = [e for exts in SUPPORTED_EXTS.values() for e in exts]

# ─── Styled Widget Factories ──────────────────────────────────────────────────

def styled_frame(parent, bg=BG_PANEL, bd=1, **kw):
    return tk.Frame(parent, bg=bg, highlightbackground=BORDER,
                    highlightthickness=bd, **kw)

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
    return tk.Button(parent, text=text, command=cmd,
                     bg=BG_INPUT, fg=TEXT_SEC, activebackground=BORDER,
                     activeforeground=TEXT_PRI, relief="flat", cursor="hand2",
                     font=("Segoe UI", 9), width=width, pady=4)

def separator(parent, bg=BG_PANEL):
    return tk.Frame(parent, bg=BORDER, height=1)

def fmt_size(b):
    for u in ["B", "KB", "MB", "GB"]:
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} TB"

def setup_ttk_styles():
    """Initializes global TTK styles to match the dark theme."""
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Green.Horizontal.TProgressbar",
                    troughcolor=BG_INPUT, background=ACCENT,
                    bordercolor=BG_INPUT, lightcolor=ACCENT,
                    darkcolor=ACCENT)
    style.configure("Treeview", background=BG_CARD, foreground=TEXT_PRI,
                    fieldbackground=BG_CARD, borderwidth=0,
                    font=("Segoe UI", 9))
    style.configure("Treeview.Heading", background=BG_INPUT,
                    foreground=TEXT_SEC, font=("Segoe UI", 8, "bold"),
                    relief="flat")
    style.map("Treeview", background=[("selected", ACCENT2)])