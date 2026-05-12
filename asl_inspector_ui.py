import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random

# ──────────────────────────────────────────────────────────────────────────────
# IMPORT PREPROCESSING FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

try:
    from asl_preprocessing import (
        load_csv,
        TRAIN_CSV,
        gaussian_blur,
        laplacian_filter,
        image_inverse,
        gamma_correction,
        log_transform,
        histogram_equalization,
        contrast_stretching,
        sharpening_filter,
        sobel_edge_detection
    )
except ImportError:
    print("Error: Could not find asl_preprocessing.py")

# ──────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ──────────────────────────────────────────────────────────────────────────────

BG_APP     = "#FEF3E2"
BG_SIDEBAR = "#FCF9F1"
BG_PANEL   = "#FFFFFF"
BORDER     = "#E5E7EB"
TEXT_PRIMARY   = "#111827"
TEXT_SECONDARY = "#6B7280"
CYAN    = "#22D3EE"
SALMON  = "#F87171"
LIME    = "#A3E635"
ORANGE  = "#FB923C"
SKY     = "#7DD3FC"
BTN_BG     = "#F0F9FF"
BTN_FG     = "#0EA5E9"
BTN_ACTIVE = "#F87171"
BTN_ACT_FG = "#FFFFFF"
FIG_BG      = "#FEF3E2"
AXES_BG     = "#FFFFFF"
TICK_COLOR  = "#6B7280"
SPINE_COLOR = "#E5E7EB"

# ──────────────────────────────────────────────────────────────────────────────
# STYLES & HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def configure_styles():
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TNotebook", background="#FFFFFF", borderwidth=0)
    style.configure("TNotebook.Tab", background="#F3F4F6", foreground=TEXT_SECONDARY, padding=[20, 10])
    style.map("TNotebook.Tab", background=[("selected", SALMON)], foreground=[("selected", "#FFFFFF")])

def make_divider(parent, color=BORDER):
    return tk.Frame(parent, height=1, bg=color)

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#FFFFFF")
    ax.tick_params(colors=TICK_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=9, pad=7, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=8)
    ax.grid(axis="y", color=BORDER, linewidth=0.6, alpha=0.5)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ──────────────────────────────────────────────────────────────────────────────

class ASLInspector:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Pipeline Inspector  —  CS.383")
        self.root.geometry("1640x960")
        self.root.configure(bg=BG_APP)

        configure_styles()

        print("Loading dataset...")
        self.X_raw, self.y_raw = load_csv(TRAIN_CSV, max_samples=1000)
        self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXY"
        self._current_idx = 0

        self._build_ui()
        self.update_display(0)

    def _build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.pipeline_tab = tk.Frame(self.notebook, bg=BG_APP)
        self.analysis_tab = tk.Frame(self.notebook, bg=BG_APP)

        self.notebook.add(self.pipeline_tab, text="  Pipeline  ")
        self.notebook.add(self.analysis_tab, text="  Analysis  ")

        self._build_pipeline_tab()
        self._build_analysis_tab()

    def _build_pipeline_tab(self):
        sidebar = tk.Frame(self.pipeline_tab, width=280, bg=BG_SIDEBAR)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Frame(self.pipeline_tab, width=1, bg=BORDER).pack(side="left", fill="y")

        self._build_sidebar(sidebar)

        content = tk.Frame(self.pipeline_tab, bg=BG_APP)
        content.pack(side="right", fill="both", expand=True)

        self.header_label = tk.Label(content, text="Loading...", font=("Segoe UI", 14, "bold"), fg=TEXT_PRIMARY, bg=BG_APP)
        self.header_label.pack(anchor="w", padx=28, pady=(20, 0))

        tk.Label(content, text="10 preprocessing steps applied in sequence", font=("Segoe UI", 9), fg=TEXT_SECONDARY, bg=BG_APP).pack(anchor="w", padx=28, pady=(3, 12))
        make_divider(content, BORDER).pack(fill="x")

        fig_frame = tk.Frame(content, bg=BG_APP)
        fig_frame.pack(fill="both", expand=True, padx=16, pady=16)

        self.fig, self.axes = plt.subplots(2, 5, figsize=(14, 7))
        self.fig.patch.set_facecolor(FIG_BG)
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.06, hspace=0.30)

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_sidebar(self, sidebar):
        tk.Frame(sidebar, height=24, bg=BG_SIDEBAR).pack()
        
        # Brand Header
        brand_row = tk.Frame(sidebar, bg=BG_SIDEBAR)
        brand_row.pack(fill="x", padx=20)
        tk.Label(brand_row, text="ASL", font=("Segoe UI", 22, "bold"), fg=CYAN, bg=BG_SIDEBAR).pack(side="left")
        tk.Label(brand_row, text=" Inspector", font=("Segoe UI", 16), fg=TEXT_PRIMARY, bg=BG_SIDEBAR).pack(side="left", pady=(5, 0))

        make_divider(sidebar, BORDER).pack(fill="x", padx=20, pady=(15, 14))

        # Letter Grid
        btn_grid = tk.Frame(sidebar, bg=BG_SIDEBAR)
        btn_grid.pack(padx=16, fill="x")
        self.letter_buttons = {}
        for i, char in enumerate(self.letters):
            lbl = tk.Label(btn_grid, text=char, width=4, height=2, bg=BTN_BG, fg=BTN_FG, font=("Segoe UI", 10, "bold"), cursor="hand2")
            lbl.grid(row=i // 5, column=i % 5, padx=2, pady=2, sticky="nsew")
            lbl.bind("<Button-1>", lambda e, idx=i: self._select_letter(idx))
            self.letter_buttons[i] = lbl

        make_divider(sidebar, BORDER).pack(fill="x", padx=20, pady=16)

        # Info Card
        self.info_card = tk.Frame(sidebar, bg="#F3F4F6", pady=14, padx=14)
        self.info_card.pack(fill="x", padx=16, pady=(0, 14))
        self.info_letter = tk.Label(self.info_card, text="?", font=("Segoe UI", 48, "bold"), fg=CYAN, bg="#F3F4F6")
        self.info_letter.pack()
        self.info_detail = tk.Label(self.info_card, text="Select sample", font=("Segoe UI", 8), fg=TEXT_SECONDARY, bg="#F3F4F6")
        self.info_detail.pack()

        # Action Button
        rand_btn = tk.Label(sidebar, text="Random Sample", bg=CYAN, fg="white", font=("Segoe UI", 10, "bold"), pady=12, cursor="hand2")
        rand_btn.pack(fill="x", padx=16, pady=(10, 8))
        rand_btn.bind("<Button-1>", lambda e: self.random_sample())

        self.status_label = tk.Label(sidebar, text="Ready", font=("Segoe UI", 8), fg=TEXT_SECONDARY, bg=BG_SIDEBAR)
        self.status_label.pack(side="bottom", anchor="w", padx=20, pady=8)

    def _build_analysis_tab(self):
        hdr = tk.Frame(self.analysis_tab, bg=BG_APP)
        hdr.pack(fill="x", padx=28, pady=(20, 0))
        tk.Label(hdr, text="Histogram & Contrast Analysis", font=("Segoe UI", 14, "bold"), fg=TEXT_PRIMARY, bg=BG_APP).pack(side="left")
        
        fig_frame = tk.Frame(self.analysis_tab, bg=BG_APP)
        fig_frame.pack(fill="both", expand=True, padx=16, pady=16)
        self.analysis_fig = Figure(figsize=(14, 7), dpi=100)
        self.analysis_fig.patch.set_facecolor(FIG_BG)
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_fig, master=fig_frame)
        self.analysis_canvas.get_tk_widget().pack(fill="both", expand=True)

    def random_sample(self):
        idx = random.randint(0, len(self.X_raw) - 1)
        self.update_display(idx)

    def _select_letter(self, letter_idx):
        for i, lbl in self.letter_buttons.items():
            lbl.configure(bg=BTN_ACTIVE if i == letter_idx else BTN_BG, fg=BTN_ACT_FG if i == letter_idx else BTN_FG)
        matches = np.where(self.y_raw == letter_idx)[0]
        if len(matches) > 0:
            self.update_display(matches[0])

    def update_display(self, idx):
        img = self.X_raw[idx]
        label = self.letters[self.y_raw[idx]]

        self.info_letter.configure(text=label)
        self.info_detail.configure(text=f"Sample #{idx} \u00b7 28x28 grayscale")
        self.header_label.configure(text=f'Pipeline \u2014 Letter "{label}"')

        # --- IMAGE PROCESSING ---
        gauss = (gaussian_blur(img.astype(float)/255.0, sigma=1)*255).astype(np.uint8)
        
        # LAPLACIAN NORMALIZATION FIX:
        lap_raw = laplacian_filter(img)
        l_min, l_max = np.min(lap_raw), np.max(lap_raw)
        if l_max - l_min != 0:
            lap_norm = (lap_raw - l_min) / (l_max - l_min) * 255
        else:
            lap_norm = lap_raw
        laplace = lap_norm.astype(np.uint8)

        inv   = image_inverse(img)
        gam   = (gamma_correction(img, 0.45)*255).astype(np.uint8)
        log_i = log_transform(img)
        heq   = histogram_equalization(img)
        cstr  = contrast_stretching(img)
        shrp  = sharpening_filter(img)
        sob   = np.clip(sobel_edge_detection(img), 0, 255).astype(np.uint8)

        steps = [
            ("Original", img), ("Gaussian", gauss), ("Laplacian", laplace),
            ("Inverse", inv), ("Gamma 0.45", gam), ("Log Trans", log_i),
            ("Hist EQ", heq), ("Contrast", cstr), ("Sharpening", shrp), ("Sobel", sob)
        ]

        for i, (name, data) in enumerate(steps):
            ax = self.axes[i // 5, i % 5]
            ax.clear()
            ax.imshow(data, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{i+1:02d} {name}", fontsize=8, color=SALMON, fontweight="bold")
            ax.axis("off")
        self.canvas.draw()

        # Analysis Tab
        self.analysis_fig.clear()
        axs = self.analysis_fig.subplots(2, 2)
        self.analysis_fig.subplots_adjust(wspace=0.3, hspace=0.4)
        
        # Histograms
        axs[0,0].hist(img.flatten(), bins=32, color=LIME, alpha=0.6, label="Orig")
        axs[0,0].hist(heq.flatten(), bins=32, color=SALMON, alpha=0.5, label="HEQ")
        style_ax(axs[0,0], "Histogram Comparison")
        
        # Profile
        axs[0,1].plot(img[14], color=SKY, label="Orig")
        axs[0,1].plot(heq[14], color=ORANGE, label="HEQ")
        style_ax(axs[0,1], "Intensity Profile (Row 14)")
        
        # Sobel Dist
        axs[1,0].hist(sob.flatten(), bins=32, color=SALMON)
        style_ax(axs[1,0], "Edge Magnitude Distribution")
        
        # Std Dev Bar
        axs[1,1].bar(["Orig", "HEQ", "Contrast"], [np.std(img), np.std(heq), np.std(cstr)], color=[LIME, ORANGE, SALMON])
        style_ax(axs[1,1], "Contrast (Std Dev)")
        
        self.analysis_canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLInspector(root)
    root.mainloop()