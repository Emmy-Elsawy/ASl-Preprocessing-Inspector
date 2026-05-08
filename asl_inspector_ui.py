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
# DESIGN TOKENS  —  Modern Minimalist Fun (light)
# ──────────────────────────────────────────────────────────────────────────────

# Base layout
BG_APP     = "#FEF3E2"
BG_SIDEBAR = "#FCF9F1"
BG_PANEL   = "#FFFFFF"
BORDER     = "#E5E7EB"

# Text
TEXT_PRIMARY   = "#111827"
TEXT_SECONDARY = "#6B7280"

# Candy accents
CYAN    = "#22D3EE"
SALMON  = "#F87171"
LIME    = "#A3E635"
ORANGE  = "#FB923C"
SKY     = "#7DD3FC"

# Letter buttons
BTN_BG     = "#F0F9FF"
BTN_FG     = "#0EA5E9"
BTN_ACTIVE = "#F87171"
BTN_ACT_FG = "#FFFFFF"

# Matplotlib
FIG_BG      = "#FEF3E2"
AXES_BG     = "#FFFFFF"
GRID_COLOR  = "#E5E7EB"
TICK_COLOR  = "#6B7280"
SPINE_COLOR = "#E5E7EB"

# ──────────────────────────────────────────────────────────────────────────────
# TTK STYLE
# ──────────────────────────────────────────────────────────────────────────────

def configure_styles():
    style = ttk.Style()
    style.theme_use("clam")

    style.configure(
        "TNotebook",
        background="#FFFFFF",
        borderwidth=0,
    )
    style.configure(
        "TNotebook.Tab",
        background="#F3F4F6", # Light gray for unselected
        foreground=TEXT_SECONDARY,
        padding=[20, 10],
    )
    # The "Magic" map for the active tab color
    style.map(
        "TNotebook.Tab",
        background=[("selected", SALMON)], # Background turns Salmon
        foreground=[("selected", "#FFFFFF")], # Text turns White
    )
# ──────────────────────────────────────────────────────────────────────────────
# WIDGET HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def make_label(parent, text, size=11, bold=False, color=TEXT_PRIMARY, **kw):
    weight = "bold" if bold else "normal"
    return tk.Label(
        parent,
        text=text,
        font=("Segoe UI", size, weight),
        fg=color,
        bg=parent.cget("bg"),
        **kw,
    )


def make_divider(parent, color=BORDER):
    return tk.Frame(parent, height=1, bg=color)


# ──────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB AXIS STYLER
# ──────────────────────────────────────────────────────────────────────────────

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
        self.root.resizable(True, True)

        configure_styles()

        print("Loading dataset...")
        self.X_raw, self.y_raw = load_csv(TRAIN_CSV, max_samples=1000)
        self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXY"
        self._current_idx = 0

        self._build_ui()
        self.update_display(0)

    # ──────────────────────────────────────────────────────────────────────
    # UI CONSTRUCTION
    # ──────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.pipeline_tab = tk.Frame(self.notebook, bg=BG_APP)
        self.analysis_tab = tk.Frame(self.notebook, bg=BG_APP)

        self.notebook.add(self.pipeline_tab, text="  Pipeline  ")
        self.notebook.add(self.analysis_tab, text="  Analysis  ")

        self._build_pipeline_tab()
        self._build_analysis_tab()

    # ── Pipeline tab ──────────────────────────────────────────────────────

    def _build_pipeline_tab(self):
        sidebar = tk.Frame(self.pipeline_tab, width=280, bg=BG_SIDEBAR)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # Right border line on sidebar
        tk.Frame(self.pipeline_tab, width=1, bg=BORDER).pack(side="left", fill="y")

        self._build_sidebar(sidebar)

        content = tk.Frame(self.pipeline_tab, bg=BG_APP)
        content.pack(side="right", fill="both", expand=True)

        # Header
        self.header_label = tk.Label(
            content,
            text="Select a letter or click Random",
            font=("Segoe UI", 14, "bold"),
            fg=TEXT_PRIMARY,
            bg=BG_APP,
        )
        self.header_label.pack(anchor="w", padx=28, pady=(20, 0))

        self.sub_label = tk.Label(
            content,
            text="10 preprocessing steps applied in sequence",
            font=("Segoe UI", 9),
            fg=TEXT_SECONDARY,
            bg=BG_APP,
        )
        self.sub_label.pack(anchor="w", padx=28, pady=(3, 12))

        make_divider(content, BORDER).pack(fill="x")

        fig_frame = tk.Frame(content, bg=BG_APP)
        fig_frame.pack(fill="both", expand=True, padx=16, pady=16)

        self.fig, self.axes = plt.subplots(2, 5, figsize=(14, 7))
        self.fig.patch.set_facecolor(FIG_BG)
        self.fig.subplots_adjust(
            left=0.01, right=0.99,
            top=0.93, bottom=0.02,
            wspace=0.06, hspace=0.30,
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Sidebar ───────────────────────────────────────────────────────────

    def _build_sidebar(self, sidebar):
        tk.Frame(sidebar, height=24, bg=BG_SIDEBAR).pack()

        # Brand
        brand_row = tk.Frame(sidebar, bg=BG_SIDEBAR)
        brand_row.pack(fill="x", padx=20)

        tk.Label(
            brand_row,
            text="ASL",
            font=("Segoe UI", 22, "bold"),
            fg="#22D3EE",
            bg=BG_SIDEBAR,
        ).pack(side="left")

        tk.Label(
            brand_row,
            text=" Inspector",
            font=("Segoe UI", 16),
            fg=TEXT_PRIMARY,
            bg=BG_SIDEBAR,
        ).pack(side="left", pady=(5, 0))

        tk.Label(
            sidebar,
            text="CS.383  \u00b7  Image Preprocessing",
            font=("Segoe UI", 8),
            fg=TEXT_SECONDARY,
            bg=BG_SIDEBAR,
        ).pack(anchor="w", padx=20, pady=(2, 18))

        make_divider(sidebar, BORDER).pack(fill="x", padx=20, pady=(0, 14))

        # Letters section label
        tk.Label(
            sidebar,
            text="LETTERS",
            font=("Segoe UI", 8, "bold"),
            fg=TEXT_SECONDARY,
            bg=BG_SIDEBAR,
        ).pack(anchor="w", padx=20, pady=(0, 8))

        # Button grid
        btn_grid = tk.Frame(sidebar, bg=BG_SIDEBAR)
        btn_grid.pack(padx=16, fill="x")

# Inside _build_sidebar(self, sidebar)
        self.letter_buttons = {}
        for i, char in enumerate(self.letters):
            lbl = tk.Label(
                btn_grid,
                text=char,
                width=4,
                height=2,
                bg=BTN_BG,
                fg=BTN_FG,
                font=("Segoe UI", 10, "bold"),
                cursor="hand2",
                relief="flat"
            )
            lbl.grid(row=i // 5, column=i % 5, padx=2, pady=2, sticky="nsew")
            
            # This captures the index correctly for the click event
            lbl.bind("<Button-1>", lambda e, idx=i: self._select_letter(idx))
            self.letter_buttons[i] = lbl

        for col in range(5):
            btn_grid.columnconfigure(col, weight=1)

        make_divider(sidebar, BORDER).pack(fill="x", padx=20, pady=16)

        # Info card
        self.info_card = tk.Frame(
            sidebar,
            bg="#F3F4F6",
            highlightbackground=BORDER,
            highlightthickness=1,
            pady=14,
            padx=14,
        )
        self.info_card.pack(fill="x", padx=16, pady=(0, 14))

        self.info_letter = tk.Label(
            self.info_card,
            text="\u2014",
            font=("Segoe UI", 48, "bold"),
            fg=CYAN,
            bg=BG_APP,
        )
        self.info_letter.pack()

        self.info_detail = tk.Label(
            self.info_card,
            text="No sample selected",
            font=("Segoe UI", 8),
            fg=TEXT_SECONDARY,
            bg=BG_APP,
            wraplength=230,
            justify="center",
        )
        self.info_detail.pack(pady=(2, 0))

# Random Sample button
# Use a Label to bypass macOS button styling restrictions
        rand_btn = tk.Label(
            sidebar,
            text="Random Sample",
            bg="#22D3EE",           # This will now show as a solid color
            fg="#FFFFFF",           # White text
            font=("Segoe UI", 10, "bold"),
            pady=12,                # Vertical padding to give it height
            cursor="hand2",         # Changes cursor to a hand on hover
            relief="flat"
        )
        rand_btn.pack(fill="x", padx=16, pady=(0, 8))

        # This line makes the Label act like a button when clicked
        rand_btn.bind("<Button-1>", lambda e: self.random_sample())
        rand_btn.pack(fill="x", padx=16, pady=(0, 8))

        # Spacer + status bar
        tk.Frame(sidebar, bg=BG_SIDEBAR).pack(fill="y", expand=True)
        make_divider(sidebar, BORDER).pack(fill="x", padx=20)
        self.status_label = tk.Label(
            sidebar,
            text="Ready",
            font=("Segoe UI", 8),
            fg=TEXT_SECONDARY,
            bg=BG_SIDEBAR,
        )
        self.status_label.pack(anchor="w", padx=20, pady=8)

    # ── Analysis tab ──────────────────────────────────────────────────────

    def _build_analysis_tab(self):
        hdr = tk.Frame(self.analysis_tab, bg=BG_APP)
        hdr.pack(fill="x", padx=28, pady=(20, 0))

        tk.Label(
            hdr,
            text="Histogram & Contrast Analysis",
            font=("Segoe UI", 14, "bold"),
            fg=TEXT_PRIMARY,
            bg=BG_APP,
        ).pack(side="left")

        make_divider(self.analysis_tab, BORDER).pack(fill="x", pady=10)

        fig_frame = tk.Frame(self.analysis_tab, bg=BG_APP)
        fig_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.analysis_fig = Figure(figsize=(14, 7), dpi=100)
        self.analysis_fig.patch.set_facecolor(FIG_BG)

        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_fig, master=fig_frame)
        self.analysis_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ──────────────────────────────────────────────────────────────────────
    # INTERACTIONS
    # ──────────────────────────────────────────────────────────────────────

    def random_sample(self):
        idx = random.randint(0, len(self.X_raw) - 1)
        self._current_idx = idx
        self.update_display(idx)

    def _select_letter(self, letter_idx):
        # Update visual states of all Labels in the grid
        for i, lbl in self.letter_buttons.items():
            if i == letter_idx:
                lbl.configure(bg=BTN_ACTIVE, fg=BTN_ACT_FG) # Turn Pink
            else:
                lbl.configure(bg=BTN_BG, fg=BTN_FG)         # Back to Light Blue
        
        # Logic to find the sample in your dataset
        matches = np.where(self.y_raw == letter_idx)[0]
        if len(matches) > 0:
            self._current_idx = int(matches[0])
            self.update_display(self._current_idx)

    # ──────────────────────────────────────────────────────────────────────
    # DISPLAY UPDATE
    # ──────────────────────────────────────────────────────────────────────

    def update_display(self, idx):
        img   = self.X_raw[idx]
        label = self.letters[self.y_raw[idx]]

        self.info_letter.configure(text=label)
        self.info_detail.configure(
            text=f"Sample #{idx}  \u00b7  28\u00d728 px  \u00b7  grayscale"
        )
        self.header_label.configure(text=f'Pipeline \u2014 Letter "{label}"')
        self.status_label.configure(text=f"Showing sample {idx}")

        # Processing steps
        gaussian = (gaussian_blur(img.astype(float) / 255.0, sigma=1) * 255).astype(np.uint8)
        laplace  = laplacian_filter(img)
        inverse  = image_inverse(img)
        gamma    = (gamma_correction(img, gamma=0.45) * 255).astype(np.uint8)
        log_img  = log_transform(img)
        hist_eq  = histogram_equalization(img)
        contrast = contrast_stretching(img)
        sharpen  = sharpening_filter(img)
        sobel    = np.clip(sobel_edge_detection(img), 0, 255).astype(np.uint8)

        steps = [
            ("Original",      img),
            ("Gaussian blur", gaussian),
            ("Laplacian",     laplace),
            ("Inverse",       inverse),
            ("Gamma  0.45",   gamma),
            ("Log transform", log_img),
            ("Histogram EQ",  hist_eq),
            ("Contrast",      contrast),
            ("Sharpening",    sharpen),
            ("Sobel edges",   sobel),
        ]

        # Candy accent color per step title
        title_colors = [
            "#F87171", "#F87171", "#F87171", "#F87171", "#F87171",
            "#F87171", "#F87171", "#F87171", "#F87171", "#F87171",
        ]

        # Pipeline tab plots
        for i, (name, data) in enumerate(steps):
            ax = self.axes[i // 5, i % 5]
            ax.clear()
            ax.imshow(data, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
            ax.set_facecolor(AXES_BG)
            ax.axis("off")

            ax.set_title(
                f"{i+1:02d}  {name}",
                fontsize=8,
                color=title_colors[i],
                loc="left",
                pad=5,
                fontweight="bold",
            )

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(BORDER)
                spine.set_linewidth(0.8)

        self.canvas.draw()

        # Analysis tab plots
        self.analysis_fig.clear()
        self.analysis_fig.patch.set_facecolor(FIG_BG)
        self.analysis_fig.subplots_adjust(
            left=0.07, right=0.97,
            top=0.90, bottom=0.10,
            wspace=0.32, hspace=0.50,
        )
        axs = self.analysis_fig.subplots(2, 2)

# --- Inside update_display() under the Analysis tab section ---

        # 1 — Histogram comparison
        ax = axs[0, 0]
        ax.hist(img.flatten(),     bins=48, alpha=0.75, color=LIME,   label="Original")
        ax.hist(hist_eq.flatten(), bins=48, alpha=0.65, color=SALMON, label="Equalized")
        style_ax(ax, title="Histogram: original vs equalized", xlabel="Pixel value", ylabel="Count")

        # 2 — Pixel intensity profile
        ax = axs[0, 1]
        ax.plot(img[14],     color=SKY,    linewidth=1.6, label="Original") # Using SKY for contrast
        ax.plot(hist_eq[14], color=ORANGE, linewidth=1.6, label="Equalized")
        style_ax(ax, title="Intensity profile (row 14)", xlabel="Column", ylabel="Intensity")

        # 3 — Edge magnitude distribution
        ax = axs[1, 0]
        ax.hist(sobel.flatten(), bins=48, color=SALMON, alpha=0.80)
        style_ax(ax, title="Sobel edge magnitude distribution", xlabel="Magnitude", ylabel="Count")

        # 4 — Contrast comparison bar chart
        ax = axs[1, 1]
        bar_labels = ["Original", "Hist EQ", "Contrast"]
        values     = [np.std(img), np.std(hist_eq), np.std(contrast)]
        bar_colors = [LIME, ORANGE, SALMON] # Matching the candy bar colors
        bars = ax.bar(bar_labels, values, color=bar_colors, width=0.5)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=8, color=TEXT_SECONDARY,
            )
        style_ax(ax, title="Contrast comparison (std dev)", ylabel="Std deviation")

        # Finalize axis styling
        for row in axs:
            for ax in row:
                ax.set_facecolor(AXES_BG)
                ax.tick_params(colors=TICK_COLOR, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor(SPINE_COLOR)
                    spine.set_linewidth(0.8)
                ax.yaxis.label.set_color(TEXT_SECONDARY)
                ax.xaxis.label.set_color(TEXT_SECONDARY)

        self.analysis_canvas.draw()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLInspector(root)
    root.mainloop()
