# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   CS.383 — Real-Time Sign Language Preprocessing Visualizer               ║
# ║   Upgraded: larger fonts · live histograms · no Lab tags · clean UI        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import tkinter as tk
from tkinter import ttk
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage import filters
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_blur(image, sigma=1):
    return filters.gaussian(image, sigma=sigma)

def laplacian_filter(image):
    lap = filters.laplace(image)
    lap = lap - lap.min()
    if lap.max() > 0:
        lap = lap / lap.max()
    return lap

def image_inverse(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    return 255 - image

def gamma_correction(image, gamma=0.5):
    img = image.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img ** gamma, 0, 1)

def log_transform(image):
    img = image.astype(float)
    result = (255.0 * np.log(1 + img)) / np.log(256)
    return np.clip(result, 0, 255).astype(np.uint8)

def histogram_equalization(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    flat    = image.flatten()
    hist, _ = np.histogram(image, 256, [0, 255])
    cdf     = hist.cumsum()
    cdf_m   = np.ma.masked_equal(cdf, 0)
    cdf_m   = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf     = np.ma.filled(cdf_m, 0).astype('uint8')
    return np.reshape(cdf[flat], image.shape)

def contrast_stretching(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    b, a = image.max(), image.min()
    if b == a:
        return image
    return (255 * (image.astype(float) - a) / (b - a)).astype(np.uint8)

def sharpening_filter(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    kernel = np.array([[ 0, -1,  0],
                        [-1,  5, -1],
                        [ 0, -1,  0]])
    return cv2.filter2D(image, -1, kernel)

def sobel_edge_detection(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.clip(cv2.magnitude(sobelx, sobely), 0, 255).astype(np.uint8)


def run_pipeline(frame_bgr):
    """
    Run the full preprocessing pipeline on a BGR webcam frame.
    Returns list of (step_name, formula, uint8_gray_image) tuples.
    """
    gray     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    blurred  = gaussian_blur(gray.astype(float) / 255.0, sigma=1)
    blurred8 = (blurred * 255).astype(np.uint8)

    lap      = laplacian_filter(gray.astype(float) / 255.0)
    lap8     = (lap * 255).astype(np.uint8)

    inv      = image_inverse(gray)
    gam      = gamma_correction(gray, gamma=0.45)
    gam8     = (gam * 255).astype(np.uint8)
    logt     = log_transform(gray)
    heq      = histogram_equalization(gray)
    st       = contrast_stretching(heq)
    sh       = sharpening_filter(st)
    sobel    = sobel_edge_detection(sh)

    return [
        ("Original",            "BGR → Grayscale",              gray),
        ("Gaussian Blur σ=1",   "filters.gaussian(img, σ=1)",   blurred8),
        ("Laplacian",           "filters.laplace(img)",          lap8),
        ("Inverse",             "255 − image",                   inv),
        ("Gamma  γ=0.45",       "(img/255)^0.45",                gam8),
        ("Log Transform",       "255·log(1+f)/log(256)",         logt),
        ("Hist. Equalization",  "CDF remapping",                 heq),
        ("Contrast Stretching", "255·(x−min)/(max−min)",         st),
        ("Sharpening",          "cv2.filter2D  [0,−1,0]",        sh),
        ("Sobel Edges",         "√(Gx²+Gy²)",                    sobel),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════
BG      = '#0b0d14'
PANEL   = '#12151f'
CARD    = '#181c28'
BORDER  = '#252d40'
TEXT    = '#dde4f5'
MUTED   = '#4a5568'
ACCENT  = '#00e5ff'
SUCCESS = '#10b981'
DANGER  = '#ef4444'
WARNING = '#f59e0b'
ACCENT2 = '#7c3aed'

PLOT_BG   = '#0d1117'
PLOT_FACE = '#161b27'

# Per-step accent colours
STEP_COLORS = [
    '#4fc3f7',  # Original
    '#38bdf8',  # Gaussian
    '#818cf8',  # Laplacian
    '#a78bfa',  # Inverse
    '#c084fc',  # Gamma
    '#34d399',  # Log
    '#6ee7b7',  # HEQ
    '#fbbf24',  # Contrast
    '#fb923c',  # Sharpening
    '#f87171',  # Sobel
]

TITLE_FONT = ('Georgia', 15, 'bold')
HEAD_FONT  = ('Georgia', 11, 'bold')
BODY_FONT  = ('Courier New', 11)
MONO_LARGE = ('Courier New', 12, 'bold')
SMALL      = ('Courier New', 9)

# Configure matplotlib globally
matplotlib.rcParams.update({
    'figure.facecolor': PLOT_BG,
    'axes.facecolor':   PLOT_FACE,
    'axes.edgecolor':   BORDER,
    'xtick.color':      MUTED,
    'ytick.color':      MUTED,
    'text.color':       TEXT,
})

THUMB    = 150   # thumbnail px
FEED_W   = 500
FEED_H   = 370


# ══════════════════════════════════════════════════════════════════════════════
# STEP CARD — image thumbnail + live histogram
# ══════════════════════════════════════════════════════════════════════════════
class StepCard(tk.Frame):
    """Card showing: step name, image thumbnail, live histogram."""

    def __init__(self, parent, step_idx, name, formula, **kwargs):
        color = STEP_COLORS[step_idx % len(STEP_COLORS)]
        super().__init__(parent, bg=CARD, bd=0,
                         highlightthickness=1, highlightbackground=color,
                         **kwargs)
        self._color = color

        # Coloured top bar
        tk.Frame(self, bg=color, height=3).pack(fill='x')

        # Step name (no lab tag)
        tk.Label(self, text=name, bg=CARD, fg=color,
                 font=('Georgia', 10, 'bold'),
                 anchor='center').pack(fill='x', padx=6, pady=(5, 0))

        # Image canvas
        self.canvas = tk.Canvas(self, width=THUMB, height=THUMB,
                                bg='#06080e', highlightthickness=0)
        self.canvas.pack(padx=6, pady=(4, 2))
        self._imgtk = None

        # Formula label
        tk.Label(self, text=formula, bg=CARD, fg=MUTED,
                 font=('Courier New', 7),
                 wraplength=THUMB + 14, justify='center').pack(pady=(0, 3))

        # Mini histogram via matplotlib
        self._fig = Figure(figsize=(THUMB / 90, 0.9), facecolor=PLOT_BG,
                           dpi=90)
        self._fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.25)
        self._ax = self._fig.add_subplot(1, 1, 1)
        self._ax.set_facecolor(PLOT_FACE)
        self._ax.tick_params(labelsize=5, colors=MUTED)
        for sp in self._ax.spines.values():
            sp.set_edgecolor(BORDER)
            sp.set_linewidth(0.5)
        self._mean_line = None

        self._hist_canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._hist_canvas.get_tk_widget().pack(padx=6, pady=(0, 6))

        # Init with blank data
        self._ax.bar([0], [0], color=color, alpha=0.0)

    def update_image(self, gray_uint8: np.ndarray):
        """Update thumbnail and histogram from a uint8 grayscale array."""
        # Thumbnail
        pil = Image.fromarray(gray_uint8).resize((THUMB, THUMB), Image.NEAREST)
        self._imgtk = ImageTk.PhotoImage(pil)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self._imgtk)

        # Histogram
        flat = gray_uint8.flatten().astype(np.float64)
        counts, edges = np.histogram(flat, bins=32, range=(0, 255))

        self._ax.clear()
        self._ax.set_facecolor(PLOT_FACE)
        self._ax.bar(edges[:-1], counts,
                     width=(edges[1] - edges[0]),
                     color=self._color, alpha=0.80, linewidth=0)
        mean_val = flat.mean()
        self._ax.axvline(mean_val, color='white', linewidth=0.8,
                         linestyle='--', alpha=0.75)
        self._ax.set_xlim(0, 255)
        self._ax.tick_params(labelsize=5, colors=MUTED)
        self._ax.set_xlabel("", fontsize=0)
        self._ax.yaxis.set_visible(False)
        for sp in self._ax.spines.values():
            sp.set_edgecolor(BORDER)
            sp.set_linewidth(0.5)
        self._ax.text(0.96, 0.85, f"μ={mean_val:.0f}",
                      transform=self._ax.transAxes,
                      ha='right', fontsize=5, color='white', alpha=0.85)
        self._hist_canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
# LIVE FEED PANEL
# ══════════════════════════════════════════════════════════════════════════════
class LiveFeed(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=CARD, bd=0,
                         highlightthickness=1, highlightbackground=BORDER,
                         **kwargs)
        hdr = tk.Frame(self, bg=PANEL, height=32)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)

        tk.Label(hdr, text="LIVE CAMERA FEED",
                 bg=PANEL, fg=ACCENT, font=MONO_LARGE
                 ).pack(side='left', padx=12)
        self.dot = tk.Label(hdr, text="● OFF", bg=PANEL, fg=DANGER,
                             font=BODY_FONT)
        self.dot.pack(side='right', padx=12)

        self.canvas = tk.Canvas(self, width=FEED_W, height=FEED_H,
                                bg='#06080e', highlightthickness=0)
        self.canvas.pack(padx=6, pady=6)
        self._imgtk = None
        self.canvas.create_text(FEED_W // 2, FEED_H // 2,
                                text="Press  ▶ Start  to begin",
                                fill=MUTED, font=('Georgia', 14))

    def update_frame(self, frame_bgr):
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb).resize((FEED_W, FEED_H), Image.LANCZOS)
        self._imgtk = ImageTk.PhotoImage(pil)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self._imgtk)

    def set_live(self, live: bool):
        if live:
            self.dot.config(text="● LIVE", fg=SUCCESS)
        else:
            self.dot.config(text="● OFF", fg=DANGER)
            self.canvas.delete('all')
            self.canvas.create_text(FEED_W // 2, FEED_H // 2,
                                    text="Press  ▶ Start  to begin",
                                    fill=MUTED, font=('Georgia', 14))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class ASLPreprocessingGUI:

    def __init__(self, root: tk.Tk):
        self.root      = root
        self._cap      = None
        self._running  = False
        self._paused   = False
        self._thread   = None
        self._cards    = []
        self._fps_last = time.time()
        self._fps_val  = 0.0

        self._setup_window()
        self._build_ui()

    def _setup_window(self):
        self.root.title("CS.383 — ASL Real-Time Preprocessing Visualizer")
        self.root.configure(bg=BG)
        self.root.geometry('1500x900')
        self.root.minsize(1200, 750)

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=PANEL, height=58)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)

        tk.Label(hdr, text="CS.383",
                 bg=PANEL, fg=ACCENT, font=('Georgia', 18, 'bold')
                 ).pack(side='left', padx=(20, 0), pady=10)
        tk.Label(hdr, text="  ·  Real-Time Preprocessing Pipeline Visualizer",
                 bg=PANEL, fg=TEXT, font=('Georgia', 14)
                 ).pack(side='left', pady=10)

        # Status bar
        sb = tk.Frame(self.root, bg=PANEL, height=30)
        sb.pack(fill='x', side='bottom')
        sb.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(sb, textvariable=self.status_var,
                 bg=PANEL, fg=MUTED, font=BODY_FONT
                 ).pack(side='left', padx=14, pady=5)
        self.fps_var = tk.StringVar(value="")
        tk.Label(sb, textvariable=self.fps_var,
                 bg=PANEL, fg=ACCENT, font=BODY_FONT
                 ).pack(side='right', padx=14)

        # Main area
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True, padx=10, pady=8)

        left = tk.Frame(main, bg=BG)
        left.pack(side='left', fill='y', padx=(0, 10))

        self.feed = LiveFeed(left)
        self.feed.pack()

        self._build_controls(left)
        self._build_pipeline_legend(left)

        right = tk.Frame(main, bg=BG)
        right.pack(side='left', fill='both', expand=True)

        tk.Label(right,
                 text="PREPROCESSING STEPS — all 10 stages  ·  image + histogram",
                 bg=BG, fg=MUTED, font=('Courier New', 10, 'bold')
                 ).pack(anchor='w', pady=(0, 6))

        self._build_step_grid(right)

    def _build_controls(self, parent):
        ctrl = tk.Frame(parent, bg=BG)
        ctrl.pack(fill='x', pady=8)

        cam_row = tk.Frame(ctrl, bg=BG)
        cam_row.pack(fill='x', pady=(0, 8))
        tk.Label(cam_row, text="Camera:", bg=BG, fg=MUTED,
                 font=BODY_FONT).pack(side='left')
        self.cam_var = ttk.Combobox(cam_row,
                                    values=['0  Built-in FaceTime',
                                            '1  iPhone (Continuity)',
                                            '2  External USB',
                                            '3'],
                                    width=24, state='readonly',
                                    font=BODY_FONT)
        self.cam_var.current(0)
        self.cam_var.pack(side='left', padx=(8, 0))

        btn_row = tk.Frame(ctrl, bg=BG)
        btn_row.pack(fill='x')

        def make_btn(text, bg, fg, cmd, state='normal'):
            return tk.Button(btn_row, text=text, bg=bg, fg=fg,
                             activebackground=bg, font=('Georgia', 11, 'bold'),
                             relief='flat', padx=16, pady=8, cursor='hand2',
                             command=cmd, state=state)

        self.start_btn = make_btn("▶  Start",  SUCCESS, 'black', self._start)
        self.start_btn.pack(side='left', padx=(0, 6))

        self.stop_btn = make_btn("■  Stop", DANGER, 'white', self._stop,
                                 state='disabled')
        self.stop_btn.pack(side='left', padx=(0, 6))

        self.pause_btn = make_btn("⏸  Pause", WARNING, 'black',
                                  self._toggle_pause, state='disabled')
        self.pause_btn.pack(side='left')

        fps_row = tk.Frame(ctrl, bg=BG)
        fps_row.pack(fill='x', pady=(10, 0))
        tk.Label(fps_row, text="Update speed:", bg=BG, fg=MUTED,
                 font=BODY_FONT).pack(side='left')
        self.fps_var2 = tk.IntVar(value=10)
        tk.Scale(fps_row, from_=1, to=30, orient='horizontal',
                 variable=self.fps_var2, bg=BG, fg=TEXT,
                 troughcolor=PANEL, highlightthickness=0, length=150,
                 label='fps', font=SMALL).pack(side='left', padx=8)

    def _build_pipeline_legend(self, parent):
        card = tk.Frame(parent, bg=CARD, bd=0,
                        highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill='x', pady=(8, 0))

        tk.Label(card, text="PIPELINE", bg=CARD, fg=ACCENT,
                 font=('Courier New', 10, 'bold')
                 ).pack(anchor='w', padx=10, pady=(8, 4))

        steps = [
            "Original → Grayscale",
            "Gaussian Blur  σ=1",
            "Laplacian Filter",
            "Image Inverse",
            "Gamma Correction  γ=0.45",
            "Log Transform",
            "Histogram Equalization",
            "Contrast Stretching",
            "Sharpening Filter",
            "Sobel Edge Detection",
        ]
        for i, name in enumerate(steps):
            row = tk.Frame(card, bg=CARD)
            row.pack(fill='x', padx=10, pady=2)
            tk.Label(row, text=f"{i+1:2d}", bg=CARD, fg=MUTED,
                     font=BODY_FONT, width=3).pack(side='left')
            tk.Label(row, text="█ ", bg=CARD,
                     fg=STEP_COLORS[i], font=('Courier New', 11)
                     ).pack(side='left')
            tk.Label(row, text=name, bg=CARD, fg=TEXT,
                     font=BODY_FONT).pack(side='left')
        tk.Frame(card, bg=CARD, height=6).pack()

    def _build_step_grid(self, parent):
        dummy = run_pipeline(np.zeros((240, 320, 3), dtype=np.uint8))

        container = tk.Frame(parent, bg=BG)
        container.pack(fill='both', expand=True)

        self._cards = []
        cols = 5

        row_frame = None
        for i, (name, formula, _) in enumerate(dummy):
            if i % cols == 0:
                row_frame = tk.Frame(container, bg=BG)
                row_frame.pack(fill='x', pady=(0, 10))

            card = StepCard(row_frame, i, name, formula)
            card.pack(side='left', padx=(0, 8), ipadx=2, ipady=2)
            card.update_image(np.zeros((THUMB, THUMB), dtype=np.uint8))
            self._cards.append(card)

    # ── Camera control ────────────────────────────────────────────────────────
    def _start(self):
        idx = int(self.cam_var.get().split()[0])
        self._cap = cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            self.status_var.set(f"Cannot open camera {idx}. Try a different index.")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._running = True
        self._paused  = False

        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.pause_btn.config(state='normal')
        self.feed.set_live(True)
        self.status_var.set(f"Camera {idx} — pipeline running")

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.pause_btn.config(state='disabled')
        self.feed.set_live(False)
        self.status_var.set("Stopped.")
        self.fps_var.set("")

    def _toggle_pause(self):
        self._paused = not self._paused
        self.pause_btn.config(
            text="▶  Resume" if self._paused else "⏸  Pause",
            bg=MUTED if self._paused else WARNING
        )
        self.status_var.set(
            "Paused — pipeline frozen on last frame." if self._paused
            else "Resumed — pipeline running."
        )

    def _loop(self):
        while self._running:
            delay = 1.0 / max(self.fps_var2.get(), 1)
            if self._paused:
                time.sleep(0.05)
                continue
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            steps = run_pipeline(frame)
            self.root.after(0, self._update_gui, frame, steps)

            now = time.time()
            self._fps_val  = 1.0 / max(now - self._fps_last, 1e-6)
            self._fps_last = now
            time.sleep(delay)

    def _update_gui(self, frame, steps):
        self.feed.update_frame(frame)
        for card, (_, _, img) in zip(self._cards, steps):
            card.update_image(img)
        self.fps_var.set(f"{self._fps_val:.1f} fps")

    def on_close(self):
        self._running = False
        if self._cap:
            self._cap.release()
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    root = tk.Tk()
    app  = ASLPreprocessingGUI(root)
    root.protocol('WM_DELETE_WINDOW', app.on_close)
    root.mainloop()