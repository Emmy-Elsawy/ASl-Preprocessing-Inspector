# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║         CS.383 — Real-Time Sign Language Translator                        ║
# ║         FILE 1 OF 2: PREPROCESSING PIPELINE                                ║
# ║                                                                              ║
# ║  Run this FIRST to preprocess the Sign Language MNIST dataset.             ║
# ║  Outputs: X_train_preprocessed.npy, y_train.npy,                           ║
# ║            X_test_preprocessed.npy,  y_test.npy                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to use a non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import cv2
from skimage import io, filters, transform, data

# ──────────────────────────────────────────────────────────────────────────────
# PATHS — adjust to your Kaggle / local dataset location
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_CSV = '/Users/emmyel-sawy/Desktop/ASL Preprocessing Inspector/sign_mnist_train/sign_mnist_train.csv'   # ← change if needed
TEST_CSV  = '/Users/emmyel-sawy/Desktop/ASL Preprocessing Inspector/sign_mnist_test/sign_mnist_test.csv'    # ← change if needed

OUTPUT_DIR = '.'                     # where .npy files will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# LAB 6 — IMAGE LOADING & FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_blur(image, sigma=1):
    """Lab 6 — filters.gaussian(): smooth image to reduce noise."""
    return filters.gaussian(image, sigma=sigma)


def laplacian_filter(image):
    """Lab 6 — filters.laplace(): highlight sharp brightness changes."""
    return filters.laplace(image)


def median_blur(image, kernel_size=3):
    """Lab 6 — cv2.medianBlur(): remove salt-and-pepper noise."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    return cv2.medianBlur(image, kernel_size)


# ══════════════════════════════════════════════════════════════════════════════
# LAB 7 — INTENSITY TRANSFORMATIONS
# ══════════════════════════════════════════════════════════════════════════════

def image_inverse(image):
    """Lab 7 — Image Inverse: dark ↔ light swap."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    return 255 - image


def gamma_correction(image, gamma=0.5):
    """Lab 7 — Power-law transform. gamma<1 brightens, gamma>1 darkens."""
    image = image.astype(np.float64)
    if image.max() > 1.0:
        image = image / 255.0
    return image ** gamma


def log_transform(image):
    """Lab 7 — Log transform: brightens dark pixels, reveals shadow detail."""
    image_f = image.astype(float)
    return (255.0 * np.log(1 + image_f) / np.log(256)).astype(int)


def histogram_equalization(image):
    """
    Lab 7 — Histogram Equalization via CDF remapping.
    Redistributes intensities to span full 0–255 range.
    """
    flatten   = image.flatten()
    hist, _   = np.histogram(image, 256, [0, 255])
    cdf       = hist.cumsum()
    cdf_m     = np.ma.masked_equal(cdf, 0)
    cdf_m     = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf       = np.ma.filled(cdf_m, 0).astype('uint8')
    return np.reshape(cdf[flatten], image.shape)


# ══════════════════════════════════════════════════════════════════════════════
# LAB 8 — CONTRAST & EDGE OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def contrast_stretching(image):
    """Lab 8 — Linear expansion to full 0–255 range using actual min/max."""
    b, a = image.max(), image.min()
    if b == a:
        return image.astype(np.uint8)
    return (255 * (image.astype(float) - a) / (b - a)).astype(np.uint8)


def sharpening_filter(image):
    """Lab 8 — cv2.filter2D with 5-centre sharpening kernel."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    kernel = np.array([[ 0, -1,  0],
                        [-1,  5, -1],
                        [ 0, -1,  0]])
    return cv2.filter2D(image, -1, kernel)


def sobel_edge_detection(image):
    """Lab 8 — Sobel magnitude: sqrt(Gx² + Gy²) across both axes."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(sobelx, sobely).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# PROJECT — PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_single(img: np.ndarray) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to one 28×28 uint8 grayscale image.

    Steps:
        1. Histogram equalization  (Lab 7)
        2. Contrast stretching     (Lab 8)
        3. Sharpening filter       (Lab 8)
        4. Sobel edge detection    (Lab 8)
        5. Normalize to [0.0, 1.0]

    Returns float32 (28, 28) array.
    """
    eq  = histogram_equalization(img)
    st  = contrast_stretching(eq)
    sh  = sharpening_filter(st)
    ed  = np.clip(sobel_edge_detection(sh), 0, 255).astype(np.uint8)
    return ed.astype(np.float32) / 255.0

def preprocess_webcam_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare a BGR webcam frame for model inference.

    Steps (optimised for real-time use):
        1. Convert BGR → grayscale
        2. Resize to 28×28
        3. Gaussian blur (σ=1) — denoise
        4. Histogram equalization
        5. Contrast stretching
        6. Sharpening filter
        7. Sobel edge detection
        8. Normalize to [0, 1]

    Returns:
        tensor  — float32 (1, 28, 28, 1) ready for Keras model
        preview — uint8   (28, 28)       for on-screen display
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    blr   = (gaussian_blur(small.astype(float) / 255.0, sigma=1) * 255).astype(np.uint8)
    proc  = preprocess_single(blr)
    preview = (proc * 255).astype(np.uint8)
    tensor  = proc.reshape(1, 28, 28, 1)
    return tensor, preview


def load_csv(csv_path: str, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Sign Language MNIST CSV into raw NumPy arrays.

    CSV columns: label, pixel1 … pixel784
    Returns: X (N, 28, 28) uint8,  y (N,) int
    """
    labels, images = [], []
    with open(csv_path, newline='') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if max_samples and i >= max_samples:
                break
            labels.append(int(row['label']))
            pixels = np.array([int(v) for k, v in row.items() if k != 'label'], dtype=np.uint8)
            images.append(pixels.reshape(28, 28))
    return np.stack(images), np.array(labels)


def preprocess_dataset(csv_path: str,
                        max_samples: int | None = None,
                        show_samples: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess an entire Sign Language MNIST CSV file.

    Args:
        csv_path    : path to sign_mnist_train.csv or sign_mnist_test.csv
        max_samples : cap number of rows (None = all)
        show_samples: how many raw-vs-processed comparisons to display

    Returns:
        X_processed : float32 (N, 28, 28)
        y           : int     (N,)
    """
    print(f"\n{'='*60}")
    print(f"Loading: {os.path.basename(csv_path)}")
    print(f"{'='*60}")

    X_raw, y = load_csv(csv_path, max_samples)
    print(f"✓ Loaded {len(X_raw)} samples | classes: {len(np.unique(y))} letters")

    X_processed = np.zeros(len(X_raw), dtype=object)
    X_out = np.zeros_like(X_raw, dtype=np.float32)

    print("Preprocessing…")
    for i, img in enumerate(X_raw):
        X_out[i] = preprocess_single(img)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1:>6} / {len(X_raw)}")

    print(f"✓ Done — shape: {X_out.shape}  range: [{X_out.min():.3f}, {X_out.max():.3f}]")

    # ── Visual comparison ────────────────────────────────────────────────
    LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    print(f"DEBUG: Plotting {show_samples} samples from an array of size {len(X_raw)}")
    if show_samples > 0 and len(X_raw) >= show_samples:
        fig, axes = plt.subplots(2, show_samples, figsize=(show_samples * 3.0, 6), constrained_layout=True)
        fig.suptitle(f'Sign Language MNIST — Raw vs Preprocessed  ({os.path.basename(csv_path)})',
                     fontsize=13, fontweight='bold')
        for j in range(show_samples):
            axes[0, j].imshow(X_raw[j],  cmap='gray')
            axes[0, j].set_title(f'RAW\n{LETTERS[y[j]]}', fontsize=9)
            axes[0, j].axis('off')
            axes[1, j].imshow(X_out[j],  cmap='gray')
            axes[1, j].set_title('PREPROCESSED', fontsize=9)
            axes[1, j].axis('off')
        plt.tight_layout()
        plt.show()

    # ── Class distribution ───────────────────────────────────────────────
    unique, counts = np.unique(y, return_counts=True)
    fig2, ax = plt.subplots(figsize=(18, 8))
    ax.bar([LETTERS[u] for u in unique], counts, color='steelblue', edgecolor='white')
    ax.set_title('Samples per Class', fontsize=12, fontweight='bold')
    ax.set_xlabel('ASL Letter'); ax.set_ylabel('Count')
    plt.tight_layout(); plt.show()

    return X_out, y


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — run preprocessing and save .npy files
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("CS.383 — ASL Preprocessing Pipeline")
    print("This will process the full dataset and save .npy files.\n")

    # Check files exist
    for path in (TRAIN_CSV, TEST_CSV):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cannot find: {path}\n"
                "Place the Kaggle CSVs in the same folder as this script,\n"
                "or update TRAIN_CSV / TEST_CSV at the top of the file."
            )

    X_train, y_train = preprocess_dataset(TRAIN_CSV, show_samples=5)
    X_test,  y_test  = preprocess_dataset(TEST_CSV,  show_samples=5)

    # Reshape to (N, 28, 28, 1) for Keras
    X_train_cnn = X_train.reshape(-1, 28, 28, 1)
    X_test_cnn  = X_test.reshape(-1,  28, 28, 1)

    # Save
    np.save(os.path.join(OUTPUT_DIR, 'X_train_preprocessed.npy'), X_train_cnn)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'),              y_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_test_preprocessed.npy'),  X_test_cnn)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'),               y_test)

    print("\n" + "="*60)
    print("SAVED:")
    print(f"  X_train_preprocessed.npy  {X_train_cnn.shape}")
    print(f"  y_train.npy               {y_train.shape}")
    print(f"  X_test_preprocessed.npy   {X_test_cnn.shape}")
    print(f"  y_test.npy                {y_test.shape}")
    print("\nLoad with:")
    print("  X_train = np.load('X_train_preprocessed.npy')")
    print("  y_train = np.load('y_train.npy')")
    print("="*60)
