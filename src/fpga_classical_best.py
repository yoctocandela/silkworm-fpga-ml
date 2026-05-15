"""
fpga_classical_best.py
======================
"Best-effort" FPGA-friendly classical-ML pipeline for silkworm disease
detection, designed for direct comparison against strict_validation_fixed.py.

Design changes vs. the current LBP+RF model
-------------------------------------------
1. Keep RGB instead of dropping to grayscale (color is a strong cue for
   Grasserie's cuticle discoloration; trivial cost on FPGA).
2. Add 2x2 spatial pooling so the classifier can see head / mid / tail
   regions of the worm separately.
3. Add a magnitude-weighted gradient-direction histogram per cell
   (Sobel + 8-bin atan2-quadrant lookup -- all integer / lookup-table on FPGA).
4. Same recall-first training:
      class_weight = {Grasserie:2.5, Healthy:1.0}
      decide Grasserie when  P(Grasserie) >= 0.40
   and the same strict, leakage-free 89/6/6 split.

Feature vector (per cropped worm, 64x64):
    4 cells x ( LBP P=8 histogram (10 bins)
              + grad-dir histogram  ( 8 bins)
              + mean B, G, R        ( 3 vals) )
    = 4 x 21  =  84 features
(vs. 10 features in the current LBP-only model)

FPGA resource estimate (rough, pre-synthesis):
    current LBP+RF       :  ~10k LUTs, 0 DSPs
    this pipeline        :  ~18k LUTs, 4 DSPs   (still <35% of Zynq-7020)
    end-to-end latency   :  ~25-30 us / worm at 150 MHz (same streaming pace)

Run:
    pip install opencv-python scikit-image scikit-learn numpy --break-system-packages
    python fpga_classical_best.py
"""

import glob
import os
import time

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)


# ---- Project-specific constants ---------------------------------------------
DATASET_DIR        = "silkworm_dataset_yolov8"   # original YOLO export
TARGET_SIZE        = 64
GRID               = 2                            # 2x2 spatial cells
LBP_P, LBP_R       = 8, 1                         # FPGA-friendly LBP
HOG_BINS           = 8                            # gradient orientation bins
THRESHOLD_DISEASE  = 0.40
DISEASE_CLASS      = 0                            # Grasserie per data.yaml
HEALTHY_CLASS      = 1                            # Healthy   per data.yaml
CLASS_NAMES        = ["Grasserie (0)", "Healthy (1)"]


# ---- Data loading (re-crop from RGB, do NOT use processed_dataset) ----------
def load_color_crops(split):
    img_dir = os.path.join(DATASET_DIR, split, "images")
    lbl_dir = os.path.join(DATASET_DIR, split, "labels")
    X, y = [], []
    for txt_file in glob.glob(os.path.join(lbl_dir, "*.txt")):
        base = os.path.basename(txt_file).replace(".txt", "")
        img_file = os.path.join(img_dir, base + ".jpg")
        if not os.path.exists(img_file):
            img_file = os.path.join(img_dir, base + ".png")
            if not os.path.exists(img_file):
                continue
        img = cv2.imread(img_file)        # BGR uint8
        if img is None:
            continue
        h, w, _ = img.shape
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                x1 = max(0, int((xc - bw / 2) * w))
                y1 = max(0, int((yc - bh / 2) * h))
                x2 = min(w, int((xc + bw / 2) * w))
                y2 = min(h, int((yc + bh / 2) * h))
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                crop = img[y1:y2, x1:x2]
                crop = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE))
                X.append(crop)
                y.append(cls)
    return X, np.array(y)


# ---- Feature extraction -----------------------------------------------------
def cell_slices(size=TARGET_SIZE, grid=GRID):
    step = size // grid
    return [(r * step, (r + 1) * step, c * step, (c + 1) * step)
            for r in range(grid) for c in range(grid)]


def extract_features(images):
    """LBP + gradient-orientation + color mean, pooled over 2x2 cells."""
    cells = cell_slices()
    out = []
    for bgr in images:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Sobel-based gradient (unsigned orientation: 0..pi)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = np.arctan2(gy, gx) % np.pi
        bins = np.minimum((ang / np.pi * HOG_BINS).astype(np.int32),
                          HOG_BINS - 1)

        # uniform LBP over the whole image (cheap)
        lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")

        feats = []
        for (r1, r2, c1, c2) in cells:
            # 1) LBP histogram, P+2 bins
            lbp_hist, _ = np.histogram(
                lbp[r1:r2, c1:c2],
                bins=np.arange(0, LBP_P + 3),
                density=True,
            )
            # 2) Magnitude-weighted gradient histogram
            cb = bins[r1:r2, c1:c2].ravel()
            cm = mag[r1:r2, c1:c2].ravel()
            ghist = np.bincount(cb, weights=cm, minlength=HOG_BINS)
            s = ghist.sum()
            if s > 0:
                ghist = ghist / s
            # 3) Mean per channel (B, G, R) normalised to [0, 1]
            color_mean = bgr[r1:r2, c1:c2].reshape(-1, 3).mean(axis=0) / 255.0

            feats.extend(lbp_hist)
            feats.extend(ghist)
            feats.extend(color_mean)
        out.append(feats)
    return np.array(out, dtype=np.float32)


# ---- Inference rule ---------------------------------------------------------
def predict_disease(model, X):
    proba_disease = model.predict_proba(X)[:, DISEASE_CLASS]
    return np.where(proba_disease >= THRESHOLD_DISEASE,
                    DISEASE_CLASS, HEALTHY_CLASS)


# ---- Main -------------------------------------------------------------------
def main():
    print(">> Loading RGB crops directly from the YOLO export...")
    t0 = time.time()
    X_train, y_train = load_color_crops("train")
    X_valid, y_valid = load_color_crops("valid")
    X_test,  y_test  = load_color_crops("test")
    print(f"   loaded in {time.time() - t0:.1f}s")
    print(f"   train: {len(y_train):>5d}   "
          f"Grasserie={int((y_train == 0).sum())}, "
          f"Healthy={int((y_train == 1).sum())}")
    print(f"   valid: {len(y_valid):>5d}   "
          f"Grasserie={int((y_valid == 0).sum())}, "
          f"Healthy={int((y_valid == 1).sum())}")
    print(f"   test : {len(y_test):>5d}   "
          f"Grasserie={int((y_test == 0).sum())}, "
          f"Healthy={int((y_test == 1).sum())}")

    print("\n>> Extracting features (LBP + grad-hist + color, 2x2 cells)...")
    t0 = time.time()
    F_train = extract_features(X_train)
    F_valid = extract_features(X_valid)
    F_test  = extract_features(X_test)
    feat_time = (time.time() - t0) / max(len(y_train) + len(y_valid) + len(y_test), 1) * 1000
    print(f"   feature dim: {F_train.shape[1]}  "
          f"(~{feat_time:.3f} ms/sample on CPU, single-thread Python)")

    print("\n>> Validation-set hyperparameter sweep (target = Grasserie recall):")
    grid = [(n, d) for n in (50, 75, 100) for d in (10, 15, 20)]
    best = None
    best_val_rec = -1.0
    for n, d in grid:
        rf = RandomForestClassifier(
            n_estimators=n,
            max_depth=d,
            class_weight={DISEASE_CLASS: 2.5, HEALTHY_CLASS: 1.0},
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(F_train, y_train)
        y_pred_v = predict_disease(rf, F_valid)
        rec = recall_score(y_valid, y_pred_v, pos_label=DISEASE_CLASS)
        acc = accuracy_score(y_valid, y_pred_v)
        prec = precision_score(y_valid, y_pred_v,
                               pos_label=DISEASE_CLASS, zero_division=0)
        print(f"   n_estimators={n:3d}, max_depth={d:2d}  ->  "
              f"val acc {acc*100:5.2f}%   "
              f"recall_G {rec*100:5.2f}%   "
              f"prec_G {prec*100:5.2f}%")
        if rec > best_val_rec:
            best_val_rec = rec
            best = (rf, n, d)

    rf, n, d = best
    print(f"\n>> Selected: n_estimators={n}, max_depth={d}  "
          f"(val Grasserie recall {best_val_rec*100:.2f}%)")

    print("\n>> Held-out TEST evaluation (no further tuning):")
    t0 = time.time()
    y_pred = predict_disease(rf, F_test)
    infer_ms = (time.time() - t0) / max(len(y_test), 1) * 1000
    print(classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    cm = confusion_matrix(y_test, y_pred,
                          labels=[DISEASE_CLASS, HEALTHY_CLASS])
    print("Confusion matrix [rows=true, cols=pred] order: Grasserie, Healthy")
    print(cm)

    print("\n============ SUMMARY (FPGA-best classical) — copy into chat ============")
    print(f"   Feature dim          : {F_train.shape[1]}")
    print(f"   RF                   : n_estimators={n}, max_depth={d}, "
          f"class_weight={{0:2.5, 1:1.0}}, threshold(P_G)>={THRESHOLD_DISEASE}")
    print(f"   Test accuracy        : "
          f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"   Test Grasserie recall    : "
          f"{recall_score(y_test, y_pred, pos_label=DISEASE_CLASS)*100:.2f}%")
    print(f"   Test Grasserie precision : "
          f"{precision_score(y_test, y_pred, pos_label=DISEASE_CLASS, zero_division=0)*100:.2f}%")
    print(f"   Test Grasserie F1        : "
          f"{2 * recall_score(y_test, y_pred, pos_label=DISEASE_CLASS) * precision_score(y_test, y_pred, pos_label=DISEASE_CLASS, zero_division=0) / max((recall_score(y_test, y_pred, pos_label=DISEASE_CLASS) + precision_score(y_test, y_pred, pos_label=DISEASE_CLASS, zero_division=0)), 1e-9) * 100:.2f}%")
    print(f"   Test Healthy recall      : "
          f"{recall_score(y_test, y_pred, pos_label=HEALTHY_CLASS)*100:.2f}%")
    print(f"   Inference per sample : {infer_ms:.3f} ms (CPU)")
    print("=======================================================================")


if __name__ == "__main__":
    main()
