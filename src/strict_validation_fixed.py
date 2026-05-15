"""
strict_validation_fixed.py
==========================
Corrects the class-label bug found in the original training scripts:

  data.yaml: names = ['Grasserie', 'Healthy']
  =>  class 0  =  Grasserie  (DISEASED, the class we want to catch)
  =>  class 1  =  Healthy

The original strict_validation.py / train_models.py used:
  - class_weight = {0: 1.0, 1: 2.5}   (REVERSED: this boosts Healthy)
  - pos_label = 1 for "disease recall" (REVERSED: 1 is Healthy)
  - threshold on predict_proba[:, 1] (REVERSED: this lowers the bar for Healthy)

This fixed script:
  1) Runs the validation-set hyperparameter sweep with the correct
     class_weight = {0: 2.5, 1: 1.0} (boost Grasserie).
  2) Selects the best model on the VALIDATION set by Grasserie recall.
  3) Reports full per-class metrics on the untouched TEST set, plus
     confusion matrix and inference timing.
  4) Writes hyperparameter_tuning_results_fixed.csv next to this file.

Run it from the project root:

    python src/strict_validation_fixed.py
"""

import os
import time

import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# --- Project-specific constants -----------------------------------------------
DATA_DIR = "processed_dataset"          # produced by preprocess.py
THRESHOLD_DISEASE = 0.40                # keep original "easy-to-call-disease" bar
DISEASE_CLASS = 0                       # Grasserie (per data.yaml)
HEALTHY_CLASS = 1                       # Healthy   (per data.yaml)
CLASS_NAMES = ["Grasserie (0)", "Healthy (1)"]

# Same hyperparameter grid as plot_rf_tuning.py
LBP_PARAMS = [(8, 1), (16, 2), (24, 3)]
RF_N_ESTIMATORS = [30, 50, 75, 100]
RF_MAX_DEPTH = [10, 15, 20]


# --- Helpers ------------------------------------------------------------------
def load_data(split_name):
    X, y = [], []
    split_dir = os.path.join(DATA_DIR, split_name)
    if not os.path.exists(split_dir):
        return X, np.array(y)
    for file in os.listdir(split_dir):
        if file.endswith(".jpg"):
            img = cv2.imread(os.path.join(split_dir, file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                class_id = int(file.split("_c")[-1].split(".jpg")[0])
                X.append(img)
                y.append(class_id)
    return X, np.array(y)


def extract_lbp(images, P, R):
    """Uniform-LBP histogram, P+2 bins."""
    out = []
    for img in images:
        lbp = local_binary_pattern(img, P=P, R=R, method="uniform")
        hist, _ = np.histogram(lbp, bins=np.arange(0, P + 3), density=True)
        out.append(hist)
    return out


def predict_disease(model, X):
    """Predict Grasserie (0) when P(class=0) >= THRESHOLD_DISEASE.

    This is the corrected version of the "lower threshold for the
    sensitive class" trick: we drop the bar for the DISEASE class.
    """
    proba_disease = model.predict_proba(X)[:, DISEASE_CLASS]
    return np.where(proba_disease >= THRESHOLD_DISEASE,
                    DISEASE_CLASS, HEALTHY_CLASS)


# --- Main ---------------------------------------------------------------------
def main():
    print(">> Loading train / valid / test splits...")
    X_img_train, y_train = load_data("train")
    X_img_valid, y_valid = load_data("valid")
    X_img_test,  y_test  = load_data("test")

    print(f"   train: {len(y_train):>5d}   "
          f"Grasserie(0)={int((y_train == 0).sum())}, "
          f"Healthy(1)={int((y_train == 1).sum())}")
    print(f"   valid: {len(y_valid):>5d}   "
          f"Grasserie(0)={int((y_valid == 0).sum())}, "
          f"Healthy(1)={int((y_valid == 1).sum())}")
    print(f"   test : {len(y_test):>5d}   "
          f"Grasserie(0)={int((y_test == 0).sum())}, "
          f"Healthy(1)={int((y_test == 1).sum())}")

    results = []
    best_recall = -1.0
    best = None

    print("\n>> Hyperparameter sweep on VALIDATION set (target = Grasserie recall):")
    for P, R in LBP_PARAMS:
        t0 = time.time()
        X_train_lbp = extract_lbp(X_img_train, P, R)
        X_valid_lbp = extract_lbp(X_img_valid, P, R)
        print(f"   LBP(P={P}, R={R}) features built in {time.time() - t0:.1f}s")

        for n_trees in RF_N_ESTIMATORS:
            for depth in RF_MAX_DEPTH:
                rf = RandomForestClassifier(
                    n_estimators=n_trees,
                    max_depth=depth,
                    class_weight={DISEASE_CLASS: 2.5, HEALTHY_CLASS: 1.0},  # FIXED
                    random_state=42,
                    n_jobs=-1,
                )
                rf.fit(X_train_lbp, y_train)

                y_pred = predict_disease(rf, X_valid_lbp)
                acc      = accuracy_score(y_valid, y_pred)
                rec_dis  = recall_score (y_valid, y_pred, pos_label=DISEASE_CLASS)
                rec_hlt  = recall_score (y_valid, y_pred, pos_label=HEALTHY_CLASS)
                prec_dis = precision_score(y_valid, y_pred,
                                           pos_label=DISEASE_CLASS, zero_division=0)
                f1_dis   = f1_score(y_valid, y_pred,
                                    pos_label=DISEASE_CLASS, zero_division=0)

                results.append({
                    "LBP_P":  P,
                    "LBP_R":  R,
                    "RF_Trees": n_trees,
                    "RF_Depth": depth,
                    "Accuracy(%)":          round(acc      * 100, 2),
                    "Recall_Grasserie(%)":  round(rec_dis  * 100, 2),
                    "Precision_Grasserie(%)": round(prec_dis * 100, 2),
                    "F1_Grasserie(%)":      round(f1_dis   * 100, 2),
                    "Recall_Healthy(%)":    round(rec_hlt  * 100, 2),
                })

                if rec_dis > best_recall:
                    best_recall = rec_dis
                    best = {
                        "model": rf,
                        "P": P, "R": R,
                        "n_trees": n_trees, "depth": depth,
                    }

    df = pd.DataFrame(results).sort_values(
        by=["Recall_Grasserie(%)", "Accuracy(%)"], ascending=[False, False]
    )
    df.to_csv("hyperparameter_tuning_results_fixed.csv", index=False)
    print("\nTop 10 configs by VALIDATION Grasserie recall:")
    print(df.head(10).to_string(index=False))

    print(f"\n>> Selected model (validation): "
          f"LBP(P={best['P']},R={best['R']}) + "
          f"RF(n={best['n_trees']}, depth={best['depth']}) | "
          f"val Grasserie-recall = {best_recall * 100:.2f}%")

    # ---- Final, untouched TEST evaluation ------------------------------------
    print("\n>> Held-out TEST evaluation (no further tuning):")
    X_test_lbp = extract_lbp(X_img_test, best["P"], best["R"])

    t0 = time.time()
    y_pred_test = predict_disease(best["model"], X_test_lbp)
    infer_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred_test)
    print(f"   Accuracy (overall)         : {acc * 100:.2f}%")
    if len(y_test):
        print(f"   Inference time (test set)  : {infer_time * 1000:.1f} ms total "
              f"({infer_time * 1000 / len(y_test):.3f} ms/sample, "
              f"sklearn RF on CPU)")

    print("\n   Per-class report:")
    print(classification_report(
        y_test, y_pred_test, target_names=CLASS_NAMES, digits=4
    ))

    cm = confusion_matrix(
        y_test, y_pred_test, labels=[DISEASE_CLASS, HEALTHY_CLASS]
    )
    print("   Confusion matrix [rows=true, cols=pred] order: Grasserie, Healthy")
    print(cm)

    # Plain summary block at the bottom, easy to copy into the report
    print("\n================ SUMMARY (copy into report) ================")
    print(f"   LBP                : P={best['P']}, R={best['R']}")
    print(f"   Random Forest      : n_estimators={best['n_trees']}, "
          f"max_depth={best['depth']}, class_weight={{0:2.5, 1:1.0}}, "
          f"threshold(P_grasserie)>={THRESHOLD_DISEASE:.2f}")
    print(f"   Test accuracy      : {acc * 100:.2f}%")
    print(f"   Test Grasserie recall    : "
          f"{recall_score(y_test, y_pred_test, pos_label=DISEASE_CLASS) * 100:.2f}%")
    print(f"   Test Grasserie precision : "
          f"{precision_score(y_test, y_pred_test, pos_label=DISEASE_CLASS, zero_division=0) * 100:.2f}%")
    print(f"   Test Healthy recall      : "
          f"{recall_score(y_test, y_pred_test, pos_label=HEALTHY_CLASS) * 100:.2f}%")
    print(f"   Inference (per sample)   : "
          f"{infer_time * 1000 / max(len(y_test), 1):.3f} ms (CPU)")
    print("============================================================")


if __name__ == "__main__":
    main()
