import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from time import time

def load_data_from_split(data_dir, split_name):
    X, y = [], []
    split_dir = os.path.join(data_dir, split_name)
    if not os.path.exists(split_dir): return X, np.array(y)
    for file in os.listdir(split_dir):
        if file.endswith('.jpg'):
            img = cv2.imread(os.path.join(split_dir, file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                class_id = int(file.split('_c')[-1].split('.jpg')[0])
                X.append(img)
                y.append(class_id)
    return X, np.array(y)

# LBP Güçlendirildi: Radius 2, Points 16 (Daha geniş doku analizi)
def extract_lbp_enhanced(images):
    return [np.histogram(local_binary_pattern(img, P=16, R=2, method='uniform'), bins=np.arange(0, 19), density=True)[0] for img in images]

print("Veriler yükleniyor...")
X_img_train, y_train = load_data_from_split("processed_dataset", "train")
X_img_test, y_test = load_data_from_split("processed_dataset", "test")

print("Güçlendirilmiş LBP Öznitelikleri Çıkarılıyor...")
X_train_lbp = extract_lbp_enhanced(X_img_train)
X_test_lbp = extract_lbp_enhanced(X_img_test)

print("\nModel Eğitiliyor...")
# Hiperparametreler Donanım Sınırlarına Göre Optimize Edildi
# class_weight: Grasserie (1) sınıfına 2.5 kat daha fazla önem veriyoruz!
rf_model = RandomForestClassifier(
    n_estimators=50, 
    max_depth=15, 
    class_weight={0: 1.0, 1: 2.5}, 
    random_state=42
)

start_time = time()
rf_model.fit(X_train_lbp, y_train)
train_time = time() - start_time

# Klasik %50 Karar Eşiği Yerine Olasılık Çıkarımı
start_time = time()
y_prob = rf_model.predict_proba(X_test_lbp)[:, 1] # Modelin "Grasserie" olma olasılığı tahmini
infer_time = time() - start_time

# Eşik Değerini Düşürüyoruz: %40 şüphe bile "Hasta" (1) demek için yeterli
THRESHOLD = 0.40 
y_pred_custom = (y_prob >= THRESHOLD).astype(int)

# Metrikler
acc = accuracy_score(y_test, y_pred_custom)
report = classification_report(y_test, y_pred_custom, target_names=["Sağlıklı (0)", "Grasserie (1)"])

print(f"\n--- SONUÇLAR (Eşik: %{int(THRESHOLD*100)}, Ağaç: 50) ---")
print(f"Accuracy (Doğruluk): %{acc*100:.2f}")
print(f"Eğitim Süresi: {train_time:.3f} saniye")
print(f"Çıkarım Süresi: {infer_time:.4f} saniye\n")
print("Detaylı Sınıflandırma Raporu:")
print(report)