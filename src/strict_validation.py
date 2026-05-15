import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report
import time

def load_data(data_dir, split_name):
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

def extract_lbp(images, P, R):
    return [np.histogram(local_binary_pattern(img, P=P, R=R, method='uniform'), bins=np.arange(0, P + 3), density=True)[0] for img in images]

print("Veriler yükleniyor (Train, Valid, Test ayrı ayrı)...")
X_img_train, y_train = load_data("processed_dataset", "train")
X_img_valid, y_valid = load_data("processed_dataset", "valid")
X_img_test, y_test = load_data("processed_dataset", "test")

# Sadece donanım dostu ve en mantıklı adaylar
lbp_params = [(16, 2), (24, 3)] 
rf_n_estimators = [30, 50, 75]
THRESHOLD = 0.40

best_recall = 0
best_params = {}
best_model = None
best_P, best_R = None, None

print("\n--- AŞAMA 1: DOĞRULAMA (VALID) SETİNDE HİPERPARAMETRE SEÇİMİ ---")
for P, R in lbp_params:
    X_train_lbp = extract_lbp(X_img_train, P, R)
    X_valid_lbp = extract_lbp(X_img_valid, P, R)
    
    for n_trees in rf_n_estimators:
        rf = RandomForestClassifier(n_estimators=n_trees, max_depth=10, class_weight={0: 1.0, 1: 2.5}, random_state=42)
        rf.fit(X_train_lbp, y_train)
        
        y_prob_valid = rf.predict_proba(X_valid_lbp)[:, 1]
        y_pred_valid = (y_prob_valid >= THRESHOLD).astype(int)
        rec_valid = recall_score(y_valid, y_pred_valid, pos_label=1)
        
        print(f"Deneniyor -> LBP({P},{R}) | Ağaç: {n_trees} | Valid Recall: %{rec_valid*100:.2f}")
        
        # Sadece Valid setindeki başarıya göre "En İyi"yi seçiyoruz
        if rec_valid > best_recall:
            best_recall = rec_valid
            best_params = {'n_estimators': n_trees, 'max_depth': 10}
            best_model = rf
            best_P, best_R = P, R

print(f"\n[SEÇİLEN MODEL] LBP({best_P},{best_R}) ve {best_params['n_estimators']} Ağaç. Valid Recall: %{best_recall*100:.2f}")

print("\n--- AŞAMA 2: TEST SETİYLE GERÇEK VE TARAFSIZ YÜZLEŞME ---")
# Seçilen LBP parametreleri ile GİZLİ TEST setinin özniteliklerini çıkarıyoruz
X_test_lbp = extract_lbp(X_img_test, best_P, best_R)

# Hiç dokunulmamış test setinde tahmin yapıyoruz
y_prob_test = best_model.predict_proba(X_test_lbp)[:, 1]
y_pred_test = (y_prob_test >= THRESHOLD).astype(int)

final_acc = accuracy_score(y_test, y_pred_test)
final_rec = recall_score(y_test, y_pred_test, pos_label=1)

print(f"GERÇEK Accuracy (Doğruluk): %{final_acc*100:.2f}")
print(f"GERÇEK Recall (Hastalık Yakalama): %{final_rec*100:.2f}")
print("\nGerçek Sınıflandırma Raporu (Sadece Test Seti):")
print(classification_report(y_test, y_pred_test, target_names=["Sağlıklı (0)", "Grasserie (1)"]))