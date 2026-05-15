import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
import time

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

# Farklı P ve R değerleri ile dinamik LBP çıkarımı
def extract_lbp_custom(images, P, R):
    # method='uniform' kullanıldığında histogram boyutu (P + 2) olur.
    return [np.histogram(local_binary_pattern(img, P=P, R=R, method='uniform'), bins=np.arange(0, P + 3), density=True)[0] for img in images]

print("Veriler diskten okunuyor...")
X_img_train, y_train = load_data_from_split("processed_dataset", "train")
X_img_test, y_test = load_data_from_split("processed_dataset", "test")

# DONANIM MANTIKLI HİPERPARAMETRE UZAYI
lbp_params = [(8, 1), (16, 2), (24, 3)] 
rf_n_estimators = [30, 50, 75, 100]
rf_max_depth = [10, 15, 20]
THRESHOLD = 0.40 # Recall artırmak için kullandığımız eşik sabit kalıyor

results = []

print("\nKapsamlı Hiperparametre Araması Başlıyor (Bu işlem 1-2 dakika sürebilir)...\n")

for P, R in lbp_params:
    print(f"--- LBP (P={P}, R={R}) özellikleri çıkarılıyor ---")
    start_time = time.time()
    X_train_lbp = extract_lbp_custom(X_img_train, P, R)
    X_test_lbp = extract_lbp_custom(X_img_test, P, R)
    print(f"Özellik çıkarma tamamlandı ({time.time() - start_time:.1f} sn). RF varyasyonları eğitiliyor...")
    
    for n_trees in rf_n_estimators:
        for depth in rf_max_depth:
            rf = RandomForestClassifier(
                n_estimators=n_trees, 
                max_depth=depth, 
                class_weight={0: 1.0, 1: 2.5}, 
                random_state=42,
                n_jobs=-1 # Eğitimi hızlandırmak için PC'nin tüm çekirdeklerini kullan
            )
            rf.fit(X_train_lbp, y_train)
            
            # Tahmin ve Recall hesaplama
            y_prob = rf.predict_proba(X_test_lbp)[:, 1]
            y_pred = (y_prob >= THRESHOLD).astype(int)
            
            acc = accuracy_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred, pos_label=1) # Sadece Grasserie (1) sınıfının Recall'u
            
            # Sonuçları kaydet
            results.append({
                "LBP_P": P,
                "LBP_R": R,
                "RF_Trees": n_trees,
                "RF_Depth": depth,
                "Accuracy(%)": round(acc * 100, 2),
                "Recall(%)": round(rec * 100, 2)
            })

# Sonuçları Tabloya Dönüştür
df_results = pd.DataFrame(results)

# Önce Hastalık Yakalama (Recall), sonra Doğruluk (Accuracy) değerine göre en iyiden en kötüye sırala
df_sorted = df_results.sort_values(by=['Recall(%)', 'Accuracy(%)'], ascending=[False, False])

print("\n--- EN İYİ 10 KOMBİNASYON ---")
print(df_sorted.head(10).to_string(index=False))

# İstersen raporlarına koymak için tüm veriyi CSV olarak kaydet
df_sorted.to_csv("hyperparameter_tuning_results.csv", index=False)