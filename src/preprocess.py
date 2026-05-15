import os
import cv2
import numpy as np
import glob

def process_and_crop_dataset(input_dir, output_dir, target_size=(64, 64)):
    """YOLO formatındaki veriyi okur, BB'leri kırpar, gri yapar ve kaydeder."""
    for split in ['train', 'valid', 'test']:
        img_path = os.path.join(input_dir, split, 'images')
        label_path = os.path.join(input_dir, split, 'labels')
        
        out_split_dir = os.path.join(output_dir, split)
        os.makedirs(out_split_dir, exist_ok=True)
        
        # Etiket dosyalarını bul
        txt_files = glob.glob(os.path.join(label_path, "*.txt"))
        
        for txt_file in txt_files:
            base_name = os.path.basename(txt_file).replace('.txt', '')
            # Görüntü formatı jpg veya png olabilir
            img_file = os.path.join(img_path, base_name + '.jpg')
            if not os.path.exists(img_file):
                 img_file = os.path.join(img_path, base_name + '.png')
                 if not os.path.exists(img_file): continue
            
            img = cv2.imread(img_file)
            h, w, _ = img.shape
            
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                
            for idx, line in enumerate(lines):
                # YOLO formatı: class_id x_center y_center width height
                parts = line.strip().split()
                if len(parts) != 5: continue
                
                class_id = int(parts[0])
                x_center, y_center, bw, bh = map(float, parts[1:])
                
                # Piksel koordinatlarına çevir
                x1 = int((x_center - bw / 2) * w)
                y1 = int((y_center - bh / 2) * h)
                x2 = int((x_center + bw / 2) * w)
                y2 = int((y_center + bh / 2) * h)
                
                # Sınırları kontrol et
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 - x1 < 10 or y2 - y1 < 10: continue # Çok küçük kırpmaları atla
                
                # 1. Kırp (Crop)
                cropped_img = img[y1:y2, x1:x2]
                # 2. Gri Tonlamaya Çevir (Grayscale)
                gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                # 3. Yeniden Boyutlandır (Resize)
                resized_img = cv2.resize(gray_img, target_size)
                
                # Kaydet
                out_filename = os.path.join(out_split_dir, f"{base_name}_{idx}_c{class_id}.jpg")
                cv2.imwrite(out_filename, resized_img)

dataset_path = "silkworm_dataset_yolov8"
process_and_crop_dataset(dataset_path, "processed_dataset", target_size=(64, 64))
print("Ön işleme tamamlandı!")