import pandas as pd
import cv2
import os
from ml_adaptive_preprocessing import ml_preprocess_image

output_dir = "ml_adaptive_enhanced_images"
os.makedirs(output_dir, exist_ok=True)

def process_and_save(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    enhanced = ml_preprocess_image(img)
    out_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, enhanced)
    return out_path

df['enhanced_path'] = df['image_path'].apply(process_and_save)
df.dropna(subset=['enhanced_path'], inplace=True)
