# apply_pipelines_to_folder.py

import os
import cv2
import numpy as np
import pandas as pd
from read_dicom import read_dicom_file
from image_quality_metrics import analyze_image_quality
from file_selection import select_folder
from generate_all_pipelines import get_all_pipelines

output_csv_path = "all_pipeline_metrics.csv"
results = []

def is_dicom_file(path):
    return path.lower().endswith('.dcm')

def add_label_above_image(img, label, height=30):
    label_img = np.ones((height, img.shape[1]), dtype=np.uint8) * 255
    cv2.putText(label_img, label, (5, int(height * 0.75)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
    return np.vstack([label_img, img])

folder_path = select_folder()
pipelines = get_all_pipelines()

for fname in os.listdir(folder_path):
    file_path = os.path.join(folder_path, fname)
    
    if not is_dicom_file(file_path):
        continue

    image, ds, status = read_dicom_file(file_path)
    if ds is None:
        print(f" Failed to read {fname}: {status}")
        continue

    base_metrics = analyze_image_quality(image)
    all_versions = [("Original", image.copy())]

    for pipeline_name, pipeline_fn in pipelines.items():
        try:
            processed = pipeline_fn(image.copy(), base_metrics)
            metrics = analyze_image_quality(processed)

            row = {
                "Filename": fname,
                "Pipeline": pipeline_name,
                **metrics
            }
            results.append(row)

            all_versions.append((pipeline_name, processed))

        except Exception as e:
            print(f" Error in {pipeline_name} for {fname}: {e}")

    # Save side-by-side comparison image
    if all_versions:
        height = 256
        combined_imgs = []
        for name, img in all_versions:
            resized = cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))
            labeled = add_label_above_image(resized, name)
            combined_imgs.append(labeled)

        comparison_img = np.hstack(combined_imgs)
        os.makedirs("comparisons", exist_ok=True)
        out_path = os.path.join("comparisons", f"{os.path.splitext(fname)[0]}_comparison.png")
        cv2.imwrite(out_path, comparison_img)

# Save all metrics to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv_path, index=False)
print(f"\n Combined CSV saved to: {output_csv_path}")
print(" All comparison images saved in: comparisons/")
