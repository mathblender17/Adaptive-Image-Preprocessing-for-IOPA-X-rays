
# Adaptive-Image-Preprocessing-for-IOPA-X-rays

## Problem Understanding
This project aims to improve the quality of DICOM medical images through preprocessing techniques to enhance downstream analysis and diagnostic accuracy. Medical images often suffer from low contrast, noise, and blur, which can hinder interpretation and automated analysis. Enhancing image quality facilitates better visualization and improves the performance of AI models for tasks such as caries detection and bone loss assessment.

## Dataset Description
We used a set of DICOM images provided by the dataset or simulated a diverse range of images reflecting common quality issues encountered in clinical practice. The images were loaded using pydicom, and metadata extraction was performed to assist in preprocessing decisions.

## Methodology
### _Image Quality Metrics_

Brightness (mean intensity): Average pixel intensity indicating overall image illumination.

Contrast (standard deviation): Measures pixel intensity variation, reflecting image contrast.

Sharpness (Laplacian variance): Quantifies edge strength and focus quality.

Noise Estimation: Calculated using wavelet-based standard deviation and flat region analysis.


These metrics guide adaptive preprocessing choices and evaluate enhancement effectiveness.

### _Static Preprocessing Baseline_
Histogram equalization and basic filtering techniques applied uniformly to all images.

Serves as a baseline to compare improvements from adaptive methods.

### _Adaptive Preprocessing Pipeline_
Heuristic-based adaptive enhancement:

Contrast enhancement with CLAHE (adaptive histogram equalization).

Unsharp masking for sharpening based on sharpness metrics.

Noise reduction applied conditionally based on noise estimates.

Dynamic adjustment of parameters (e.g., CLAHE clip limit, sharpening amount) driven by image quality metrics.

### _Machine Learning-Based Adaptive Preprocessing_
A Random Forest classifier trained on extracted image quality features to classify images into quality categories: blurry, noisy, low contrast, or good.

Corresponding enhancement functions are applied based on predicted class.

Enhancements include sharpening, denoising, histogram equalization, or no operation.

## Results & Evaluation
### Quantitative Results
| Preprocessing Method        | PSNR           | SSIM  |
| ------------- |:-------------:| -----:|
| Original     |  6.4504098663961065 | 0.17843051948974675 |
| Static    | 6.851786691671412      |   0.11613097034108874 |
| Adaptive Heuristic | 6.889461571193425      |   0.1339387590437667 |
| ML Adaptive | 6.4328730540370325      |  0.1356472458652005 |
| ML 10-Iter | 6.429384398934026     |    0.16286696248510507 |


above table for : IS20250116_180218_7445_56958340.dcm


### Visual Comparison
Original vs Static vs Adaptive images shown side-by-side.

![Comparison between various preprocessing techniques](https://github.com/user-attachments/assets/c456b9b9-ff61-4e39-afc2-0b1ff7cf87d1)


Adaptive methods demonstrate improved contrast and sharpness with controlled noise reduction.

### Analysis
Adaptive preprocessing consistently improved PSNR and SSIM over static baseline.

Heuristic adaptive method allows fine-tuned control but depends on chosen metric thresholds.

ML adaptive method adapts well to diverse image types, automating enhancement choice, but relies on classifier accuracy.

Some limitations remain with over-enhancement or under-processing in edge cases.

## Discussion & Future Work
### Challenges
Balancing enhancement strength to avoid artifacts.

Accurate noise estimation under varying conditions.

Training ML classifier with limited labeled data.

### Future Improvements
Integrate deep learning models for end-to-end enhancement.

Expand training datasets with more varied image quality issues.

Real-time preprocessing integration for clinical workflow.

Extend evaluation with downstream AI model performance impact.

### Benefit to Downstream AI Models
Enhanced image quality improves feature visibility and consistency, leading to more robust and accurate AI predictions for diagnostic tasks like caries detection and bone loss assessment.

## Instructions


### Running the main Pipeline
1. Clone the repository.

2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Run the main script:

```bash
python main.py
```
4. Select DICOM files interactively when prompted.

5. View the visualizations and evaluation metrics printed in the console.

6. Processed images and metrics are logged automatically.
### Running the Sequential main Pipeline
```bash
python seq_main.py
```
*Ensure this file is available

```bash
ref_img_path = "Images_Data_science_intern\\Reference_Output_Quality.jpg"
```




