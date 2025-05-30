import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from image_quality_metrics import analyze_image_quality


def compute_psnr(img1, img2):
    return cv2.PSNR(img1, img2)


def compute_ssim(img1, img2):
    ssim_val, _ = ssim(img1, img2, full=True)
    return ssim_val


def compute_edge_pixels(img):
    edges = cv2.Canny(img, 100, 200)
    return np.count_nonzero(edges)


def evaluate_quality(original, processed):
    psnr_value = compute_psnr(original, processed)
    ssim_value = compute_ssim(original, processed)
    edge_count = compute_edge_pixels(processed)
    metrics = analyze_image_quality(processed)

    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'Edge Pixels': edge_count,
        'Sharpness (Laplacian)': metrics['Sharpness (Laplacian)'],
        'Noise (wavelet std)': metrics['Noise (wavelet std)'],
        'Contrast (std dev)': metrics['Contrast (std dev)'],
        'Brightness (mean)': metrics['Brightness (mean)']
    }


def compare_images(images, titles):
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def evaluate_and_display(original, static_img, adaptive_img, ml_img):
    print("\n--- Quantitative Evaluation ---")

    versions = ["Original", "Static", "Adaptive", "ML-Based"]
    images = [original, static_img, adaptive_img, ml_img]

    results = []
    for title, img in zip(versions, images):
        print(f"\n{title} Image:")
        res = evaluate_quality(original, img)
        for k, v in res.items():
            print(f"{k}: {v:.4f}")
        results.append(res)

    print("\n--- Visual Comparison ---")
    compare_images(images, titles=versions)

    return results
