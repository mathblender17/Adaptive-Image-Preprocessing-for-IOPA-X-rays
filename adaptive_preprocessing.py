
import cv2
import numpy as np

def adaptive_preprocessing(img, metrics):
    # Unpack metrics (adjust keys as needed to match your metric dict)
    contrast_std = metrics.get('Contrast (std dev)', metrics.get('contrast_std'))
    sharpness_laplacian = metrics.get('Sharpness (Laplacian)', metrics.get('sharpness_laplacian'))
    noise_estimate = metrics.get('Noise (flat region std)', metrics.get('noise_estimate'))

    # 1. Contrast Enhancement with CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    if contrast_std is not None:
        if contrast_std < 40:
            clahe.setClipLimit(3.0)  # stronger enhancement
        elif contrast_std < 60:
            clahe.setClipLimit(2.0)  # moderate
        else:
            clahe.setClipLimit(1.0)  # mild or skip

    # Convert to 8-bit if needed
    if img.dtype != np.uint8:
        img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_8bit = img.copy()

    contrast_enhanced = clahe.apply(img_8bit)

    # 2. Sharpening with Unsharp Mask
    if sharpness_laplacian is not None:
        if sharpness_laplacian < 150:
            amount = 1.5  # strong
        elif sharpness_laplacian < 250:
            amount = 1.0  # moderate
        else:
            amount = 0.5  # light
    else:
        amount = 1.0  # default sharpening

    gaussian = cv2.GaussianBlur(contrast_enhanced, (5, 5), 1.0)
    sharpened = cv2.addWeighted(contrast_enhanced, 1 + amount, gaussian, -amount, 0)

    # 3. Denoising based on noise estimate
    noise_threshold = 0.02  # tweak this threshold as needed
    if noise_estimate is not None:
        noise_capped = min(noise_estimate, noise_threshold)
    else:
        noise_capped = 0.0

    if noise_capped > noise_threshold:
        denoised = cv2.fastNlMeansDenoising(sharpened, None, h=10, templateWindowSize=7, searchWindowSize=21)
    else:
        denoised = sharpened

    return denoised


# Too strong
# def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, strength=1.0):
#     blurred = cv2.GaussianBlur(image, kernel_size, sigma)
#     mask = cv2.subtract(image, blurred)
#     sharpened = cv2.addWeighted(image, 1.0, mask, strength, 0)
#     return sharpened

# def adaptive_preprocessing(img, metrics):
#     contrast_std = metrics.get('Contrast (std dev)', 0)
#     sharpness_laplacian = metrics.get('Sharpness (Laplacian)', 0)
#     noise_estimate = metrics.get('Noise (flat region std)', 0)

#     # Convert to 8-bit if needed
#     if img.dtype != np.uint8:
#         img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     else:
#         img_8bit = img.copy()

#     # --- 1. Stronger CLAHE Contrast ---
#     clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))  # stronger default
#     if contrast_std < 40:
#         clahe.setClipLimit(6.0)  # stronger
#     elif contrast_std < 60:
#         clahe.setClipLimit(4.0)  # moderate
#     else:
#         clahe.setClipLimit(2.0)  # light

#     contrast_enhanced = clahe.apply(img_8bit)

#     # --- 2. Stronger Unsharp Mask ---
#     if sharpness_laplacian < 150:
#         amount = 2.0  # very strong
#     elif sharpness_laplacian < 250:
#         amount = 1.5  # strong
#     else:
#         amount = 1.0  # moderate

#     sharpened = unsharp_mask(contrast_enhanced, strength=amount)

#     # --- 3. Denoising ---
#     if noise_estimate > 20:  # if high noise
#         denoised = cv2.fastNlMeansDenoising(sharpened, None, h=15, templateWindowSize=7, searchWindowSize=21)
#     elif noise_estimate > 10:
#         denoised = cv2.fastNlMeansDenoising(sharpened, None, h=10)
#     else:
#         denoised = sharpened

#     return denoised