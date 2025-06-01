

import cv2
import numpy as np
import joblib
# from image_quality_metrics import analyze_image_quality
import pandas as pd

# --- Enhancement Functions ---
def enhance_blurry(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def enhance_noisy(img):
    return cv2.medianBlur(img, 3)

def enhance_low_contrast(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(img)

def enhance_good(img):
    return img

# --- Load model & encoder once ---
clf = joblib.load("rf_quality_classifier.pkl")
le = joblib.load("label_encoder.pkl")

# with return class
def ml_preprocess_ops_image(image, metrics, return_class=False):
    # Map keys to expected feature names
    try:
        mapped_metrics = {
            'brightness': metrics['Brightness (mean)'],
            'contrast': metrics['Contrast (std dev)'],
            #'Michelson_Contrast': metrics['Michelson Contrast'],
            'sharpness': metrics['Sharpness (Laplacian)'],
            #'Sharpness_Tenengrad': metrics['Sharpness (Tenengrad)'],
            #'Noise_STD_CenterPatch': metrics['Noise (flat region std)'],
            'noise': metrics['Noise (wavelet std)']
            #'Noise_STD_Estimated': metrics['Estimate_noise_std(lapplacian variance)']
        }
    except KeyError as e:
        raise ValueError(f"Missing required metric key: {e}")
    
    


    input_df = pd.DataFrame([{
    'brightness': mapped_metrics['brightness'],
    'contrast': mapped_metrics['contrast'],
    'sharpness': mapped_metrics['sharpness'],
    'noise': mapped_metrics['noise']
    }])
    
    
    
    # debugging
    print("[ML DEBUG] Final input_df sent to model:")
    print(input_df)

   
    probs = clf.predict_proba(input_df)[0]
    predicted_index = np.argmax(probs)
    predicted_class = le.inverse_transform([predicted_index])[0]
    confidence = probs[predicted_index]
    
    print(f"[ML DEBUG] Predicted class: {predicted_class} (confidence: {confidence:.2%})")

   
    if predicted_class == 'blurry':
        print("[ML DEBUG] Applying sharpening for blurry image.")
        enhanced = enhance_blurry(image)
    elif predicted_class == 'noisy':
        print("[ML DEBUG] Applying denoising for noisy image.")
        enhanced = enhance_noisy(image)
    elif predicted_class == 'low_contrast':
        print("[ML DEBUG] Applying histogram equalization for low contrast.")
        enhanced = enhance_low_contrast(image)
    else:
        print("[ML DEBUG] Image predicted as good -> no enhancement applied.")
        enhanced = enhance_good(image)
    if return_class:
        return enhanced, predicted_class
    else:
        return enhanced


