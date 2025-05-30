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


def ml_preprocess_image(image, metrics):
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
    
    

    # Construct feature vector
    input_features = np.array([[
        mapped_metrics['brightness'],
        mapped_metrics['contrast'],
       # mapped_metrics['Michelson_Contrast'],
        mapped_metrics['sharpness'],
       # mapped_metrics['Sharpness_Tenengrad'],
      #  mapped_metrics['Noise_STD_CenterPatch'],
        mapped_metrics['noise']
       # mapped_metrics['Noise_STD_Estimated']
    ]])
    # Convert to DataFrame to match training input format
    # input_df = pd.DataFrame([mapped_metrics])  # DataFrame with named columns
    input_df = pd.DataFrame([{
    'brightness': mapped_metrics['brightness'],
    'contrast': mapped_metrics['contrast'],
    'sharpness': mapped_metrics['sharpness'],
    'noise': mapped_metrics['noise']
    }])
    # debugging
    print("[ML DEBUG] Final input_df sent to model:")
    print(input_df)

    # Predict quality class
    # predicted_class = clf.predict(input_df)[0]
    #itr 2
    # predicted_index = clf.predict(input_df)[0]
    # predicted_class = le.inverse_transform([predicted_index])[0]
    # confidence scores
    probs = clf.predict_proba(input_df)[0]
    predicted_index = np.argmax(probs)
    predicted_class = le.inverse_transform([predicted_index])[0]
    confidence = probs[predicted_index]
    
    print(f"[ML DEBUG] Predicted class: {predicted_class} (confidence: {confidence:.2%})")

    
    # predicted_class = 'low_contrast'
    # for debuging ------
    # print(f"[ML DEBUG] Predicted class: {predicted_class}")
    # ---------
    # Apply appropriate enhancement
    if predicted_class == 'blurry':
        print("[ML DEBUG] Applying sharpening for blurry image.")
        return enhance_blurry(image)
    elif predicted_class == 'noisy':
        print("[ML DEBUG] Applying denoising for noisy image.")
        return enhance_noisy(image)
    elif predicted_class == 'low_contrast':
        print("[ML DEBUG] Applying histogram equalization for low contrast.")
        return enhance_low_contrast(image)
    else:
        print("[ML DEBUG] Image predicted as good -> no enhancement applied.")
        return enhance_good(image)








# # def ml_preprocess_image(image, metrics):
# #     # Map from analyze_image_quality() keys to ML model's expected keys
# #     mapped_metrics = {
# #         'brightness': metrics['Brightness (mean)'],
# #         'contrast': metrics['Contrast (std dev)'],
# #         'Michelson_Contrast': metrics['Michelson Contrast'],
# #         'sharpness': metrics['Sharpness (Laplacian)'],
# #         'Sharpness_Tenengrad': metrics['Sharpness (Tenengrad)'],
# #         'Noise_STD_CenterPatch': metrics['Noise (flat region std)'],
# #         'noise': metrics['Noise (wavelet std)'],
# #         'Noise_STD_Estimated': metrics['Estimate_noise_std(lapplacian variance)']
# #     }

# #     # Prepare input features
# #     input_features = np.array([[ 
# #         mapped_metrics['brightness'],
# #         mapped_metrics['contrast'],
# #         mapped_metrics['Michelson_Contrast'],
# #         mapped_metrics['sharpness'],
# #         mapped_metrics['Sharpness_Tenengrad'],
# #         mapped_metrics['Noise_STD_CenterPatch'],
# #         mapped_metrics['noise'],
# #         mapped_metrics['Noise_STD_Estimated']
# #     ]])

# #     # Predict class
# #     predicted_class = clf.predict(input_features)[0]
# #     print(f"Predicted Quality Class: {predicted_class}")

# #     # Apply enhancement based on predicted class
# #     if predicted_class == 'blurry':
# #         return enhance_blurry(image)
# #     elif predicted_class == 'noisy':
# #         return enhance_noisy(image)
# #     elif predicted_class == 'low_contrast':
# #         return enhance_low_contrast(image)
# #     else:
# #         return enhance_good(image)








# # #--- Main ML-Based Enhancement Function ---
# # def ml_preprocess_image(image,metrics):
# #     """
# #     Predict image quality class and apply corresponding enhancement.
# #     """
# #     # Ensure grayscale input
# #     if len(image.shape) == 3:
# #         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# #     metrics = analyze_image_quality(image)

# #     feature_vector = np.array([[
# #         metrics['brightness'],
# #         metrics['contrast'],
# #         metrics['Michelson_Contrast'],
# #         metrics['sharpness'],
# #         metrics['Sharpness_Tenengrad'],
# #         metrics['Noise_STD_CenterPatch'],
# #         metrics['noise'],
# #         metrics['Noise_STD_Estimated']
# #     ]])

# #     predicted_class = le.inverse_transform(clf.predict(feature_vector))[0]

# #     if predicted_class == 'blurry':
# #         return enhance_blurry(image)
# #     elif predicted_class == 'noisy':
# #         return enhance_noisy(image)
# #     elif predicted_class == 'low_contrast':
# #         return enhance_low_contrast(image)
# #     else:
# #         return enhance_good(image)


####################################
# Too strong

# import cv2
# import numpy as np
# import joblib
# import pandas as pd

# # Stronger Sharpening using Unsharp Mask
# def enhance_blurry(img):
#     img = to_uint8_if_needed(img)
#     blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)
#     sharpened = cv2.addWeighted(img, 1.8, blurred, -0.8, 0)  # stronger sharpening
#     return sharpened

# # Stronger Denoising using Fast NLM
# def enhance_noisy(img):
#     img = to_uint8_if_needed(img)
#     return cv2.fastNlMeansDenoising(img, None, h=15, templateWindowSize=7, searchWindowSize=21)

# # Stronger Contrast using CLAHE
# def enhance_low_contrast(img):
#     img = to_uint8_if_needed(img)
#     clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
#     return clahe.apply(img)

# def enhance_good(img):
#     return to_uint8_if_needed(img)

# def to_uint8_if_needed(img):
#     """Convert image to uint8 if it's not already."""
#     if img.dtype != np.uint8:
#         return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     return img

# # --- Load classifier & label encoder ---
# clf = joblib.load("rf_quality_classifier.pkl")
# le = joblib.load("label_encoder.pkl")

# def ml_preprocess_image(image, metrics):
#     try:
#         mapped_metrics = {
#             'brightness': metrics['Brightness (mean)'],
#             'contrast': metrics['Contrast (std dev)'],
#             'sharpness': metrics['Sharpness (Laplacian)'],
#             'noise': metrics['Noise (wavelet std)']
#         }
#     except KeyError as e:
#         raise ValueError(f"Missing required metric key: {e}")

#     input_df = pd.DataFrame([mapped_metrics])

#     print("[ML DEBUG] Final input_df sent to model:")
#     print(input_df)

#     # Predict class with probabilities
#     probs = clf.predict_proba(input_df)[0]
#     predicted_index = np.argmax(probs)
#     predicted_class = le.inverse_transform([predicted_index])[0]
#     confidence = probs[predicted_index]

#     print(f"[ML DEBUG] Predicted class: {predicted_class} (confidence: {confidence:.2%})")

#     # --- Apply appropriate enhancement ---
#     if predicted_class == 'blurry':
#         print("[ML DEBUG] Applying Unsharp Mask (strong) for blurry image.")
#         return enhance_blurry(image)
#     elif predicted_class == 'noisy':
#         print("[ML DEBUG] Applying Fast NLM denoising for noisy image.")
#         return enhance_noisy(image)
#     elif predicted_class == 'low_contrast':
#         print("[ML DEBUG] Applying CLAHE (strong) for low contrast.")
#         return enhance_low_contrast(image)
#     else:
#         print("[ML DEBUG] Image predicted as good → no enhancement.")
#         return enhance_good(image)
