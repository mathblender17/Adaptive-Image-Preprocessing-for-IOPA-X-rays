import pandas as pd
import joblib

def main():
    # Paths - update as needed
    model_path = "rf_quality_classifier.pkl"
    label_encoder_path = "label_encoder.pkl"
    metrics_csv_path = "NEW_image_quality_metrics.csv"  # Your CSV path
    output_csv_path = "dataset_predictions.csv"
    
    # Load model and label encoder
    clf = joblib.load(model_path)
    le = joblib.load(label_encoder_path)
    
    # Load dataset
    data = pd.read_csv(metrics_csv_path)
    print("Loaded dataset with columns:", list(data.columns))
    
    # Map dataset columns to model input features
    # Adjust these mappings if your model expects different noise or sharpness features
    mapped_features = pd.DataFrame({
        'brightness': data['Brightness'],
        'contrast': data['Contrast_STD'],
        'sharpness': data['Sharpness_Laplacian'],  # or 'Sharpness_Tenengrad'
        'noise': data['Noise_STD_Wavelet']         # or 'Noise_STD_CenterPatch'
    })
    
    # Check for missing data
    if mapped_features.isnull().any().any():
        print("Warning: Some feature values are missing (NaN). They will be handled as-is.")
    
    # Predict
    predicted_indices = clf.predict(mapped_features)
    predicted_labels = le.inverse_transform(predicted_indices)
    
    # Add predictions to original dataframe
    data['Predicted_Label'] = predicted_labels
    
    # Save results
    data.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to '{output_csv_path}'")
    
    # Print summary
    print("\nPrediction summary:")
    print(data['Predicted_Label'].value_counts())

if __name__ == "__main__":
    main()
