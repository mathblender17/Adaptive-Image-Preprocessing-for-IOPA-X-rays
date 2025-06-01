# generate_all_pipelines.py

from static_preprocessing import static_preprocess
from adaptive_preprocessing import adaptive_preprocessing
from ml_adaptive_preprocessing import ml_preprocess_image
from image_quality_metrics import analyze_image_quality
from ml_pipeline_ops import ml_preprocess_ops_image

# Function to apply ML enhancement iteratively
# def iterative_ml(image, metrics, iterations=10):
#     for _ in range(iterations):
#         image = ml_preprocess_image(image, metrics)
#         #metrics = {}  # Skip re-analysis if you want fixed behavior
#     return image
def iterative_ml(image, metrics, iterations=10):
    for i in range(iterations):
        # Call ML enhancement and get prediction & confidence
        # For that, we need ml_preprocess_image to optionally return predicted class

        # Let's modify ml_preprocess_image to optionally return predicted_class too
        image, predicted_class = ml_preprocess_ops_image(image, metrics, return_class=True)

        print(f"[ITER {i+1}] Predicted class: {predicted_class}")

        if predicted_class == 'good':
            print("[ITER] Image predicted good - stopping iterations early.")
            break
        
        # Recalculate metrics for updated image for next iteration
        metrics = analyze_image_quality(image)

    return image




# Define all possible named pipelines
def get_all_pipelines():
    return {
        "Static": lambda img, metrics: static_preprocess(img),
        "Adaptive": lambda img, metrics: adaptive_preprocessing(img, metrics),
        "ML_1x": lambda img, metrics: ml_preprocess_image(img, metrics),
        "ML_10x": lambda img, metrics: iterative_ml(img, metrics, iterations=10),
        "ML10_Static": lambda img, metrics: static_preprocess(iterative_ml(img, metrics, 10)),
        "ML10_Adaptive": lambda img, metrics: adaptive_preprocessing(iterative_ml(img, metrics, 10), metrics),
        "ML10_Adaptive_Static": lambda img, metrics: static_preprocess(adaptive_preprocessing(iterative_ml(img, metrics, 10), metrics)),
        "ML10_Static_Adaptive": lambda img, metrics: adaptive_preprocessing(static_preprocess(iterative_ml(img, metrics, 10)), metrics),
    }
