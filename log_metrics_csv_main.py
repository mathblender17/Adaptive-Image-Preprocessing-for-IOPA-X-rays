import pandas as pd
import hashlib
import os

def log_metrics_to_csv(metrics_dict, stage,filename, csv_path="image_metrics_log.csv"):
    # Add stage (Original, Static, Adaptive, ML)
    metrics = metrics_dict.copy()
    metrics["Stage"] = stage
    metrics["Filename"] = filename
    
    # Create hash to detect duplicates
    # unique_str = "_".join([f"{k}:{v:.4f}" for k, v in metrics.items() if k != "Stage"])
    unique_str = "_".join([
    f"{k}:{v:.4f}" if isinstance(v, (int, float)) else f"{k}:{v}"
    for k, v in metrics.items()
    if k not in ("Stage", "Filename")
    ])

    
    row_hash = hashlib.md5(unique_str.encode()).hexdigest()
    metrics["Hash"] = row_hash

    # Load or create DataFrame
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if row_hash in df["Hash"].values:
            print(f"[LOG] Metrics for file '{filename}', stage '{stage}' already logged. Skipping duplicate.")
            return
    else:
        df = pd.DataFrame()

    # Append and save
    df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"[LOG] Metrics for file '{filename}', stage '{stage}' saved.")
