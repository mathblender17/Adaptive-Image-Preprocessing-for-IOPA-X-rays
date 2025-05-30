import pydicom
import numpy as np
import cv2

def read_dicom_file(path):
    try:
        ds = pydicom.dcmread(path)

        # Extract pixel data
        img = ds.pixel_array.astype(np.float32)

        # Apply rescale slope/intercept if available
        slope = float(ds.get("RescaleSlope", 1))
        intercept = float(ds.get("RescaleIntercept", 0))
        img = img * slope + intercept

        # Normalize image to 0–255
        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return img_normalized, ds, " DICOM read and processed"

    except Exception as e:
        return None, None, f" Failed to read DICOM: {e}"
def extract_dicom_metadata(ds):
    metadata = {
        "Patient Name": str(ds.get("PatientName", "N/A")),
        "Patient ID": ds.get("PatientID", "N/A"),
        "Modality": ds.get("Modality", "N/A"),
        "Manufacturer": ds.get("Manufacturer", "N/A"),
        "Photometric Interpretation": ds.get("PhotometricInterpretation", "N/A"),
        "Rows": ds.get("Rows", "N/A"),
        "Columns": ds.get("Columns", "N/A"),
        "Pixel Spacing": ds.get("PixelSpacing", "N/A"),
        "Bits Stored": ds.get("BitsStored", "N/A"),
        "Tooth Region": (
            ds[0x0008, 0x2228][0].get("CodeMeaning", "N/A")
            if (0x0008, 0x2228) in ds else "N/A"
        ),
        "Sharpness (if present)": ds.get((0x0009, 0x11B2), "N/A"),
    }
    return metadata
