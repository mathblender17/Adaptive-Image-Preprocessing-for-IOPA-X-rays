import tkinter as tk
from tkinter import filedialog

def select_dicom_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select DICOM File",
        filetypes=[("DICOM files", "*.dcm")]
    )
    return file_path
