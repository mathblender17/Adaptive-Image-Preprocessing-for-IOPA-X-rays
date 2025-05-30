import matplotlib.pyplot as plt

def visualize_dicom_image(image, title="Raw DICOM Image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
