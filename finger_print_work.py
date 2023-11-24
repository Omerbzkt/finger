# ANA KOD
import cv2
from sklearn.preprocessing import normalize
import numpy as np
import os
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def process_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Resim yüklenemedi veya okunamadı: {image_path}")
        return None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gürültü azaltma
    norm_low_n = gaussian_filter(gray_image, sigma=1)

    # Kenar tespiti
    aperture_size = 5
    edges = cv2.Canny(norm_low_n, 100, 150, apertureSize=aperture_size)

    # Parmak izlerini bulma ve ayırma
    def find_fingerprints(edges):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        dilated = cv2.erode(dilated, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fingerprint_boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 100:
                fingerprint_boxes.append((x, y, w, h))

        fingerprint_boxes.sort(key=lambda box: box[0])

        largest_fingerprints = fingerprint_boxes[:4]

        fingerprints = []
        for i, fingerprint_box in enumerate(largest_fingerprints):
            x, y, w, h = fingerprint_box
            fingerprint = edges[y:y+h, x:x+w]
            fingerprints.append(fingerprint)

        return fingerprints

    fingerprints = find_fingerprints(edges)

    return fingerprints,image

print("Preprocess aşaması tamamlandı.")

folder_path = "fingerprints"
output_folder = "fingerprint_output"


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        file_name_without_extension = os.path.splitext(filename)[0]
        image_path = os.path.join(folder_path, filename)
        fingerprints, image = process_image(image_path)

        if fingerprints is not None:
            for i, fingerprint in enumerate(fingerprints):
                output_path = os.path.join(output_folder, f"{file_name_without_extension}_{i+1}.jpg")
                cv2.imwrite(output_path, fingerprint)


output_dir = "fingerprint_output"
os.makedirs(output_dir, exist_ok=True)

print("Parmak izleri kaydedildi.")

def convert_imagess_to_npy(input_folder, output_folder):
    image_files = sorted(os.listdir(input_folder)) 
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for file in image_files:
        image_path = os.path.join(input_folder, file)
        image = cv2.imread(image_path)
        
        if image is not None:
            output_file = os.path.splitext(file)[0] + ".npy" 
            output_path = os.path.join(output_folder, output_file)
            np.save(output_path, image)

input_folder = "fingerprint_output"
output_file = "fingernpy"

convert_imagess_to_npy(input_folder, output_file)
print("Fotoğraflar npy formatına çevrildi.")
