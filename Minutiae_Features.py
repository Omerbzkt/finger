import cv2
from sklearn.preprocessing import normalize
import numpy as np
import os
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
#from pgmagick.api import Image
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import skimage
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

folder_path = "deneme_finger"
# folder_path = "fingerprints"
output_folder = "fingerprint_output"

output_dir = "fingerprint_output"
os.makedirs(output_dir, exist_ok=True)

features_dir = "deneme_fing"
os.makedirs(features_dir, exist_ok=True)

# termination - bifurcation bölgelerinin belirlenmesi
def getTerminationBifurcation(img, mask):
    img = img == 255;
    (rows, cols) = img.shape;
    minutiaeTerm = np.zeros(img.shape);
    minutiaeBif = np.zeros(img.shape);
    
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            if(img[i][j] == 1):
                block = img[i-1:i+2,j-1:j+2];
                block_val = np.sum(block);
                if(block_val == 2):
                    minutiaeTerm[i,j] = 1;
                elif(block_val == 4):
                    minutiaeBif[i,j] = 1;
    
    mask = convex_hull_image(mask>0)
    mask = erosion(mask, square(5))         
    minutiaeTerm = np.uint8(mask)*minutiaeTerm
    return(minutiaeTerm, minutiaeBif)

# bölgelerin özelliklerine ayrılması
class MinutiaeFeature(object):
    def __init__(self, locX, locY, Orientation, Type):
        self.locX = locX;
        self.locY = locY;
        self.Orientation = Orientation;
        self.Type = Type;

# orientation hesaplanması
def computeAngle(block, minutiaeType):
    angle = 0
    (blkRows, blkCols) = np.shape(block);
    CenterX, CenterY = (blkRows-1)/2, (blkCols-1)/2
    if(minutiaeType.lower() == 'termination'):
        sumVal = 0;
        for i in range(blkRows):
            for j in range(blkCols):
                if((i == 0 or i == blkRows-1 or j == 0 or j == blkCols-1) and block[i][j] != 0):
                    angle = -math.degrees(math.atan2(i-CenterY, j-CenterX))
                    sumVal += 1
                    if(sumVal > 1):
                        angle = float('nan');
        return(angle)
    elif(minutiaeType.lower() == 'bifurcation'):
        (blkRows, blkCols) = np.shape(block);
        CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
        angle = []
        sumVal = 0;
        for i in range(blkRows):
            for j in range(blkCols):
                if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                    angle.append(-math.degrees(math.atan2(i - CenterY, j - CenterX)))
                    sumVal += 1
        if(sumVal != 3):
            angle = float('nan')
        return(angle)


def extractMinutiaeFeatures(skel, minutiaeTerm, minutiaeBif):
    FeaturesTerm = []

    minutiaeTerm = skimage.measure.label(minutiaeTerm, connectivity=2);
    RP = skimage.measure.regionprops(minutiaeTerm)
    
    WindowSize = 2          
    FeaturesTerm = []
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        block = skel[row-WindowSize:row+WindowSize+1, col-WindowSize:col+WindowSize+1]
        angle = computeAngle(block, 'Termination')
        FeaturesTerm.append(MinutiaeFeature(row, col, angle, 'Termination'))

    FeaturesBif = []
    minutiaeBif = skimage.measure.label(minutiaeBif, connectivity=2);
    RP = skimage.measure.regionprops(minutiaeBif)
    WindowSize = 1 
    for i in RP:
        (row, col) = np.int16(np.round(i['Centroid']))
        block = skel[row-WindowSize:row+WindowSize+1, col-WindowSize:col+WindowSize+1]
        angle = computeAngle(block, 'Bifurcation')
        FeaturesBif.append(MinutiaeFeature(row, col, angle, 'Bifurcation'))
    return(FeaturesTerm, FeaturesBif)


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
            if w > 250 and h > 250:
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

    return fingerprints, image



def printFeatures(featuresTerm,featuresBifurc,file_name):
    
    locations = []
    types = []
    locations2 = []
    types2 = []
          
    for feature in featuresTerm:

        #print(f"Location: ({feature.locX}, {feature.locY}), Orientation: {feature.Orientation}, Type: {feature.Type}")
        types.append(feature.Type)
        label_encoder = LabelEncoder()
        type_encoded = label_encoder.fit_transform(types)
        locations.append([feature.locX, feature.locY])
        data = np.concatenate((locations, type_encoded[:, np.newaxis]), axis=1)
        output_path = os.path.join(features_dir, f"Termination_{file_name}.npy")
        np.save(output_path,data)
        
    for featur in featuresBifurc:

        #print(f"Location: ({featur.locX}, {featur.locY}), Orientation: {featur.Orientation}, Type: {featur.Type}")
        types2.append(featur.Type)
        label_encoder2 = LabelEncoder()
        types_encoded = label_encoder2.fit_transform(types2)+1
        locations2.append([featur.locX, featur.locY])
        data2 = np.concatenate((locations2,types_encoded[:, np.newaxis]), axis=1)
        output_path = os.path.join(features_dir, f"Bifurcation_{file_name}.npy")
        np.save(output_path,data2)


print("Preprocess aşaması tamamlandı.")



if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(folder_path):
    if filename.endswith(".bmp") or filename.endswith(".jpg"):
        file_name_without_extension = os.path.splitext(filename)[0]
        image_path = os.path.join(folder_path, filename)
        fingerprints, image = process_image(image_path)

        if fingerprints is not None:
            for i, fingerprint in enumerate(fingerprints):
                output_path = os.path.join(output_folder, f"{file_name_without_extension}_{i+1}.jpg")
                cv2.imwrite(output_path, fingerprint)


print("Parmak izleri kaydedildi.")

def detect_ridges(gray, sigma= 0.1):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

def plot_images(*images):
    images = list(images)
    n = len(images)
    fig, ax = plt.subplots(ncols=n, sharey=True, figsize = (12,12))
    for i, img in enumerate(images):
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
    plt.show()

for filename in os.listdir(output_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        file_name_without_extension = os.path.splitext(filename)[0]
        image_path2 = os.path.join(output_dir, filename)
        #print(image_path2)
        img = cv2.imread(image_path2, 0) # 0 imports a grayscale
        if img is None:
            raise(ValueError(f"Image didn\'t load. Check that '{image_path2}' exists."))

        a, b = detect_ridges(img, sigma=0.15)
        THRESHOLD1 = img.mean()
        img = np.array(img > THRESHOLD1).astype(int)
        skel = skimage.morphology.skeletonize(img)
        skel = np.uint8(skel)*255;
        mask = img*255;

        (minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask);
        FeaturesTerm, FeaturesBif = extractMinutiaeFeatures(skel, minutiaeTerm, minutiaeBif)
        BifLabel = skimage.measure.label(minutiaeBif, connectivity=1)
        TermLabel = skimage.measure.label(minutiaeTerm, connectivity=1)
        printFeatures(FeaturesTerm,FeaturesBif,file_name_without_extension)