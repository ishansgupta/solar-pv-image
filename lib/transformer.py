from skimage.feature import hog
import cv2
import numpy as np
class Transformer():
    def __init__(self, images):
        self.images = images
    
    def hog(self):
        hog_features = []
        for image in self.images:
            h = hog(image, orientations = 9, pixels_per_cell = (30, 30), cells_per_block = (1,1),
                    transform_sqrt = True, block_norm = "L1")
            hog_features.append(h)
        return np.array(hog_features)
    
    def sift(self):
        sift_features = []
        for image in self.images:
            gray_img = self.to_gray(image)
            kp, desc = self.gen_sift_features(gray_img)
            sift_features.append(desc)
        return np.array(sift_features)
    
    def to_gray(self, color_img):
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        return gray
    
    def gen_sift_features(self, gray_img):
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(gray_img, None)
        return kp, desc