'''
Import the packages needed for classification
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import datetime
import os
'''
Include the functions used for loading, preprocessing, features extraction, 
classification, and performance evaluation
'''
class Loader():
    def __init__(self, **kwargs):
        self.dir_train_images  = '../data/training/' if 'dir_train_images' not in kwargs else kwargs['dir_train_images']
        self.dir_test_images   = '../data/testing/' if 'dir_test_images' not in kwargs else kwargs['dir_test_images']
        self.dir_train_labels  = '../data/labels_training.csv' if 'dir_train_labels' not in kwargs else kwargs['dir_train_labels']

    def load_data(self):
        labels_pd = pd.read_csv(self.dir_train_labels)
        ids       = labels_pd.id.values
        data      = []
        for identifier in ids:
            fname     = self.dir_train_images + identifier.astype(str) + '.tif'
            image     = mpl.image.imread(fname)
            data.append(image)
        labels = labels_pd.label.values
        data = np.array(data) # Convert to Numpy array
        return data, labels


