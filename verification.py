import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils.image_read import image_read
from utils.calculate_distance import Distance
from model import model
from utils.generate_random_pairs import random_pairs

X , Y = image_read(r'data/yalefaces_final')

data_groups = [X[np.where(Y == i)] for i in np.unique(Y)]

# Face Verification:
def verification(no_examples = 5):
    face_recog = model()
    source_images, test_images = random_pairs(data_groups, no_examples)
    source_image_embed = face_recog.predict(source_images)
    test_image_embed = face_recog.predict(test_images)
    dist = []
    for i in range(source_image_embed.shape[0]):
        euc_dist = Distance(source_image_embed[i] , test_image_embed[i])
        dist.append(euc_dist)
    for i in range(source_images.shape[0]):
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(source_images[i])
        plt.title('Source Image')
        
        plt.subplot(2,1,2)
        plt.imshow(test_images[i])
        plt.title('\nTest Image')
        
        if dist[i] < 0.55:
            fig.suptitle('Verified - Same Person')
        else:
            fig.suptitle('Verified - Different Person')
        
        plt.show() 

verification() 
