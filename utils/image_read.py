import matplotlib.image as mpimg
import os
from skimage.transform import resize
import numpy as np
import pandas as pd


def image_read(l):
    folder = l
    y = []
    images = []
    desired_size = 224
    for filename in os.listdir(folder):
        f = str(filename)
        for i in range(len(f)):
            if f[i] == 't':
                c = f[i+1] + f[i+2]
                c = int(c)
                y.append(c)
                break
        img = mpimg.imread(os.path.join(folder , filename))
        if img is not None: 
            img = resize(img , (desired_size , desired_size))
            images.append(img)
    x = np.asarray(images , dtype = np.float32)
    x = np.reshape(x , (x.shape[0] , x.shape[1] , x.shape[2] , 1))
    x = np.concatenate((x , x , x) , axis = 3)
    y = np.asarray(y)
    return x , y




