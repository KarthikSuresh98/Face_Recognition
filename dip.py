import matplotlib.image as mpimg
import os
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Model , Sequential
from keras.layers import Input , Convolution2D , MaxPooling2D , BatchNormalization , Flatten , Dense , Dropout , Concatenate , Activation , ZeroPadding2D
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from keras.regularizers import l1 , l2
from sklearn.utils import shuffle
from keras.models import model_from_json
#Function to read images from the folder and convert the images to a desired size and store it in a array  
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

X , Y = image_read(r'data/yalefaces_final')

X_train , X_test , y_train , y_test = train_test_split(X , Y , random_state = 0)

train_groups = [X_train[np.where(y_train == i)] for i in np.unique(y_train)]
test_groups = [X_test[np.where(y_test == i)] for i in np.unique(y_test)]

#Generating ramdom pairs of images for face verification
def random_pairs(groups , size):
    out_img_a , out_img_b , out_score = [] , [] , []
    all_groups = list(range(len(groups)))
    for match_group in [True , False]:
        group_idx = np.random.choice(all_groups , size = size)
        out_img_a += [groups[c_idx][np.random.choice(range(groups[c_idx].shape[0]))] for c_idx in group_idx]
        if match_group:
            b_group_idx = group_idx
            out_score += [1]*size
        else:
            non_group_idx = [np.random.choice([i for i in all_groups if i != c_idx]) for c_idx in group_idx]
            b_group_idx = non_group_idx
            out_score += [0]*size
        out_img_b += [groups[c_idx][np.random.choice(range(groups[c_idx].shape[0]))] for c_idx in b_group_idx]  
    return np.stack(out_img_a , 0) , np.stack(out_img_b , 0) , np.stack(out_score , 0)  

# Convolutional network that accepts inputs of size 224 X 224 and gives a 2622 X 1 dimensional embedding(image represented by a smaller set of numbers) of the input image for verification and recognition applications.
model = Sequential()
model.add(ZeroPadding2D((1,1) , input_shape = (224,224,3)))
model.add(Convolution2D(64 , (3,3) , activation = 'relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))
model.load_weights('vgg_face_weights.h5')

face_recog = Model(inputs = model.layers[0].input , outputs = model.layers[-2].output)


# Function to calculate the eucledian distance between the source and test image
def Distance(a, b):
    euclidean_distance = a - b
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

# Face Verification:
def show_model_output(nb_examples = 5):
    pv_a, pv_b, pv_sim = random_pairs(test_groups, nb_examples)
    #pv_a , pv_b , pv_sim = shuffle(pv_a , pv_b , pv_sim ,  random_state = 0)
    img_embed1 = face_recog.predict(pv_a)
    img_embed2 = face_recog.predict(pv_b)
    dist = []
    for i in range(img_embed1.shape[0]):
        euc_dist = Distance(img_embed1[i] , img_embed2[i])
        dist.append(euc_dist)
    for i in range(pv_a.shape[0]):
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(pv_a[i])
        plt.title('Source Image')
        
        plt.subplot(2,1,2)
        plt.imshow(pv_b[i])
        plt.title('\nTest Image')
        
        if dist[i] < 0.55:
            fig.suptitle('Verified - Same Person')
        else:
            fig.suptitle('Verified - Different Person')
        
        plt.show() 

show_model_output() 

#Face Recognition:

X , Y = image_read(r'data/face_recog_source')
X , Y = shuffle(X , Y , random_state = 0)
s_rep = face_recog.predict(X)
X_t , Y_t = image_read(r'data/face_recog_test')
t_rep = face_recog.predict(X_t)
for i in range(t_rep.shape[0]):
    min_euc = 1.0
    index = -1
    for j in range(s_rep.shape[0]):
        euc_dist = Distance(s_rep[j] , t_rep[i])
        if euc_dist < min_euc:
            index = j
            min_euc = euc_dist

    if index != -1 and min_euc < 0.55:
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(X[index])
        plt.title('Source Image')
        
        plt.subplot(2,1,2)
        plt.imshow(X_t[i])
        plt.title('Test Image')
        
        fig.suptitle('Welcome Employee ' + str(Y[index]))
        
        plt.show() 
    else:
        fig = plt.figure()
        plt.imshow(X_t[i])
        plt.title('Test Image')
        fig.suptitle('Sorry, Entry not Allowed. You are not a employee of this company')

