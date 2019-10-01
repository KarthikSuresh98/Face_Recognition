import matplotlib.pyplot as plt
from utils.image_read import image_read
from utils.calculate_distance import Distance
from model import model
from sklearn.utils import shuffle


#Face Recognition:
face_recog = model()
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

