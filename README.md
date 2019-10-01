Face Recognition and Verification using Yale faces dataset
----------------------------------------------------------

This project deals with both verification and recognition tasks. Before looking at the code , we need to understand the difference between these two operations. 

Face Verification is a one to one operation wherein an input image is compared with another image.  This task performs deals with probelm statement : Given these two images , do they belong to the same person.

On the other hand , Face Recogntion is a one to many operation wherein our input image to program is compared with all these images present in our database for a match. 

Dataset Description:
--------------------

Contains 165 grayscale images in GIF format of 15 individuals. There are 11 images per subject, one per different facial expression or configuration: center-light, w/glasses, happy, left-light, w/no glasses, normal, right-light, sad, sleepy, surprised, and wink.

Transfer Learning:
------------------

Since the dataset used is very small for a deep learning model to perform well hence transfer learning has been used. Transfer learning is a technique is in which we employ a pre-trained model for our problem statement. Since machine learning alogorithms are data specific we generally fine tune a pre-trained model on our own dataset to produce better accuracy. Here the model used is vggface network which has approximately 145 million parameters in its network architecture. Hence due to lack of resources  at my disposal for training such a big model , the pre-trained is directly used for both verification and recognition task without any fine tuning. The model performs pretty well on the yale faces dataset.

Limitation: The model produces a very good accuracy in verification task. But since the pretrained model is used directly , in recognition task sometimes it matches the test image to a wrong source image from our database. 


Utils Folder Content:
---------------------

1. calculate_distance.py - Contains a function to calculate the eucledian distance between the source and test image
2. generate_random_pairs.py - Contains a function to generate ramdom pairs of images for face verification
3. image_read.py - Contains a function to read images from a directory and convert it into a numpy array.


Steps involved:
---------------

1. The yalefaces.zip file is the dataset used in this code. After extracting the files, you will see that the files may jpg format but the .extension for these files will not be jpg. But this will not raise any error since imread functions in python is capable of reading such formats. 

2. The matlab code is executed on these images for preprocessing the dataset. The preprocessed dataset is stored in the yalefaces_final folder.
Note: Some of the images in the dataset due to illumination settings have shadow cast on them. Preprocessing in this case is not capable of removing these shadows. Hence, they are not used in the final dataset to be passed to our python code as of now. 

3. The verification.py is executed on this final dataset for face verification task. The image_read function in the utils folder is used to read the images from a directory and convert it into a numpy array for further processing. The generate_random_pairs function in the utils folder is used to generate random pairs of images. Now the Conv-net model is used to convert these pairs of images into a smaller dimensional representation. The Distance function in the utils folder makes use of these representations to calculate the eucledian distance between the images in a pair and thus ckecks for similarity. 

4. The face_recog_source and face_recog_test are folder made from images in yalefaces_final dataset for face recognition task. The face_recog_source serves as our database and images in face_recog_test is used as our test input. The recognition.py uses the model function from utils folder to compare each image from the face_recog_test folder to the images in our database(face_recog_source) for recogntion task. Comparison is done by calculating the eucledian distance between the image embeddings of the source and test image using the Distance function


Sample Output folder:
---------------------

Contains images of the outputs given by both the verification and recognition tasks. 


Note:
-----
The model weights are not uploaded on github repository but the link to download these weights are: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view

