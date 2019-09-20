Face Recognition and Verification using Yale faces dataset
----------------------------------------------------------

This project deals with both verification and recognition tasks. Before looking at the code , we need to understand the difference between these two operations. 

Face Verification is a one to one operation wherein an input image is compared with another image.  This task performs deals with probelm statement : Given these two images , do they belong to the same person.

On the other hand , Face Recogntion is a one to many operation wherein our input image to program is compared with all these images present in our database for a match. 


Dataset Description:
--------------------

Contains 165 grayscale images in GIF format of 15 individuals. There are 11 images per subject, one per different facial expression or configuration: center-light, w/glasses, happy, left-light, w/no glasses, normal, right-light, sad, sleepy, surprised, and wink.


Steps involved:
---------------

1. The yalefaces.zip file is the dataset used in this code. After extracting the files, you will see that the files may jpg format but the .extension for these files will not be jpg. But this will not raise any error since imread functions in python is capable of reading such formats. 

2. The matlab code is executed on these images for preprocessing the dataset. The preprocessed dataset is stored in the yalefaces_final folder.
Note: Some of the images in the dataset due to illumination settings have shadow cast on them. Preprocessing in this case is not capable of removing these shadows. Hence, they are not used in the final dataset to be passed to our python code as of now. 

3. The python code is executed on this final dataset for face verification task. Random pairs of images are generated and compared for verification task. The face_recog_source and face_recog_test are folder made from images in yalefaces_final dataset for face recognition task. The face_recog_source serves as our database and images in face_recog_test is used as our test input. Each image from this folder is compared with the images in our database for recogntion task


Note:
-----
1. In the python code, due to very linited number of images at our disposal , a pretrained model is used for this purpose. The model weights for vgg_face are downloaded and loaded to the keras model architecturte to perform the above tasks.

2. The model weights are not uploaded on github repository but the link to download these weights are: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view

