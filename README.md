# AccNet24

The Matlab code for traning, validating, and testing the AccNet-24 is in [AccNet24.m](AccNet24.m). This file receives the signal images created with  Gramian Angular Field (GAF). Crearing Gramian Angular Field (GAF) images was done with pyts (A Python Package for Time Series Classification). The Jypiter notebook [Create_GAF_images.ipynb](Create_GAF_images.ipynb)contains the code for creating GAF images.

## Read the data 
- Update address for rootFolder containing training, validation, and test data (GAF images).  i.e., there is a seperate code for converting signal to GAF images.
- The folders should contain 4 sub-folders each contaning activity classes (Sleep, Sedentary, LPA, MVPA).  
- The exact category names (i.e., subfolders) should be provided as categories.
- It should be noted that we created three images for each 30-second window. You will also need the matlab code for majority voting, if you want to vote between the predictions for in x, y, and z images. 

## Pretrained models
- It is possible to import the pretained models, and test the data data on new images. 
- The size of the pretrained model is large. Please send an email to Vahid.Farrahi@oulu.fi, if you need the pretrained models. 

## Creating Gramian Angular Field (GAF) images.
- For creating GAF images, you need to download the raw acceleration the data. 
- The open access dataset conatining raw acceleration data can be accessed and download from here [Capture-24 dataset](https://github.com/OxWearables/capture24)

