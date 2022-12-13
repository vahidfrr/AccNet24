# AccNet24

The Matlab code for traning, validating, and testing the AccNet-24 is in [AccNet24.m](AccNet24.m). This file receives the signal images created with  Gramian Angular Field (GAF). Crearing Gramian Angular Field (GAF) images was done with pyts (A Python Package for Time Series Classification). The python code for 

## read the data - Update address for rootFolder containing training, validation, and test data (GAF images).  i.e., there is a seperate code for converting signal to GAF images.
-The folders should contain 4 sub-folders each contaning activity classes (Sleep, SED, LPA, MVPA).  
-The exact category names (i.e., subfolders) should be provided as categories.  


## pretrained models
It is possible to simply import the pretained models, and test the data data on new images. 
The size of the pretrained model is large. Please send an email to Vahid.Farrahi@oulu.fi. 

## creating Gramian Angular Field (GAF) images.
For creating GAG images, you need to download the raw acceleration the data. 
The dataset conatining raw acceleration data can be accessed from here [Capture-24 dataset](https://github.com/OxWearables/capture24)
