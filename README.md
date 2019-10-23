# Single Image Super Resolution by Parallel Block Network(Keras Implementation)

## Overview

Different sized kernels generate different features. Multipath cascade of different sized filters can improve the performance of single image super rresolution. This process enables the fusion of local & global features at the same time. To reduce the parameter numbers, factored convolution technique is applied. The whole process is implemented with Keras on top of TensorFlow.

Debjoy Chowdhury, and  Dimitrios Androutsos. "Single Image Super-Resolution via Cascaded Parallel Multisize Receptive Field." Proceedings of the 2019 IEEE International Conference on Image Processing (ICIP).

* [The author's project page](https://ieeexplore.ieee.org/document/8803300)

## Files

* Train.py: Runs the script to start training.
* Test.py: Predicts the images in a folder and saves in the given directory.
* sr_utilities.py: The utility functions are kept in this script. 
* PSNR_SSIM.m: Calculates the average PSNR & SSIM in a folder of images.

## Data

Matlab files are used as tarining and validation data. 
* aug_train_data.m: Flips, rotates the training data according to the desired patch
* aug_test_data.m: Generates the validation patch of given size

The training and validation mat files can be found [here](https://drive.google.com/file/d/123sfKk2gTbTDY9j4kfUcZr0EQrvqJ0pm/view?usp=sharing).
In addition, all the benchmark training and testing images are given [here](https://drive.google.com/file/d/1ug-B6FPuWfFKLays91rGUBuyrc9tHJD6/view?usp=sharing).  

## Implementation

Just simply run the training script by mentioning the scaling factors, and input training/validation data. For testing, keep in mind the directory of the saved model. This technique supports multiscale training as pre-upscaling method is used here. So, a single model can handle multiple upscalings. 
