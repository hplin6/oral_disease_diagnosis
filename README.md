# Automatic oral diseases in smartphone-based images

## Introduction
This is the source code of the paper “Automatic detection of oral cancer in smartphone-based images using deep learning for early diagnosis” (under review).


## Network

- The multi-class network is using the modified HRNet-W18, which achieves similar classification performance compared with the [official code](https://github.com/HRNet/HRNet-Image-Classification) while reducing the computation complexity and the number of parameters. 
- The code is a slightly modified version of the official code, by removing the BN&FC layers and changing the number of output channels in the representation head.

### Training
- The training is only slight differences (with the same method but different super-parameters) on the original implementation.
- The official pre-trained weights are also used.

### Testing
- The training is only slight differences (with the same method but different super-parameters) on the original implementation.
- The results were reported using PyCM and the Scikit-learn library 

For more details using HRNet, please go to the [official website](https://github.com/HRNet/HRNet-Image-Classification).

## Software

### requirement
- PyQT5
- Pytorch1.0
- Numpy
- Cv2
- CUDA 10.1

**The source code of the software and trained model will be released soon**
