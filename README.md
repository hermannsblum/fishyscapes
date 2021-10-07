# Real-Time Sigmoid

Thank you for managing the benchmarking platform.
We are currently developing an road obstacle detection method and are preparing to submit our paper to CVPR2022.
In the following, we describe the environment in which our method was trained and how to test our model. 

# Environments
- pytorch 1.9.0
- torchvision 0.10.0
- numpy 1.20.3
- pillow 8.3.1

# Usage
1. Place our checkpoint file in './experiments/network/resnet50_best_model.pth'.
2. Run your test script for the function we have added.

# Notes
We have added new functions to 'fishyscapes.py' and 'timing.py'.
We cannot measure Cityscapes mIoU of our method because it uses a proprietary label map.
Therefore, we have not added a new function to 'cityscapes_mIoU.py'.