# OCRT For Medical Devices

We present a demo of implementation of Resnet-20 Computer Vision algorithm for OCRT of readings of Blood Glucose Levels from users medical devices.

Demonstration Jupyter notebook: `OCRT_Demonstration_Notebook_Aviva.ipynb`

Our system is capable of

- Reading off numbers 20-400 mg/dL
- Agnostic to distance to phone camera
- Agnostic of light conditions
- Agnostic of environment (inside/outside)

Implementation details (`train_resnetv1_20_400_BG_range`):

- Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
- Last ReLU is after the shortcut connection.
- At the beginning of each stage, the feature map size is halved (downsampled)
by a convolutional layer with strides=2, while the number of filters is
doubled. Within each stage, the layers have the same number filters and the
same number of filters.

Features maps sizes:
stage 0: 32x32, 16
stage 1: 16x16, 32
stage 2:  8x8,  64

ResNet20 ~ 0.27M

Accuracy o  unseen data: ~95%
Inference time: ~50ms
Memory: ~35MB for the model with loaded weights, ready to be deployed on a back-end server.
