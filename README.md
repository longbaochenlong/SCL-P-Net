# SCL-P-Net
Discriminative feature constraints via supervised contrastive learning for few-shot forest tree species classification using airborne hyperspectral images
# GFF-B dataset download link
URL: https://pan.baidu.com/s/1MW8Wz3_I7xf_5isV8osS9A<br>
Extraction code: 77ws
# Step 1
Download airborne hyperspectral image and corresponding ground truth from Baidu Netdisk to 'data' directory.
# Step 2
Run GFFB_data_preprocess.py to divide samples into training and test dataset in a ratio of 2.5% and 97.5%, respectively.
# Step 3
Run GFFB_SCL_P_Net_train.py to train the optimal model and test it.
# Step 4
Run GFFB_classification_map.py to generate classification model using the optimal model.
# Deep learning environment
torch 1.13.1
cuda 11.6
torchvision 0.14.1
opencv-python 4.7.0.68
gdal 3.4.3
# Cite
[1]	Chen L, Wu J, Xie Y, et al. Discriminative feature constraints via supervised contrastive learning for few-shot forest tree species classification using airborne hyperspectral images[J]. Remote Sensing of Environment, 2023, 295: 113710.