# VRDL_image-classification
## The link of my Colab

Click [My colab link](https://colab.research.google.com/drive/1b4FmQeQB7rE5cmALDqgpUmIPi2WaioM3?usp=sharing) or just run VRDL_HW1.ipynb

## Dataset Download
```
!gdown --id '1aerVHZJo5GTRU-06PICcCvl39Y-VR3rz' --output 2021VRDL_HW1_datasets.zip

!apt-get install unzi
!unzip -q '2021VRDL_HW1_datasets.zip' -d 2021VRDL_HW1_datasets
```
## Data Augmentation
Before starting training, I apply some data augmentations below on the training set. 
The size of training data is small, so data augmentations may cause the better result.
1. ColorJitter
![color](https://user-images.githubusercontent.com/63098487/158816405-ae30ff8a-1e66-41b5-ad7a-406265509403.jpg)
2. Pad
![pad](https://user-images.githubusercontent.com/63098487/158816480-7be0c3ea-80ce-4e23-9dcc-6ddc658a86ee.jpg)
3. RandomPerspective
![randomperspective](https://user-images.githubusercontent.com/63098487/158816593-a8c3a114-704b-4cd8-adfe-972318161de3.jpg)
4. RandomVerticalFlip
![verti](https://user-images.githubusercontent.com/63098487/158816647-64bdb164-6115-4781-9bb2-b1459ed36590.jpg)

5. RandomHorizontalFlip
![hori](https://user-images.githubusercontent.com/63098487/158816676-e2c83c9c-e0a8-4458-bdf7-4c5b0936b33f.jpg)

6. RandomRotation
![randomrotation](https://user-images.githubusercontent.com/63098487/158816691-59ae39b9-f162-4013-8f18-67c75364f88a.jpg)

## Training
Use pretrained ResNet-50 model.

## Validation
In this case, I use 0.85 * training data as validation data.

## Testing
I use pretrained model, so it's not necessory to download the weights.

## Inference
You can just run inference.py

## Reference
1. Load data: https://www.cnblogs.com/denny402/p/7512516.html
2. Model: https://pytorchtutorial.readthedocs.io/en/latest/tutorial/chapter04_advanced/4_1_finetuning/
3. Training and Testing : https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW03/HW03.ipynb
4. Data augmentations : https://pytorch.org/vision/stable/transforms.html
