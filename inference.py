# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:03:52 2021

@author: USER
"""

# Import necessary packages.
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import cv2
import torchvision.models as models
from tqdm.auto import tqdm

from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, Subset
from torchvision.datasets import DatasetFolder

model=torchvision.models.resnext50_32x4d(pretrained=True,progress=True) 

device = "cuda" if torch.cuda.is_available() else "cpu"

for param in model.parameters():
    param.requires_grad = False

#print(model.fc)
num_fc_ftr = model.fc.in_features 
model.fc = nn.Linear(num_fc_ftr, 201) 
model=model.to(device)
#print(model)

model.eval()

# Initialize a list to store the predictions.
predictions = []
    
with open("predict.txt", "w") as f:
    for i in range(len(predictions)):
        predictions[i] = int(predictions[i])
    for i in range(len(predictions)):
        if predictions[i] < 100 and predictions[i] >= 10:
            predictions[i] = "0"+str(predictions[i])
        elif predictions[i] < 10 and predictions[i] > 0:
            predictions[i] = "00"+str(predictions[i])
        f.write(str(predictions[i])+"\n")
        
with open('testing_img_order.txt') as f:
    test_images = [x.strip() for x in f.readlines()]  # all the testing images
classes = dict() 
with open('classes.txt') as f: 
    for line in f.readlines():
        line = line.strip('\n')
        words = line.split('.')
        classes.update({words[0]: words[1]})
test_class = []
with open('predict.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip('\n')
        words = line.split()
        for i in range(len(words)):
            if words[i] in classes:
                test_class.append(words[i]+"."+str(classes.get(words[i])))
print(test_images[1]+" "+test_class[1])
with open('answer.txt', mode='w') as f:
    for i in range(len(test_images)):
        f.write(test_images[i]+" "+test_class[i]+"\n")
f.close()