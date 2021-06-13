import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import ViTForImageClassification
from random import seed, shuffle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# +
with open('class.txt', 'r', encoding = 'utf8') as file:
    classes = file.read().splitlines()
classes.append('isnull')
# DataList = list(os.walk('/work/frank0618/Handwritten-Chinese-Character-Classification/public2'))[0][2]
DataList = list(os.walk('/work/frank0618/train2/'))[0][2]
# AdditionalData = list(os.walk('../pretrain/'))[0][2]
seed(999)
shuffle(DataList)

model = torch.load('tmp.weight')
model.eval()
ToTensor = transforms.ToTensor()
addition_transform = torch.jit.script(torch.nn.Sequential(
        transforms.Pad((60, 0), padding_mode = 'edge'),
#         transforms.RandomRotation(10, fill = 0.9411),
        transforms.CenterCrop((67, 100)),
        transforms.Resize((384, 384)),
#         transforms.GaussianBlur(3, 2.0),
#         transforms.ColorJitter(brightness = (0.75, 1.25), saturation = (0.75, 1.25), contrast = (0.75, 1.25), hue = 0.025),
        transforms.Normalize((0.3812, 0.3593, 0.3634), (0.3999, 0.3854, 0.3828), inplace = True),
    ))
transform_train = transforms.Compose([
        transforms.Pad((60, 30), fill = 240),
#         transforms.Pad((60, 30), padding_mode = 'edge'),
        transforms.RandomRotation(30, fill = 240),
        transforms.RandomPerspective(0.25, fill = 240),
        transforms.CenterCrop((67, 100)),
        transforms.Resize((384, 384)),
        transforms.GaussianBlur(3, (0.1, 2.0)),
        transforms.ColorJitter(brightness = (0.5, 1.5), saturation = (0.5, 1.5), contrast = (0.5, 1.5), hue = 0.1),
#         transforms.ColorJitter(brightness = (0.75, 1.25), saturation = (0.75, 1.25), contrast = (0.75, 1.25), hue = 0.025),
])
def show_dataset(dataset, n = 10):
    img = np.vstack((np.hstack((np.asarray(transform_train(Image.open('/work/frank0618/train2/' + dataset[i]))) for _ in range(n))) for i in range(20)))
    img2 = np.vstack((np.hstack((np.asarray(transform_train(Image.open('/work/frank0618/train2/' + dataset[i]))) for _ in range(n))) for i in range(20, 40)))
    plt.figure(figsize=(100, 100))
    plt.imshow(img * 0.7 + img2 * 0.3)
    plt.axis('off')
torch.backends.cudnn.benchmark = True
show_dataset(DataList[3000:])
plt.show()
# -


