import math

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

image_path = "2018_75775_1-1_2019-02-21 00_58_00-lv1-27957-36850-2866-3925p0001.png"
image = Image.open(image_path)

# Define normalization transform
normalization = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[123.675/255, 116.28/255, 103.53/255],
        std=[58.395/255, 57.12/255, 57.375/255],
    ),
    transforms.ToPILImage()
])

augmentations = {
    "Original": lambda x: x,
    "Random Resized Crop": transforms.RandomResizedCrop(
        size=(128, 128),
        scale=(0.05, 0.9),
        ratio=(0.75, 1.33),
    ),
    "Enhanced Rotation": transforms.RandomRotation(degrees=(-90, 90)),
    "Random Horizontal Flip": transforms.RandomHorizontalFlip(),
    "Random Vertical Flip": transforms.RandomVerticalFlip(),
    "Enhanced Color Jitter": transforms.ColorJitter(
        brightness=(0.5, 1.5),
        contrast=(0.5, 1.5),
        saturation=(0.5, 1.5),
        hue=(-0.2, 0.2)
    ),
    "Random Affine Transform": transforms.RandomAffine(
        degrees=30,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10
    ),
    "Random Grayscale": transforms.RandomGrayscale(p=0.2),
    "Compose of Multiple Random Transforms": transforms.RandomApply([
        transforms.RandomRotation(degrees=45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.CenterCrop(size=300)
    ], p=0.5)
}

n_cols = 3
n_rows = math.ceil(len(augmentations.items()) / n_cols)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 15))
axs = axs.ravel()

for i, (name, transform) in enumerate(augmentations.items()):
    augmented_image = image
    if name is not "Original":
        augmented_image = normalization(transform(image))
    axs[i].imshow(np.array(augmented_image))
    axs[i].axis('off')
    axs[i].set_title(name)

plt.tight_layout()
plt.show()
