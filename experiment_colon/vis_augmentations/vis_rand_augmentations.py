import math
import random

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from augmentation_config import *


def generate_augmented_images(original_image, N):
    """
    Generate N augmented images from the original image.

    Args:
        original_image (PIL.Image): The original image.
        N (int): Number of augmented images to generate.

    Returns:
        list: A list containing the original and N augmented images.
    """
    all_images = [original_image]
    all_titles = ["Original"]

    for _ in range(N):
        # Randomly select a subset of augmentations
        num_transforms = random.randint(1, len(augmentations) - 1)  # Exclude the Original
        selected_transforms = random.sample(list(augmentations.values())[1:], num_transforms)

        # Apply the selected augmentations
        transform = transforms.Compose(selected_transforms)
        augmented_image = transform(original_image)
        all_images.append(augmented_image)

        # Store the names of the applied augmentations for the title
        applied_transforms = [name for name, transform in augmentations.items() if
                              transform in selected_transforms]
        all_titles.append(", ".join(applied_transforms))

    return all_images, all_titles


image_path = "2018_75775_1-1_2019-02-21 00_58_00-lv1-27957-36850-2866-3925p0001.png"
image = Image.open(image_path)
N = 8
n_cols=3
augmented_images, titles = generate_augmented_images(image, N)

# Plot the original and augmented images
fig, axs = plt.subplots(math.ceil((N + 1) / n_cols), n_cols, figsize=(20, 15))
axs = axs.ravel()

for i, (img, title) in enumerate(zip(augmented_images, titles)):
    axs[i].imshow(np.array(normalization(img)))
    axs[i].axis('off')
    axs[i].set_title(title)

# Hide any remaining empty subplots
for i in range(N + 1, len(axs)):
    axs[i].axis('off')

plt.tight_layout()
plt.show()
