import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


class ColonDataset(Dataset):
    def __init__(self, root_dir, excel_file, transform=None):
        self.df = pd.read_excel(os.path.join(root_dir, excel_file))
        self.root_dir = os.path.join(root_dir, 'images')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.loc[idx, 'img_id'])
        image = Image.open(img_name)
        label = self.df.loc[idx, 'tumor']

        if self.transform:
            image = self.transform(image)

        return image, label


# Load the pre-trained VGG16 model
base_model = models.vgg16(weights=True)

# Remove the top layer to get the feature extractor
base_model.classifier = base_model.classifier[:-1]

# Define the image transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the image datasets for the training and validation sets
dataset = ColonDataset('../data/MedFMC_train/colon', 'colon_train.xlsx', transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Define the data loaders for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


def extract_features(loader):
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Extracting features"):
            outputs = base_model(inputs)
            features.append(outputs.numpy())
            labels.append(targets.numpy())

    return np.concatenate(features), np.concatenate(labels)


# Use the VGG16 model to extract features from the images
train_features, train_labels = extract_features(train_loader)
val_features, val_labels = extract_features(val_loader)

# Train a logistic regression model on the extracted features
clf = LogisticRegression()
clf.fit(train_features, train_labels)

# Evaluate the model on the validation set
val_accuracy = clf.score(val_features, val_labels)
print(f'Validation accuracy: {val_accuracy * 100:.2f}%')
