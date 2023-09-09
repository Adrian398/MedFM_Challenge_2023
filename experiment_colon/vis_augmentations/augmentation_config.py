import torchvision.transforms as transforms


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
        size=(384, 384),
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