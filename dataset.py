import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


def _convert_to_segmentation_mask(mask):
    height, width = mask.shape[:2]
    segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
    for label_index, label in enumerate(VOC_COLORMAP):
        segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)/255
    return segmentation_mask


class VOC2012Dataset(Dataset):

    def __init__(self, image_dir, mask_dir, segment_txt, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        with open(segment_txt, "r") as file:
            lines = file.readlines()
        self.images = lines

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx].replace("\n", "") + ".jpg")
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace("\n", "") + ".png")
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert('RGB'))
        mask = _convert_to_segmentation_mask(mask)
        if self.transform is not None:
            augumentations = self.transform(image=image, mask=mask)
            image_aug = augumentations['image']
            mask_aug = augumentations['mask']
        return image_aug, mask_aug


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_paths = os.path.join(self.image_dir, self.images[item])
        mask_paths = os.path.join(self.mask_dir, self.images[item].replace(".jpg", "_mask.gif"))

        images = np.array(Image.open(image_paths).convert("RGB"))
        masks = np.array(Image.open(mask_paths).convert("L"), dtype=np.float32)
        masks[masks == 255] = 1.0

        if self.transform is not None:
            augumentations = self.transform(image=images, mask=masks)
            image_aug = augumentations['image']
            mask_aug = augumentations['mask']
        return image_aug, mask_aug


if __name__ == '__main__':
    train_transform = A.Compose(
        [
            A.Resize(height=240, width=160),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    train_dataset = datasets.VOCSegmentation(root=".", year='2012', image_set='train', download=True, transform=train_transform)
    train_ds = CarvanaDataset(image_dir="/home/j/Dataset/Carvana/train", mask_dir="/home/j/Dataset/Carvana/train_masks", transform=train_transform)
    train_loader = DataLoader(dataset=train_ds, batch_size=16, shuffle=True)
    print(next(iter(train_loader)))