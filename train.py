import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
from loss import dice_loss
from model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import VOC2012Dataset, CarvanaDataset
from torch.utils.data import DataLoader
from utils import *

# Hyperparameter
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKER = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

IMG_DIR = "/home/j/Dataset/VOC2012/JPEGImages"
IMAG_MASK = "/home/j/Dataset/VOC2012/SegmentationClass"
TRAIN_SEGMENTS_TXT = "/home/j/Dataset/VOC2012/ImageSets/Segmentation/train.txt"
VAL_SEGMENTS_TXT = "/home/j/Dataset/VOC2012/ImageSets/Segmentation/trainval.txt"
LOCATION = "saved_image/multiclass"


def dice_coef(y_true, y_pred, smoodth=1):
    pass


def train(trail_loader, val_loader, model, optimizer, loss_fn, writer):
    bar = tqdm(trail_loader)
    for idx, (data, targets) in enumerate(bar):
        if idx == 20:
            break
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        predictions = model(data)

        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar.set_postfix(loss=loss.item())

        if idx % 10 == 0:
            dice, acc = check_accuracy(loader=val_loader, model=model, log=False)
            writer.add_scalar('Loss/val', dice, idx)
            writer.add_scalar('Acc/val', acc, idx)
            writer.add_scalar('Loss/train', loss, idx)
            save_predictions_as_imgs(val_loader, model)


def train_voc(trail_loader, model, optimizer, loss_fn, writer):
    bar = tqdm(trail_loader)
    num_correct = 0
    num_pixels = 0
    for idx, (data, targets) in enumerate(bar):
        if idx > 10:
            break
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        predictions = model(data)
        predictions = predictions.unsqueeze(1)
        predictions = predictions.permute(0, 1, 3, 4, 2)
        loss = loss_fn(predictions, targets)
        dice_l = dice_loss(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_correct += (predictions == targets).sum()
        num_pixels += torch.numel(predictions)
        acc = num_correct / num_pixels

        bar.set_postfix({'loss':loss.item(), 'dice_loss': dice_l.item()/len(trail_loader), 'acc':acc})

        if idx % 10 == 0:
            writer.add_scalar('Loss', loss.item(), idx)
            writer.add_scalar('dice_loss', dice_l.item(), idx)


def validate(val_loader, model, loss_fn, device='cpu'):
    model.eval()
    num_correct = 0
    num_pixels = 0
    for idx, (data, targets) in enumerate(val_loader):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        predictions = model(data)
        predictions = predictions.unsqueeze(1)
        predictions = predictions.permute(0, 1, 3, 4, 2)
        loss = loss_fn(predictions, targets)
        dice_l = dice_loss(predictions, targets)

        num_correct += (predictions == targets).sum()
        num_pixels += torch.numel(predictions)

        print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
        print(f"Dice score: {dice_l / len(val_loader)}")
        print(f"Loss: ", loss)

    model.train()

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
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

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )

    # Using UNet for Carvanda Dataset
    # train_ds = CarvanaDataset(image_dir="/home/j/Dataset/Carvana/train", mask_dir="/home/j/Dataset/Carvana/train_masks", transform=train_transform)
    # train_loader = DataLoader(dataset=train_ds, batch_size=16, shuffle=True)

    # val_ds = CarvanaDataset(image_dir="/home/j/Dataset/Carvana/val", mask_dir="/home/j/Dataset/Carvana/val_masks", transform=val_transform)
    # val_loader = DataLoader(dataset=val_ds, batch_size=16, shuffle=True)

    # Using Unet for VOC2012
    train_ds = VOC2012Dataset(image_dir=IMG_DIR, mask_dir=IMAG_MASK, segment_txt=VAL_SEGMENTS_TXT,
                              transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    test_ds = VOC2012Dataset(image_dir=IMG_DIR, mask_dir=IMAG_MASK, segment_txt=VAL_SEGMENTS_TXT,
                              transform=val_transform)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    writer = SummaryWriter()
    model = UNet(in_channels=3, out_channels=21).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimize = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUM_EPOCHS):
        print("Training on epoch:", epoch)

        # train with binary segmantic segmentation
        # train(
        #     trail_loader=train_loader,
        #     val_loader=val_loader,
        #     model=model,
        #     optimizer=optimize,
        #     loss_fn=loss_fn,
        #     writer=writer)

        train_voc(trail_loader=train_loader, model=model, optimizer=optimize, loss_fn=loss_fn,
              writer=writer)

        validate(val_loader=test_loader, loss_fn=loss_fn, model=model)
        save_weight(model, epoch, optimize, loss_fn, f'saved_images/{epoch}.pth')


if __name__ == '__main__':
    main()
