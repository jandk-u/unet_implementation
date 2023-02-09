import os.path

import torch
import torchvision.utils
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import cv2


from torchvision.transforms import Compose, ToTensor, ToPILImage

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
label_colour = dict(zip(range(len(VOC_COLORMAP)), VOC_COLORMAP))


def display_transform():
    return Compose(
        ToPILImage(),
        ToTensor()
    )


def save_as_images(tensor_pred, folder, image_name):
    tensor_pred = transforms.ToPILImage()(tensor_pred.byte())
    filename = f"{folder}\{image_name}.png"
    tensor_pred.save(filename)


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cpu"):
    model.eval()
    if not os.path.isdir(folder):
        os.makedirs(folder)
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
    model.train()


def check_accuracy(loader, model, device="cpu", log=True):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum())/ ((preds + y).sum() + 1e-8)
    if log:
        print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
        print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    return dice_score/len(loader), num_correct/num_pixels


def save_weight(model, epoch, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_weight(path):
    check_point = torch.load(path)
    return check_point


def convert_pred_to_image(preds):
    _, indices = torch.max(preds, dim=1, keepdim=True)
    result = []
    for ind in indices:
        ind = ind.detach().numpy()
        r = []
        g = []
        b = []
        for w in ind[0]:
            r_t = []
            g_t = []
            b_t = []
            for h in w:
                for idx, value in label_colour.items():
                    if h == idx:
                        r_t.append(value[0])
                        g_t.append(value[1])
                        b_t.append(value[2])
            r.append(r_t)
            g.append(g_t)
            b.append(b_t)
        image = np.dstack((r, g, b))
        image = Image.fromarray(image.astype(np.uint8))
        result.append(image)
    return result

if __name__ == '__main__':
    print(label_colour)