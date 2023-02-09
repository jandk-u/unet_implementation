import torch
from utils import *
from model import *
import albumentations as A
from albumentations.pytorch import ToTensorV2

checkpoint = load_weight('saved_images/28.pth')

val_transform = A.Compose(
    [
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
)


model = UNet(in_channels=3, out_channels=21)
model.load_state_dict(checkpoint['model_state_dict'])

image_test = Image.open('/home/j/Dataset/VOC2012/JPEGImages/2007_000333.jpg')
image_test = np.array(image_test)
image_test = val_transform(image=image_test)['image']
image_test = image_test.unsqueeze(0)
print(image_test.shape)

pred = model(image_test)
print(pred.shape)
convert_pred_to_image(pred)