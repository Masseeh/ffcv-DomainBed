from torchvision import transforms as T

import torch
from domainbed.datasets.ffcv_transforms import RandomGrayscale, ColorJitter, ResizedCrop
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.transforms import RandomHorizontalFlip, Convert, ToDevice, ToTensor, ToTorchImage, RandomResizedCrop
from ffcv.transforms.common import Squeeze

basic = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
aug = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def ffcv_tf(device=torch.device('cuda'), use_amp=True):

    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]

    aug_image_pipeline = [
        SimpleRGBImageDecoder(),
        RandomResizedCrop(size=224, scale=(0.7, 1.0), ratio=(3 / 4, 4 / 3)),
        RandomHorizontalFlip(),
        ColorJitter(0.3, 0.3, 0.3, 0.3),
        RandomGrayscale(0.1),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16 if use_amp else torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize using image statistics
    ]

    basic_image_pipeline = [
        SimpleRGBImageDecoder(),
        ResizedCrop(224),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        Convert(torch.float16 if use_amp else torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize using image statistics
    ]

    return aug_image_pipeline, basic_image_pipeline, label_pipeline
