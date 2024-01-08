import torch
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T


class ClassificationPresetTrain:
    def __init__(
        self,
        *,
        resize_size,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
    ):
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))

        transforms += [
            T.Resize(size=(resize_size, resize_size), interpolation=interpolation, antialias=True),
            # T.CenterCrop(crop_size),
        ]
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
            transforms.append(T.RandomVerticalFlip(hflip_prob))
            transforms.append(T.ColorJitter(brightness=0.3, contrast=0.3))
            transforms.append(T.RandomErasing(p=0.2))
        transforms.append(T.Normalize(mean=mean, std=std))
        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        resize_size,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
    ):
        transforms = []
        transforms.append(T.PILToTensor())
        transforms.append(T.ConvertImageDtype(torch.float))
        transforms += [
            T.Resize(size=(resize_size, resize_size), interpolation=interpolation, antialias=True),
            # T.CenterCrop(crop_size),
        ]
        transforms.append(T.Normalize(mean=mean, std=std))
        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
    