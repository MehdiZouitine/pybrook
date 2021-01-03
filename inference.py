from PIL import Image
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from collections import namedtuple
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_erosion


MODELPATH = [
    "/home/mehdi.zouitine/skull_stripping/model/skull_stripper_resnest50d.pth",
    "/home/mehdi.zouitine/skull_stripping/model/skull_stripper_timm-efficientnet-b4.pth",
    "/home/mehdi.zouitine/skull_stripping/model/skull_stripper_resnet18.pth",
    "/home/mehdi.zouitine/skull_stripping/model/skull_stripper_timm-resnest50d_3D.pth",
    "/home/mehdi.zouitine/skull_stripping/model/skull_stripper_timm-efficientnet-b4_3D.pth",
    "/home/mehdi.zouitine/skull_stripping/model/skull_stripper_resnet18_3D.pth",
]
# STACKING OF THE DEAD
MODELS = [
    smp.Unet(
        encoder_name="timm-resnest50d",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ),
    # ),
    smp.Unet(
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ),
    smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ),
    smp.Unet(
        encoder_name="timm-resnest50d",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ),
    # ),
    smp.Unet(
        encoder_name="timm-efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ),
    smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ),
]


class Blender:
    def __init__(self, model_path=MODELPATH, models=MODELS):
        self.models = models
        for model_idx in range(len(models)):
            self.models[model_idx].load_state_dict(torch.load(model_path[model_idx]))
            self.models[model_idx].eval()

    def __call__(self, x):
        H, W = x.shape
        x = x / x.max()
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mask = torch.ones_like(x)
        for model in self.models:
            logits = model(x)
            mask *= nn.Sigmoid()(logits)

        mask = mask ** (1 / len(self.models))  # geometric mean
        mask = torch.where(mask < 0.5, 0, 1)
        self.mask = mask.squeeze(0).squeeze(0).numpy()
        # remove little artefact on edge
        return self.mask * x.squeeze(0).squeeze(0).numpy()

    def post_process(self):
        self.mask = binary_closing(
            self.mask, structure=np.ones((H // 3, W // 3))
        )  # fill big holes in mask
        self.mask = binary_fill_holes(self.mask).astype(int)  # fill small holes in mask
        self.mask = binary_erosion(self.mask, structure=np.ones((H // 25, W // 25)))


Data = namedtuple("Data", "path tensor")


REF = (
    np.asarray(
        Image.open("/home/mehdi.zouitine/skull_stripping/data/27/image/y_slice_9.png")
    )
    / 255
)