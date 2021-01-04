from PIL import Image
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from collections import namedtuple
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_erosion
import cv2
from skimage.exposure import match_histograms

MODELPATH = [
    # "/home/mehdi.zouitine/skull_stripping/model/skull_stripper_timm-resnest50d_3D.pth",
    "/home/mehdi.zouitine/skull_stripping/model/skull_stripper_timm-efficientnet-b4_3D.pth",
    "/home/mehdi.zouitine/skull_stripping/model/skull_stripper_resnet18_3D.pth",
]

MODELS = [
    # smp.Unet(
    #     encoder_name="timm-resnest50d",
    #     encoder_weights="imagenet",
    #     in_channels=1,
    #     classes=1,
    # ),
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


class Brook:
    def __init__(self, model_path=MODELPATH, models=MODELS):
        self.models = models
        for model_idx in range(len(models)):
            self.models[model_idx].load_state_dict(torch.load(model_path[model_idx]))
            self.models[model_idx].eval()
    

    def strip(self, x,pre_process=False,post_process=False):
        H, W = x.shape
        cp = x.copy()
        ref = cv2.resize(REF, dsize=(H,W), interpolation=cv2.INTER_CUBIC)
        x = match_histograms(x, ref, multichannel=False)
        if x.max()>1:
            x = x / x.max()
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mask = torch.ones_like(x)
        for model in self.models:
            logits = model(x)
            mask *= nn.Sigmoid()(logits)

        mask = mask ** (1 / len(self.models))  # geometric mean
        mask = torch.where(mask < 0.5, 0, 1)
        mask = mask.squeeze(0).squeeze(0).numpy()
        # remove little artefact on edge
        if post_process:
            mask = binary_closing(
            mask, structure=np.ones((H // 3, W // 3))
        )  # fill big holes in mask
            mask = binary_fill_holes(mask).astype(int)  # fill small holes in mask
            mask = binary_erosion(mask, structure=np.ones((H // 25, W // 25)))

        

        return Segmented(cp*mask,mask)


        


Segmented = namedtuple("Segmented", "brain mask")


REF = (
    np.asarray(
        Image.open("../example/y_slice_9.png")
    )
    / 255
)