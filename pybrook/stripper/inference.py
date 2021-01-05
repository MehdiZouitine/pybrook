from PIL import Image
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from collections import namedtuple
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_erosion
import cv2
from skimage.exposure import match_histograms
from typing import List

MODELPATH = [
    "../model/skull_stripper_timm-efficientnet-b4_3D.pth",
    "../model/skull_stripper_resnet18_3D.pth",
]

MODELS = [
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

Segmented = namedtuple("Segmented", "brain mask")

# the reference images use to make histograms matching
REF = np.asarray(Image.open("../example/y_slice_9.png")) / 255


class Brook:
    def __init__(self, model_path: List[str] = MODELPATH, models: List[str] = MODELS):
        """[Core class of pybrook package. That class strip a 2D SLICE OF NUMPY TENSOR]
        Args:
            model_path (List[str], optional): [Path of pretrained model weight, SHOULD BE
            CHANGE by the user]. Defaults to MODELPATH.
            models (List[str], optional): [Models architecture (should be in the following
            order [timm-efficientnet-b4,resnet18])]. Defaults to MODELS.
        """
        self.models = models
        for model_idx in range(len(models)):
            self.models[model_idx].load_state_dict(torch.load(model_path[model_idx]))
            self.models[model_idx].eval()

    def strip(
        self, x: np.ndarray, pre_process: bool = False, post_process: bool = False
    ) -> Segmented:
        """[Strip a slice of array,using stacking of 2 pretrained models.]

        Args:
            x (np.ndarray): [2D tensor (slice of the MRI)]
            pre_process (bool, optional): [True to make histogram matching beetween new MRI and
            training MRI]. Defaults to False.
            post_process (bool, optional): [True to apply morphological operator to fill holes
            and remove artefacts]. Defaults to False.

        Returns:
            Segmented: [Namedtuple that contain, the segmented brain images and the segmentation mask]
        """

        H, W = x.shape
        cp = x.copy()
        if pre_process:
            ref = cv2.resize(REF, dsize=(H, W), interpolation=cv2.INTER_CUBIC)
            x = match_histograms(x, ref, multichannel=False)
        if x.max() > 1:
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

        return Segmented(cp * mask, mask)
