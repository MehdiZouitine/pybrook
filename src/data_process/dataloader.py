from torch.utils.data import Dataset
import torch
import numpy as np
import glob
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image


class SkullDataset3D(Dataset):
    def __init__(self, visualize: bool = False, train: bool = True):
        """[Pytorch Dataset object that generate MRI images and segmentation mask.]

        Args:
            visualize (bool, optional): [True if you want to see the pair data/mask]. Defaults to False.
            train (bool, optional): [If you want a train loader or a test loader ]. Defaults to True.
        """

        self.train = train
        self.visualize = visualize
        if train:
            self.img_path = glob.glob("../data/[0-9][0-9]/image/*")
            self.label_path = glob.glob("../data/[0-9][0-9]/label/*")
        else:
            self.img_path = glob.glob("../data/[0-9][0-9][0-9]/image/*")
            self.label_path = glob.glob("../data/[0-9][0-9][0-9]/label/*")

    def __getitem__(self, idx: int):
        img = np.asarray(Image.open(self.img_path[idx])) / 255
        label = np.asarray(Image.open(self.label_path[idx])) / 255

        if self.visualize:
            _, axarr = plt.subplots(1, 2)
            axarr[0].imshow(img, cmap="gray")
            axarr[1].imshow(label, cmap="gray")
            plt.show()
        return {
            "data": torch.tensor(img, dtype=torch.float32).unsqueeze(0),
            "label": torch.tensor(label, dtype=torch.float32).unsqueeze(0),
        }

    def __len__(self):
        return len(self.img_path)
