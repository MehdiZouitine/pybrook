from torch.utils.data import Dataset
import torch
import numpy as np
import glob
import nibabel as nib
import matplotlib.pyplot as plt


class SkullDataset(Dataset):
    # dataset
    def __init__(self, data_regex, label_regex, visualize=False, train=True):
        self.patient = 100
        self.train = train
        if train:
            self.data_path = glob.glob(data_regex)[0 : self.patient]
            self.label_path = glob.glob(label_regex)[0 : self.patient]
            self.n = self.patient
        else:
            self.data_path = glob.glob(data_regex)[self.patient :]
            self.label_path = glob.glob(label_regex)[self.patient :]
            self.n = 125 - self.patient

        self.visualize = visualize

    def __getitem__(self, idx):
        patient_idx = idx % self.n
        img = nib.load(self.data_path[patient_idx]).get_fdata()
        mask = nib.load(self.label_path[patient_idx]).get_fdata()
        slice_idx = idx % 90
        data = img[:, 40 + slice_idx, :]
        label = mask[:, 40 + slice_idx, :]
        if self.visualize:
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(data, cmap="gray")
            axarr[1].imshow(label, cmap="gray")
            plt.show()
        return {
            "data": torch.tensor(data, dtype=torch.float32).unsqueeze(0),
            "label": torch.tensor(label, dtype=torch.float32).unsqueeze(0),
        }

    def __len__(self):
        return self.n * 90
