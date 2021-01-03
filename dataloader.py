from torch.utils.data import Dataset
import torch
import numpy as np
import glob
import nibabel as nib
import matplotlib.pyplot as plt


class SkullDataset3D(Dataset):
    def __init__(
        self, list_data_label, n_slice=(150, 125, 125), visualize=False, train=True
    ):
        self.patient = 100
        self.train = train
        self.n_slice = n_slice
        if train:
            self.data = list_data_label[0 : self.patient]
            self.label = list_data_label[0 : self.patient]
            self.n = self.patient
        else:
            self.data = list_data_label[self.patient :]
            self.label = list_data_label[self.patient :]
            self.n = 125 - self.patient

        self.visualize = visualize

    def __getitem__(self, idx):
        patient_idx = idx % self.n
        dim_idx = idx % 3
        slice_idx = idx % self.n_slice[dim_idx]
        if dim_idx == 0:
            data = self.data[patient_idx].data[70 + slice_idx, :, :]
            label = self.label[patient_idx].label[70 + slice_idx, :, :]

        elif dim_idx == 1:
            data = self.data[patient_idx].data[:, 40 + slice_idx, :]
            label = self.label[patient_idx].label[:, 40 + slice_idx, :]
        else:
            data = self.data[patient_idx].data[:, :, 30 + slice_idx][:, 0:192]
            label = self.label[patient_idx].label[:, :, 30 + slice_idx][:, 0:192]

        assert data.shape == (256, 192)
        assert label.shape == (256, 192)
        if self.visualize:
            _, axarr = plt.subplots(1, 2)
            axarr[0].imshow(data, cmap="gray")
            axarr[1].imshow(label, cmap="gray")
            plt.show()
        return {
            "data": torch.tensor(data, dtype=torch.float32).unsqueeze(0),
            "label": torch.tensor(label, dtype=torch.float32).unsqueeze(0),
        }

    def __len__(self):
        return self.n * np.sum(list(self.n_slice))
