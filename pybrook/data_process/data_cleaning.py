import os
import numpy as np
import nibabel as nib
import torch
import glob
from tqdm import tqdm
from PIL import Image
from typing import NoReturn, List


LABEL_REGEX = "NFBS_Dataset/*/*brainmask*"
DATA_REGEX = "NFBS_Dataset/*/*T1w.nii.gz"
gs = lambda x: np.uint8((x / x.max()) * 255)
data_path = glob.glob(DATA_REGEX)
label_path = glob.glob(LABEL_REGEX)


def clean_patient(data_path: str, label_path: str, patient_id: str):

    """[Recovers data from the NFBS dataset in MRI format (3D),
     then transforms it into a numpy array and slices it (2D).
     Allows to have data in the form (slice2D, mask_Slice2D).
     By processing each slice we go from 125 label images to 47500
    label images.]

    Args:
        data_path (str): [Path of the skull]
        label_path (str): [Path of the mask]
        patient_id (str): [Id for each patient]
    """
    data, label = nib.load(data_path).get_fdata(), nib.load(label_path).get_fdata()
    # Get the Axial, coronal and sagittal slice of the brain and select only images that contains brain
    data_x, label_x = data[70:200, :, :], label[70:200, :, :]
    data_y, label_y = data[:, 40:165, :], label[:, 40:165, :]
    data_z, label_z = data[:, :, 30:155], label[:, :, 30:155]
    if patient_id < 10:
        patient_id = "0" + str(
            patient_id
        )  # Will be use full to create regex for the dataloaders
    for x in range(data_x.shape[0]):
        data_im = Image.fromarray(gs(data_x[x, :, :]))
        label_im = Image.fromarray(gs(label_x[x, :, :]))
        data_im.save(f"data/{patient_id}/image/x_slice_{x}.png")
        label_im.save(f"data/{patient_id}/label/x_slice_{x}.png")
        # x mean first kind of slice
    for y in range(data_y.shape[1]):
        data_im = Image.fromarray(gs(data_y[:, y, :]))
        label_im = Image.fromarray(gs(label_y[:, y, :]))
        data_im.save(f"data/{patient_id}/image/y_slice_{y}.png")
        label_im.save(f"data/{patient_id}/label/y_slice_{y}.png")
        # y mean second kind of slice
    for z in range(data_z.shape[2]):
        data_im = Image.fromarray(gs(data_z[:, :, z][:, 0:192]))
        label_im = Image.fromarray(gs(label_z[:, :, z][:, 0:192]))
        data_im.save(f"data/{patient_id}/image/z_slice_{z}.png")
        label_im.save(f"data/{patient_id}/label/z_slice_{z}.png")
        # z mean last kind of slice


def clean(data_path: List[str], label_path: List[str]):
    os.mkdir(f"data")
    for patient_id in tqdm(range(len(data_path))):
        if patient_id < 10:
            tmppatient = "0" + str(patient_id)
        else:
            tmppatient = patient_id
        os.mkdir(f"data/{tmppatient}")
        os.mkdir(f"data/{tmppatient}/image")
        os.mkdir(f"data/{tmppatient}/label")
        clean_patient(data_path[patient_id], label_path[patient_id], patient_id)


# if __name__ == "__main__":
#     LABEL_REGEX = "NFBS_Dataset/*/*brainmask*"
#     DATA_REGEX = "NFBS_Dataset/*/*T1w.nii.gz"

#     data_path = glob.glob(DATA_REGEX)
#     label_path = glob.glob(LABEL_REGEX)
#     clean(data_path, label_path)
