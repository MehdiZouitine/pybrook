import os
import numpy as np
import nibabel as nib
import torch
import glob
from tqdm import tqdm
from PIL import Image
from collections import namedtuple
import psutil
import shutil


LABEL_REGEX = "NFBS_Dataset/*/*brainmask*"
DATA_REGEX = "NFBS_Dataset/*/*T1w.nii.gz"
gs = lambda x : np.uint8((x/x.max())*255)
data_path = glob.glob(DATA_REGEX)
label_path = glob.glob(LABEL_REGEX)
def clean_patient(data_path,label_path,patient_id):
    data,label = nib.load(data_path).get_fdata(), nib.load(label_path).get_fdata()
    data_x,label_x = data[70:200, :, :],label[70:200, :, :]
    data_y,label_y = data[:,40:165, :],label[:,40:165, :]
    data_z,label_z = data[:, :,30:155],label[:, :,30:155]
    if patient_id<10:
        patient_id = '0'+str(patient_id)
    for x in range(data_x.shape[0]):
        data_im = Image.fromarray(gs(data_x[x,:,:]))
        label_im = Image.fromarray(gs(label_x[x,:,:]))
        data_im.save(f"data/{patient_id}/image/x_slice_{x}.png")
        label_im.save(f"data/{patient_id}/label/x_slice_{x}.png")

    for y in range(data_y.shape[1]):
        data_im = Image.fromarray(gs(data_y[:,y,:]))
        label_im = Image.fromarray(gs(label_y[:,y,:]))
        data_im.save(f"data/{patient_id}/image/y_slice_{y}.png")
        label_im.save(f"data/{patient_id}/label/y_slice_{y}.png")
    
    for z in range(data_z.shape[2]):
        data_im = Image.fromarray(gs(data_z[:,:,z][:,0:192]))
        label_im = Image.fromarray(gs(label_z[:,:,z][:,0:192]))
        data_im.save(f"data/{patient_id}/image/z_slice_{z}.png")
        label_im.save(f"data/{patient_id}/label/z_slice_{z}.png")


def clean(data_path,label_path):
    os.mkdir(f"data")
    for patient_id in tqdm(range(len(data_path))):
        if patient_id<10:
            tmppatient = '0'+str(patient_id)
        else :
            tmppatient = patient_id
        os.mkdir(f"data/{tmppatient}")
        os.mkdir(f"data/{tmppatient}/image")
        os.mkdir(f"data/{tmppatient}/label")
        clean_patient(data_path[patient_id],label_path[patient_id],patient_id)



if __name__ == '__main__':
    LABEL_REGEX = "NFBS_Dataset/*/*brainmask*"
    DATA_REGEX = "NFBS_Dataset/*/*T1w.nii.gz"

    data_path = glob.glob(DATA_REGEX)
    label_path = glob.glob(LABEL_REGEX)
    clean(data_path,label_path)

