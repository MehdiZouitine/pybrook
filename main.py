import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
from dataloader2 import SkullDataset3D
from train import learn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import segmentation_models_pytorch as smp
from tqdm import tqdm


model = smp.Unet(
    encoder_name="timm-resnest50d",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pretrained weights for encoder initialization
    in_channels=1,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your dataset)
)
# The model was train on several gpu's
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# Create 2 pytorch dataset one for training and one for validation
train_dataset = SkullDataset3D(visualize=False, train=True)
val_dataset = SkullDataset3D(visualize=False, train=False)

# Create pytorch dataloaders for generate batch of data and mask on shuffle samples
train_dataloader = DataLoader(
    train_dataset, batch_size=150, shuffle=True, num_workers=30
)
val_dataloader = DataLoader(val_dataset, batch_size=150, shuffle=True, num_workers=30)

# Use Adam optimizer to minimize the loss function. We retrain the pretrain layer with
# a small learning rate and the last layers with a bigger learning rate
optimizer = Adam(
    [
        {"params": model.module.encoder.parameters(), "lr": 1e-5},
        {"params": model.module.decoder.parameters(), "lr": 1e-4},
    ]
)
# Use BCEWL loss
loss_function = nn.BCEWithLogitsLoss()
# The pretrained model dont need so much epochs to give very good result.
# We could use early stopping but it work very well without
epochs = 8
device = "cuda:0"

# Push the model on gpu
model.to(device)
learn(
    train_dataloader,
    val_dataloader,
    model,
    loss_function,
    optimizer,
    device,
    epochs,
    scheduler=None,
)
# Save weight
torch.save(model.module.state_dict(), "model/skull_stripper_timm-resnest50d_3D.pth")
