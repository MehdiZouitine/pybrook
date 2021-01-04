import torch.nn as nn
import torch


class UnetSgm(nn.Module):
    def __init__(self, unet):

        super(UnetSgm, self).__init__()

        self.encoder = unet
        # if freeze:
        #     for param in self.unet.parameters():
        #         param.requires_grad = False

    def forward(self, x):

        embedding = self.encoder(x)
        return x