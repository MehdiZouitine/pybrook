import time
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
from tqdm import tqdm
import segmentation_models_pytorch as smp


def train(
    train_dataloader: torch.utils.data.dataloader,
    model: nn.Module,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
):
    """[Perfom one training epoch]

    Args:
        train_dataloader (torch.utils.data.dataloader): [Pytorch dataloader]
        model (nn.Module): [Unet based model]
        loss_function (nn.Module): []
        optimizer (torch.optim.Optimizer): []
        device (torch.device): [Training was done on multi GPU]
        scheduler ([type], optional): [description]. Defaults to None.
    """
    model.train()

    total_loss = 0

    for step, batch in tqdm(enumerate(train_dataloader)):

        if step % 50 == 0 and not step == 0:
            print("  Batch {:>5,}  of  {:>5,}.".format(step, len(train_dataloader)))

        data = batch["data"].to(device)
        label = batch["label"].to(device)
        model.zero_grad()

        preds = model(data)

        loss = loss_function(preds, label)

        total_loss = total_loss + loss.item()

        loss.backward()

        optimizer.step()
        # scheduler.step()


def evaluate(
    dataloader: torch.utils.data.dataloader,
    model: nn.Module,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """[Perform one evaluation step]

    Args:
        dataloader (torch.utils.data.dataloader): [Pytorch dataloader]
        model (nn.Module): [Unet based model]
        loss_function (nn.Module): []
        optimizer (torch.optim.Optimizer): []
        device (torch.device): []

    Returns:
        [tuple]: [Loss and metric over all the dataloader]
    """

    iou = smp.utils.metrics.IoU(
        threshold=0.5
    )  # Metric choosen is IOU, could be DICE or Logloss
    print("\nEvaluating...")

    model.eval()

    total_loss = 0
    total_metric = 0

    total_preds = []
    total_labels = []

    for step, batch in enumerate(dataloader):

        # Progress update every 50 batches.
        if step % 10 == 0 and not step == 0:

            # Report progress.
            print("  Batch {:>5,}  of  {:>5,}.".format(step, len(dataloader)))

        data = batch["data"].to(device)
        label = batch["label"].to(device)

        with torch.no_grad():

            preds = model(data)

            loss = loss_function(preds, label)

            metric = iou(preds, label)

            total_loss = total_loss + loss.item()

            total_metric = total_metric + metric.item()

            preds = preds.detach().cpu().numpy()

            labels = label.detach().cpu().numpy()

            total_preds.append(preds)
            total_labels.append(labels)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(dataloader)
    avg_metric = total_metric / len(dataloader)

    total_preds = np.concatenate(total_preds, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    print("End evaluate")

    return avg_loss, avg_metric


def learn(
    train_dataloader: torch.utils.data.dataloader,
    val_dataloader: torch.utils.data.dataloader,
    model: nn.Module,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    scheduler=None,
):
    """[Perform alternatively one step for training and one step on evaluation across multiple epochs]

    Args:
        train_dataloader (torch.utils.data.dataloader): [description]
        val_dataloader (torch.utils.data.dataloader): [description]
        model (nn.Module): [description]
        loss_function (nn.Module): [description]
        optimizer (torch.optim.Optimizer): [description]
        device (torch.device): [description]
        epochs (int): [description]
        scheduler ([type], optional): [description]. Defaults to None.
    """
    for epoch in tqdm(range(epochs)):
        start = time.time()

        print("\n Epoch {:} / {:}".format(epoch + 1, epochs))

        # train model
        train(train_dataloader, model, loss_function, optimizer, device)

        train_loss, train_metric = evaluate(
            train_dataloader, model, loss_function, optimizer, device
        )

        # evaluate model
        valid_loss, val_metric = evaluate(
            val_dataloader, model, loss_function, optimizer, device
        )

        # scheduler.step()

        print(f"\nTraining Loss: {train_loss:.3f} , training metric : {train_metric}")
        print(f"Validation Loss: {valid_loss:.3f}, val metric : {val_metric}")

        now = time.time()
        print(f"Time for epoch {epoch} is {(now - start)/60} min")
