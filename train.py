import time
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
from tqdm import tqdm
import segmentation_models_pytorch as smp


def train(train_dataloader, model, loss_function, optimizer, device, scheduler=None):
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


def evaluate(dataloader, model, loss_function, optimizer, device):
    iou = smp.utils.metrics.IoU(threshold=0.5)
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

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    print("End evaluate")

    return avg_loss, avg_metric


def learn(
    train_dataloader,
    val_dataloader,
    model,
    loss_function,
    optimizer,
    device,
    epochs,
    scheduler=None,
):
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

        # res_train = np.argmax(train_pred, axis=1).tolist()
        # res_val = np.argmax(val_pred, axis=1).tolist()

        # f1_train = f1_score(train_lab, res_train, average="macro")
        # f1_val = f1_score(val_lab, res_val, average="macro")

        # scheduler.step()

        print(f"\nTraining Loss: {train_loss:.3f} , training metric : {train_metric}")
        print(f"Validation Loss: {valid_loss:.3f}, val metric : {val_metric}")

        now = time.time()
        print(f"Time for epoch {epoch} is {(now - start)/60} min")
        #torch.save(model.module.state_dict(), "model/skull_stripper_timm-efficientnet-b4_3D.pth")
