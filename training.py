import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from IPython.display import clear_output

import wandb

from sklearn.metrics import f1_score, accuracy_score


def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies):
    clear_output()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))  
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(test_losses) + 1), test_losses, label='test')
    axs[0].set_ylabel('loss')

    axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='train')
    axs[1].plot(range(1, len(test_accuracies) + 1), test_accuracies, label='test')
    axs[1].set_ylabel('accuracy')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()

    plt.show()
    # print(train_losses, test_losses)
    # print(f"Validation F1 score = {test_f1:.5f}")

def training_epoch(model, optimizer, criterion, train_loader,
                    tqdm_desc, device='cpu', threshold=0.05):
    train_loss = 0.0
    acc = 0
    model.train()
    for feat, labels in tqdm(train_loader, desc=tqdm_desc):
        # print(feat.shape)
        feat = feat.to(device)
        if labels:
            labels = labels.to(device)

        optimizer.zero_grad()
        res = model(feat, feat)  # src=tgt 
        loss = criterion(res, feat)  # .mean()
        loss.backward()
        optimizer.step()

        full_loss = torch.nn.MSELoss(reduce=False)(res, feat)
        acc  += evaluate_loss(full_loss, labels)
        train_loss += loss.item() * feat.shape[0]

    # print(train_loss)
    train_loss /= len(train_loader.dataset)
    acc /= len(train_loader.dataset)

    return train_loss, acc


@torch.no_grad()
def validation_epoch(model, criterion, test_loader,
                      tqdm_desc, device='cpu', threshold=0.05):
    test_loss = 0.0
    acc = 0
    model.eval()

    for feat, labels in tqdm(test_loader, desc=tqdm_desc):
        feat = feat.to(device)
        labels = torch.Tensor(labels)
        labels = labels.to(device)  # labels: batch_size

        res = model(feat, feat)  # src=tgt 
        loss = criterion(res, feat)

        full_loss = torch.nn.MSELoss(reduce=False)(res, feat)
        acc  += evaluate_loss(full_loss, labels)

        test_loss += loss.item() * feat.shape[0]
        

    test_loss /= len(test_loader.dataset)
    acc /= len(test_loader.dataset)

    return test_loss, acc


def train(model, optimizer, scheduler, criterion,
           train_loader, test_loader, num_epochs,
            device='cpu', threshold=0.05):
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}',
            device=device,
            threshold=threshold,
        )
        test_loss, test_accuracy = validation_epoch(
            model, criterion, test_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}',
            device=device,
            threshold=threshold,
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        train_accuracies += [train_accuracy]
        test_losses += [test_loss]
        test_accuracies += [test_accuracy]
        # test_f1_scores += [test_f1]
        plot_losses(train_losses, test_losses, train_accuracies, test_accuracies)
        wandb.log({'train_loss': train_loss, 'val_loss': train_loss,
                    "train_acc": train_accuracy, "val_acc": test_accuracy})

        if epoch % 10 == 0:
            plot_output(model, test_loader, device)
            if epoch != 0:
                torch.save(model.state_dict(), f'.\checkpoints\AE_{epoch}e.pt')

    return train_losses, test_losses, train_accuracies, test_accuracies


@torch.no_grad()
def evaluate_loss(loss, labels, thresh=0.05):
    # print(loss.shape)
    loss_w = loss.mean(dim=2)
    predictions = np.float32(loss_w.cpu() > thresh)
    if len(labels) == 0:  # train
        labels = np.zeros(predictions.shape)
    else:
        labels = labels.cpu().numpy()
    return (predictions == labels).sum(axis=1).mean()


@torch.no_grad()
def get_series(model, test_loader,  device='cpu', threshold=0.05):
    res_result = np.empty((64, 100, 19))
    for feat, labels in tqdm(test_loader):
        feat = feat.to(device)
        labels = torch.Tensor(labels)
        labels = labels.to(device)  # labels: batch_size
        
        pred = model(feat, feat).detach().cpu().numpy()
        res_result = np.concatenate((res_result, pred), axis=0) 
    return res_result

def array_from_loader(loader):
    data_list = []
    for inputs, _ in loader:
        data_list.append(inputs)
    return torch.cat(data_list).numpy()

@torch.no_grad()
def plot_output(model, loader, device='cpu'):
    val_output = get_series(model, loader,  device=device)
    val_input = array_from_loader(loader)
    fig, axs = plt.subplots(2 * val_output.shape[1], 1, figsize =(10, 70))
    fig.tight_layout()
    for i in range(val_output.shape[1]):
        axs[2*i].plot(val_input[:, i])
        axs[2*i+1].plot(np.where(val_output[:, i] > 10 or val_output[:, i] < -10, 3, val_output[:, i]))

        axs[2*i+1].set_title(f"AE output [{i}]", fontsize=7)
        axs[2*i].set_title(f"AE input [{i}]", fontsize=7)

    

     