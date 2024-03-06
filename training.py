import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from IPython.display import clear_output

import wandb

from sklearn.metrics import f1_score, accuracy_score


def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies, test_f1):
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
    print(train_losses, test_losses)
    print(f"Validation F1 score = {test_f1:.5f}")

def training_epoch(model, optimizer, criterion, train_loader,
                    tqdm_desc, device='cpu', threshold=0.05):
    train_loss = 0.0
    POS_sum, NEG_sum = 0, 0
    # model.train()
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

        full_loss = torch.nn.MSELoss(reduction='none')(res, feat)
        TP, TN, NEG = evaluate_loss(full_loss, labels)
        POS_sum += (TP + TN)
        NEG_sum += NEG

        train_loss += loss.item() * feat.shape[0]

    print(train_loss)
    train_loss /= len(train_loader.dataset)
    train_accuracy =  (POS_sum) / (NEG_sum + POS_sum)

    return train_loss, train_accuracy


@torch.no_grad()
def validation_epoch(model, criterion, test_loader,
                      tqdm_desc, device='cpu', threshold=0.05):
    test_loss = 0.0
    TP_sum, TN_sum, NEG_sum = 0, 0, 0
    model.eval()

    for feat, labels in tqdm(test_loader, desc=tqdm_desc):
        feat = feat.to(device)
        labels = torch.Tensor(labels)
        labels = labels.to(device)  # labels: batch_size

        res = model(feat, feat)  # src=tgt 
        loss = criterion(res, feat)

        full_loss = torch.nn.MSELoss(reduce=False)(res, feat)
        TP, TN, NEG = evaluate_loss(full_loss, labels)
        TP_sum += TP
        TN_sum += TN
        NEG_sum += NEG

        # loss = loss.mean()
        test_loss += loss.item() * feat.shape[0]
        

    test_loss /= len(test_loader.dataset)
    test_accuracy =  (TN_sum + TP_sum) / (NEG_sum + TN_sum + TP_sum)
    test_f1 = 2. * TP_sum / (2. * TP_sum + NEG_sum)

    return test_loss, test_accuracy, test_f1


def train(model, optimizer, scheduler, criterion,
           train_loader, test_loader, num_epochs,
            device='cpu', threshold=0.05):
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    test_f1_scores = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}',
            device=device,
            threshold=threshold,
        )
        test_loss, test_accuracy, test_f1 = validation_epoch(
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
        test_f1_scores += [test_f1]
        plot_losses(train_losses, test_losses, train_accuracies, test_accuracies, test_f1)
        wandb.log({'train_loss': train_loss, 'val_loss': train_loss, 'f1': test_f1_scores})


    return train_losses, test_losses, train_accuracies, test_accuracies, test_f1_scores

@torch.no_grad()
def evaluate_loss(loss, label, thresh=0.02):
    # print(loss.shape)
    loss_w = loss.mean(dim=(0, 2))
    TP, FP, NEG = 0, 0, 0
    predictions = np.float32(loss_w.cpu() > thresh)
    # print(predictions)
    if len(label) == 0:  # train
        label = np.zeros(predictions.shape[0])
    else:
        label = label.cpu().numpy()
    TP = ((predictions == 1.) & (label == 1.)).sum()
    FP = ((predictions == 1.) & (label == 0.)).sum()
    NEG = (predictions != label).sum()

    return TP, FP, NEG

     