import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from IPython.display import clear_output

import metrics_eval

import wandb


def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies):  # better use wandb
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


def training_epoch(model, optimizer, criterion, train_loader,
                    tqdm_desc, device='cpu', threshold=None):
    train_loss = 0.0
    acc = 0
    model.train()
    for feat, labels in tqdm(train_loader, desc=tqdm_desc):
        # print(feat.shape)
        feat = feat.to(device)
        if labels:
            labels = labels.to(device)

        optimizer.zero_grad()
        res = model(feat, feat)      # src=tgt 
        loss = criterion(res, feat)  # loss: float
        loss.backward()
        optimizer.step()

        full_loss = torch.nn.MSELoss(reduce=False)(res, feat)
        if threshold:
            acc  += metrics_eval.get_accuracy(full_loss, labels, thresh=threshold)
        else:
            acc += metrics_eval.get_accuracy(full_loss, labels)
        train_loss += loss.item() * feat.shape[0]

    train_loss /= len(train_loader)
    acc /= len(train_loader)

    return train_loss, acc


def vae_loss_function(recon_x, x, mu, logvar):
    batch_size = recon_x.shape[0]
    MSE = torch.nn.functional.mse_loss(recon_x.view(batch_size,-1), x.view(batch_size, -1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE, KLD


def train_vae_epoch(model, optimizer, train_loader,
                    tqdm_desc, device='cpu'):
    train_mse, train_kld, train_loss = 0, 0, 0
    
    model.train()
    for feat, _ in tqdm(train_loader, desc=tqdm_desc):
        feat = feat.to(device)

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(feat, feat)     # src=tgt 

        mse_loss, kld_loss = vae_loss_function(recon_batch, feat, mu, logvar)
        loss = mse_loss + kld_loss
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        train_loss += loss.item()
        train_mse += mse_loss.item()
        train_kld += kld_loss.item()
        optimizer.step()

    train_loss /= len(train_loader)
    train_mse /= len(train_loader)
    train_kld /= len(train_loader)

    return train_loss, train_mse, train_kld


@torch.no_grad()
def validation_vae_epoch(model, criterion, test_loader,
                      tqdm_desc, epoch, device='cpu', threshold=None):
    test_loss = 0.0

    full_val_loss = None
    recons = None

    model.eval()
    for feat, labels in tqdm(test_loader, desc=tqdm_desc):
        feat = feat.to(device)
        labels = torch.Tensor(labels)
        labels = labels.to(device)  # labels: batch_size

        res, _, _ = model(feat, feat)  # src=tgt 
        loss = criterion(res, feat)

        full_loss = torch.nn.MSELoss(reduction='none')(res, feat)

        test_loss += loss.item() * feat.shape[0]

        if full_val_loss is None:
            full_val_loss =  full_loss.detach().cpu().numpy()
            recons = res.detach().cpu().numpy()
        else:
            full_val_loss = np.concatenate((full_val_loss, full_loss.detach().cpu().numpy()), axis=0)
            recons = np.concatenate((recons, res.detach().cpu().numpy()), axis=0)

    test_loss /= len(test_loader)
    return test_loss, full_val_loss, recons


@torch.no_grad()
def validation_epoch(model, criterion, test_loader,
                      tqdm_desc, epoch, device='cpu', threshold=None):
    '''
    return: test_loss (mean loss) : float,
            test_acc: float,
            full_val_loss: np.array((len(test_loader) * bath_size) x window x feat_dim)
    '''
    test_loss = 0.0
    acc = 0
    full_val_loss = None
    recons = None

    model.eval()

    for feat, labels in tqdm(test_loader, desc=tqdm_desc):
        feat = feat.to(device)
        labels = torch.Tensor(labels)
        labels = labels.to(device)  # labels: batch_size

        res = model(feat, feat)  # src=tgt 

        loss = criterion(res, feat)

        full_loss = torch.nn.MSELoss(reduction='none')(res, feat)
        if threshold:
            acc  += metrics_eval.get_accuracy(full_loss, labels, thresh=threshold)
        else:
            acc += metrics_eval.get_accuracy(full_loss, labels)

        test_loss += loss.item() * feat.shape[0]

        if full_val_loss is None:
            full_val_loss =  full_loss.detach().cpu().numpy()
            recons = res.detach().cpu().numpy()
        else:
            full_val_loss = np.concatenate((full_val_loss, full_loss.detach().cpu().numpy()), axis=0)
            recons = np.concatenate((recons, res.detach().cpu().numpy()), axis=0)        

    test_loss /= len(test_loader)
    acc /= len(test_loader)

    return test_loss, acc, full_val_loss, recons


def train_vae(model, optimizer, scheduler, criterion,
           train_loader, test_loader, num_epochs,
            device='cpu', threshold=0.1, start_epoch=1,
            save_checkpoints=False,
            val_labels_path=None,
            name_post=""):
    train_losses, test_losses = [], []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_loss, train_mse, train_kld = train_vae_epoch(
            model, optimizer, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}',
            device=device,
        )

        print(train_loss, train_mse, train_kld)

        test_loss, full_test_loss, recons = validation_vae_epoch(
            model, criterion, test_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}',
            epoch=epoch,
            device=device,
            threshold=threshold,
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        test_losses += [test_loss]

        plot_output(model, test_loader, device=device, output=recons,  param_dim=recons.shape[-1])

        assert np.sum(np.isnan(full_test_loss)) < 100
        loss_w = full_test_loss.mean(axis=2)
        loss_w = loss_w.reshape(-1)

        if save_checkpoints and epoch != 1 and epoch % 10 == 0:
            torch.save(model.state_dict(), f'.\checkpoints\{model.name}{epoch - 1}e_{name_post}.pt')
        
        test_labels = np.load(val_labels_path)

        dict_wandb = {'train_loss': train_loss, 'val_loss': train_loss}

        if (epoch - 1) % 5 == 0: 
            print(epoch, "evaluating")
            val_results = metrics_eval.evaluate(loss_w, test_labels, validation_thresh=None)
            dict_wandb = dict_wandb | val_results
        
        wandb.log(dict_wandb)        

    return train_losses, test_losses


def train(model, optimizer, scheduler, criterion,
           train_loader, test_loader, num_epochs,
            device='cpu', threshold=0.1, start_epoch=1,
            save_checkpoints=False,
            val_labels_path=None,
            name_post=""):
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_loss, train_accuracy = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}',
            device=device,
            threshold=threshold,
        )

        test_loss, test_accuracy, full_test_loss, recons = validation_epoch(
            model, criterion, test_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}',
            epoch=epoch,
            device=device,
            threshold=threshold,
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        train_accuracies += [train_accuracy]
        test_losses += [test_loss]
        test_accuracies += [test_accuracy]

        plot_output(model, test_loader, device=device, output=recons, param_dim=recons.shape[-1])

        assert np.sum(np.isnan(full_test_loss)) < 100

        if np.sum(np.isnan(full_test_loss)) > 0:
            print(recons)
        
        loss_w = full_test_loss.mean(axis=2)
        # print(loss_w.shape)
        loss_w = loss_w.reshape(-1)
        # print(loss_w.shape)


        if save_checkpoints and epoch != 1 and epoch % 10 == 0:
            torch.save(model.state_dict(), f'.\checkpoints\{model.name}{epoch - 1}e_{name_post}.pt')
        
        test_labels = np.load(val_labels_path)

        dict_wandb = {'train_loss': train_loss, 'val_loss': train_loss,
                    "train_acc": train_accuracy, "val_acc": test_accuracy}
        
        if (epoch - 1) % 4 == 0: 
            print(epoch, "evaluating")
            val_results = metrics_eval.evaluate(loss_w, test_labels, validation_thresh=None)
            dict_wandb = dict_wandb | val_results
        
        wandb.log(dict_wandb)        

    return train_losses, test_losses, train_accuracies, test_accuracies
    


@torch.no_grad()
def evaluate_loss(loss, labels, thresh=0.1):
    loss_w = loss.mean(dim=2)
    predictions = np.float32(loss_w.cpu() > thresh)
    if labels is None or len(labels) == 0:  # train
        labels = np.zeros(predictions.shape)
    else:
        labels = labels.cpu().numpy()
    return (predictions == labels).mean()


@torch.no_grad()
def get_series(model, test_loader,  device='cpu'):
    result = None
    for feat, _ in test_loader:
        feat = feat.to(device)
        pred = model(feat, feat)
        if len(pred) == 3:  # vae output
            pred = pred[0]
        pred = pred.detach().cpu().numpy()
        if result is None:
            result = pred
        else:
            result = np.concatenate((result, pred), axis=0)
    return result


def array_from_loader(loader):
    data_list = []
    for inputs, _ in loader:
        data_list.append(inputs)
    return torch.cat(data_list).numpy()


@torch.no_grad()
def plot_output(model, loader, device='cpu', output=None,
                 param_dim=19, max_plots=7,
                   input=None, model_name="AE",
                   begin_idx=0, end_idx=-1,
                   labels=None):
    clear_output()
    if output is None:
        output = get_series(model, loader, device=device).reshape(-1, param_dim)
    else:
        output = output.reshape(-1, param_dim)

    if input is None:
        input = array_from_loader(loader).reshape(-1, param_dim)
    
    print(f"data is ready!")

    num_plots = min(max_plots, output.shape[1])
    fig, axs = plt.subplots(2 * num_plots, 1, figsize =(10, num_plots*3))
    fig.tight_layout()
    # wandb_dict = {}
    for i in range(num_plots):
        # axs[2*i+1].set_ylim([input[:, i].min() - 0.5, input[:, i].max() + 0.5])
        axs[2*i+1].set_ylim([input[:, i].min(), input[:, i].max()])

        axs[2*i].plot(input[begin_idx:end_idx, i])

        if labels is not None:  # To show the anomaly points
            one_idx = np.where(np.diff(labels, prepend=0) == 1)[0]
            zero_idx = np.where(np.diff(labels, prepend=0) == -1)[0]  # len(one_idx) == len(zero_idx), else error
            for j in range(len(one_idx)):
                st = one_idx[j]
                end = zero_idx[j]
                axs[2*i].plot(np.arange(st, end), input[st:end, i], color="red")

        axs[2*i+1].plot(output[begin_idx:end_idx, i])

        axs[2*i+1].set_title(f"{model_name} output [{i}]", fontsize=7)
        axs[2*i].set_title(f"input [{i}]", fontsize=7)

    plt.show()

     