import time
from tkinter import W

import numpy as np

from tqdm.autonotebook import tqdm

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from data import get_dataset, get_dataloader
from model import get_model
from util import save_csv, cal_auc

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def train_one_epoch(epoch, model, optimiser, data_loader, loss_function, scheduler=None):
    losses = []
    auc_scores = []

    epoch_start = time.time()

    model.train()

    dataloader = tqdm(data_loader)
    for _, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward 
        prob_preference = model(inputs)

        # loss + backward
        loss = loss_function(prob_preference, labels.float()) / (inputs.size()[0])
        loss.backward()
        losses.append(loss.item())

        # auc
        auc = cal_auc(labels, prob_preference)
        auc_scores.append(auc)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # optimise (update weights)
        optimiser.step()

        dataloader.set_postfix(epoch=epoch, loss=np.mean(losses), auc=np.mean(auc_scores))

    if scheduler is not None:
        scheduler.step()

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    return {'epoch': epoch, 'loss': np.mean(losses), 'auc': np.mean(auc_scores), 'time': epoch_time}


def valid_one_epoch(epoch, model, data_loader, loss_function):
    losses = []
    auc_scores = []

    model.eval()

    dataloader = tqdm(data_loader)
    for _, (inputs, labels) in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = inputs.to(device), labels.to(device)

        # forward 
        prob_preference = model(inputs)

        # loss + backward
        loss = loss_function(prob_preference, labels.float()) / (inputs.size()[0])
        losses.append(loss.item())

        # auc
        auc = cal_auc(labels, prob_preference)
        auc_scores.append(auc)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        dataloader.set_postfix(epoch=epoch, loss=np.mean(losses), auc=np.mean(auc_scores))

    return np.mean(losses), np.mean(auc_scores)


def test(model, data_loader, loss_function):
    losses = []
    auc_scores = []

    model.eval()

    dataloader = tqdm(data_loader)
    for _, (inputs, labels) in enumerate(dataloader, 0):
        # get the inputs
        inputs, labels = inputs.to(device), labels.to(device)

        # forward 
        prob_preference = model(inputs)

        # loss + backward
        loss = loss_function(prob_preference, labels.float()) / (inputs.size()[0])
        losses.append(loss.item())

        # auc
        auc = cal_auc(labels, prob_preference)
        auc_scores.append(auc)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # sparsity, params = calc_sparsity(model.embedding)

        dataloader.set_postfix(loss=np.mean(losses), auc=np.mean(auc_scores))

    return {'test_loss': np.mean(losses), 'test_auc': np.mean(auc_scores)}


def train(dataset, model,
          epoch_size=30, log_path='./log/bl-{model_name}-{dataset_name}/{file_name}',
          learning_rate=1e-3, weight_decay=1e-5, gamma=1):
    print('*' * 10, 'Start Training', '*' * 18)

    train_loader, valid_loader, test_loader = get_dataloader(dataset)

    model = model.to(device)
    print('model:', model)

    loss_function = BCEWithLogitsLoss(reduction='sum')
    optimiser = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ExponentialLR(optimiser, gamma=gamma)

    for epoch in range(epoch_size):
        # Train
        train_result = train_one_epoch(epoch, model, optimiser, train_loader, loss_function, scheduler)

        # Vaildation
        valid_loss, valid_auc = valid_one_epoch(epoch, model, valid_loader, loss_function)
        train_result['valid_loss'] = valid_loss
        train_result['valid_acc'] = valid_auc

        result_path = log_path.format(
            file_name='bl-train-log'
        )
        if epoch == 0:
            header = ['epoch', 'loss', 'auc', 'time', 'valid_loss', 'valid_acc']
            save_csv(result_path, train_result, header)
        else:
            save_csv(result_path, train_result, mode='a+')

        # save embedding param every 3 epoch and test it
    test_result = test(model, test_loader, loss_function)

    test_path = log_path.format(file_name='test-log')
    header = ['test_loss', 'test_auc']
    save_csv(test_path, test_result, header)
                
    print('**** Finished Training ****\n')
