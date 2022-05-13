import time

import numpy as np

import math

from tqdm.autonotebook import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from data import get_dataloader
from util import cal_auc, save_csv, save_np

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def penalty(model, x, lamb):
    xv = model.embedding(x)
    xv_sq = xv.pow(2)
    xv_penalty = xv_sq * lamb
    xv_penalty = xv_penalty.sum()
    return xv_penalty


def calc_sparsity(self):
    base = self.feature_num * self.latent_dim
    non_zero_values = torch.nonzero(self.sparse_v).size(0)
    percentage = 1 - (non_zero_values / base)
    return percentage, non_zero_values


def get_threshold(model):
    return model.embedding.g(model.embedding.s)


def get_embedding(model):
    return model.embedding.sparse_v.detach().cpu().numpy()


def train(dataset, model,
          epoch_size=300, path='./log/pep-{model_name}-{dataset_name}/{file_name}',
          learning_rate=1e-3, weight_decay=1e-5, gamma=1):
    print('*' * 10, 'Start Training', '*' * 18)

    emb_nums = []

    data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    model = model.to(device)
    print('model:', model)

    param_num = model.embedding.feature_num * model.embedding.latent_dim
    print("BackBone Embedding Parameters: ", param_num)

    loss_function = BCEWithLogitsLoss(reduction='sum')
    optimiser = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ExponentialLR(optimiser, gamma=gamma)

    # save initial embedding param
    emb_path = path.format(file_name='embedding-initial')
    save_np(emb_path, get_embedding(model))

    ratios = [0.03, 0.05, 0.1, 0.5, 0.9]

    # the epoch loop
    for epoch in range(epoch_size):
        if len(ratios) == 0:
            break

        # Train
        losses = []
        auc_scores = []

        epoch_start = time.time()

        model.train()

        dataloader = tqdm(data_loader)
        for _, (inputs, labels) in enumerate(dataloader, 0):
            # get the inputs
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

            sparsity, params = calc_sparsity(model.embedding)

            for ratio in ratios:
                if math.fabs(params - int(param_num * ratio)) < 50:
                    emb_nums.append(params)
                    print(f'\nReach {ratio * 100}% Target, saving')
                    save_np(path.format(file_name=f'embedding-{params}'), get_embedding(model))
                    ratios.remove(ratio)
                    break

            dataloader.set_postfix(epoch=epoch, loss=np.mean(losses), auc=np.mean(auc_scores), sparsity=sparsity,
                                   params=params)

        if scheduler is not None:
            scheduler.step()

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        train_result = {'epoch': epoch, 'params_num': params, 'loss': np.mean(losses), 'auc': np.mean(auc_scores),
                'sparsity': sparsity, 'time': epoch_time}
        
        result_path = path.format(
            file_name='pep-train-log'
        )
        if epoch == 0:
            header = ['epoch', 'loss', 'auc', 'time', 'valid_loss', 'valid_acc']
            save_csv(result_path, train_result, header)
        else:
            save_csv(result_path, train_result, mode='a+')

    print('**** Finished Training ****\n')
    return emb_nums


def retrain(dataset, model, num,
            epoch_size=30, path='./log/pep-{model_name}-{dataset_name}/{file_name}',
            learning_rate=1e-3, weight_decay=1e-5, gamma=1):
    print('*' * 10, 'Start Training', '*' * 18)

    train_loader, valid_loader, test_loader = get_dataloader(dataset)

    model = model.to(device)
    print('model:', model)

    param_num = model.embedding.feature_num * model.embedding.latent_dim
    print("BackBone Embedding Parameters: ", param_num)

    loss_function = BCEWithLogitsLoss(reduction='sum')
    optimiser = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ExponentialLR(optimiser, gamma=gamma)

    # the epoch loop
    for epoch in range(epoch_size):
        # Train
        train_result = train_one_epoch(epoch, model, train_loader, loss_function, optimiser, scheduler=scheduler)

        # Validation
        valid_loss, valid_auc = valid_one_epoch(epoch, model, valid_loader, loss_function)
        train_result['valid_loss'] = valid_loss
        train_result['valid_acc'] = valid_auc

        result_path = path.format(
            file_name=f'retrain-log-{num}'
        )
        if epoch == 0:
            header = ['epoch', 'params_num', 'loss', 'auc', 'sparsity', 'time', 'valid_loss', 'valid_acc']
            save_csv(result_path, train_result, header)
        else:
            save_csv(result_path, train_result, mode='a+')

    # test
    test_result = test(model, test_loader, loss_function)

    test_path = path.format(file_name=f'retrain-test-log-{num}')
    header = ['test_loss', 'test_auc']
    save_csv(test_path, test_result, header)

    print('**** Finished Training ****\n')


def train_one_epoch(epoch, model, data_loader, loss_function, optimiser, scheduler=None):
    losses = []
    auc_scores = []

    epoch_start = time.time()

    model.train()

    dataloader = tqdm(data_loader)
    for _, (inputs, labels) in enumerate(dataloader, 0):
        # get the inputs
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

        sparsity, params = calc_sparsity(model.embedding)

        dataloader.set_postfix(epoch=epoch, loss=np.mean(losses), auc=np.mean(auc_scores), sparsity=sparsity,
                               params=params)

    if scheduler is not None:
        scheduler.step()

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    return {'epoch': epoch, 'params_num': params, 'loss': np.mean(losses), 'auc': np.mean(auc_scores),
            'sparsity': sparsity, 'time': epoch_time}


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

        sparsity, params = calc_sparsity(model.embedding)

        dataloader.set_postfix(epoch=epoch, loss=np.mean(losses), auc=np.mean(auc_scores), sparsity=sparsity,
                               params=params)

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
