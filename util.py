import os

import numpy as np
import pandas as pd

import torch

from sklearn import metrics


def save_np(path, npfile):
    if path is None:
        return

    dir, _ = os.path.split(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.save(path, npfile)


def save_csv(filename, data, header=False, mode='w'):
    if filename is None:
        return

    if data is None:
        return

    dir, _ = os.path.split(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)

    df = pd.DataFrame(data, index=[0])

    # 写入csv文件,'a+'是追加模式
    try:
        df.to_csv(f'{filename}.csv', header=header, index=False, mode=mode, encoding='utf-8')
    except UnicodeEncodeError:
        print('error')


def cal_auc(label, predict):
    # pred = torch.sigmoid(predict)
    return metrics.roc_auc_score(label.detach().cpu().numpy(), predict.detach().cpu().numpy())