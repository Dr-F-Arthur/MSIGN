# %%
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from graph_constructor import GraphDataset, collate_fn
from MSIGN import MSIGN

import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import warnings

from config.config_dict import *
from log.train_logger import *
from utils import *

import csv
warnings.filterwarnings('ignore')


# %%
def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        bg, label = data
        bg, label = bg.to(device), label.to(device)

        with torch.no_grad():
            pred_lp, pred_pl = model(bg)
            pred = (pred_lp + pred_pl) / 2
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    pr = pearsonr(pred, label)[0]
    rmse = np.sqrt(mean_squared_error(label, pred))


    return rmse, pr, label, pred


if __name__ == '__main__':
    graph_type = 'Atom_Graph2'
    data_root = './data'
    data_dir = os.path.join(data_root, 'out_pdb')
    data_df = pd.read_csv(os.path.join(data_root, "molecule.csv")).sample(frac=1., random_state=123)
    split_idx = int(0.9 * len(data_df))
    train_df = data_df.iloc[:split_idx]
    valid_df = data_df.iloc[split_idx:]

    train_set = GraphDataset(data_dir, train_df, graph_type=graph_type, create=False)
    valid_set = GraphDataset(data_dir, valid_df, graph_type=graph_type, create=False)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False, collate_fn=collate_fn)



    device = torch.device('cuda')
    model = MSIGN(node_feat_size=35, edge_feat_size=17, hidden_feat_size=256, layer_num=3).to(device)
    model.load_state_dict(torch.load("./model/full_finetune_model.pt"))

    train_rmse, train_pr, train_label, train_predict= val(model, train_loader, device)
    valid_rmse, valid_pr, valid_label, valid_predict = val(model, valid_loader, device)

    train_rows = zip(train_label, train_predict)
    # store results to csv file
    # with open('train_molecule.csv', 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(train_rows)
    print(train_rmse, train_pr)

    test_rows = zip(valid_label, valid_predict)
    # store results to csv file
    with open('test_molecule.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(test_rows)
    print(valid_rmse, valid_pr)

# %%
