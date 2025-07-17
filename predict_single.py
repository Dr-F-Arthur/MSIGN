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

if __name__ == '__main__':
    data_root = './data'
    device = torch.device('cuda')

    molecule_dir1 = os.path.join(data_root, 'out_pdb/molecule_1')
    pred_dir1 = os.path.join(molecule_dir1, 'Atom_Graph2-molecule_1.dgl')

    molecule_dir2 = os.path.join(data_root, 'out_pdb/molecule_19')
    pred_dir2 = os.path.join(molecule_dir2, 'Atom_Graph2-molecule_19.dgl')

    molecule_dir3 = os.path.join(data_root, 'out_pdb/molecule_105')
    pred_dir3 = os.path.join(molecule_dir3, 'Atom_Graph2-molecule_105.dgl')

    graph_data1, label1 = torch.load(pred_dir1)
    gd1 = graph_data1.to(device)

    graph_data2, label2 = torch.load(pred_dir2)
    gd2 = graph_data2.to(device)

    graph_data3, label3 = torch.load(pred_dir3)
    gd3 = graph_data3.to(device)




    model = MSIGN(node_feat_size=35, edge_feat_size=17, hidden_feat_size=256, layer_num=3).to(device)
    model.load_state_dict(torch.load("./model/full_finetune_model.pt"))

    model.eval()
    pred_p1, pred_c1 = model(gd1)
    pred1 = (pred_p1 + pred_c1) / 2

    pred_p2, pred_c2 = model(gd2)
    pred2 = (pred_p2 + pred_c2) / 2

    pred_p3, pred_c3 = model(gd3)
    pred3 = (pred_p3 + pred_c3) / 2

    print("molecule_1:Prediction:%.4f" % pred1)
    print("molecule_19:Prediction:%.4f" % pred2)
    print("molecule_105:Prediction:%.4f" % pred3)

# %%
