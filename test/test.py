# %%
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Add the parent directory to the path to allow module import.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from graph_constructor import GraphDataset, collate_fn
from MSIGN import MSIGN
from utils import load_model_dict

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import warnings
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

    model.train()

    return rmse, pr, pred, label


if __name__ == '__main__':
    # Project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_root = os.path.join(project_root, 'data')

    # Data directory (all graph files for the test sets are located in the v2020-other-PL directory)
    graph_data_dir = os.path.join(data_root, 'PDBBind','Lppdb')

    # Read the CSV file and rename the column names to match the format expected by graph_constructor.
    column_rename = {'pdbid': 'BindingDB MonomerID', '-logKd/Ki': 'Gbinding Average'}

    test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv')).rename(columns=column_rename)
    test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv')).rename(columns=column_rename)
    test2019_df = pd.read_csv(os.path.join(data_root, 'test2019.csv')).rename(columns=column_rename)

    # Filter out pdbids with missing image files.
    def filter_existing(df, data_dir):
        mask = df['BindingDB MonomerID'].apply(
            lambda x: os.path.exists(os.path.join(data_dir, str(x), f'Atom_Graph2-{x}.dgl')))
        removed = len(df) - mask.sum()
        if removed > 0:
            print(f"Filtered out {removed} pdbids without graph files")
        return df[mask].reset_index(drop=True)

    test2013_df = filter_existing(test2013_df, graph_data_dir)
    test2016_df = filter_existing(test2016_df, graph_data_dir)
    test2019_df = filter_existing(test2019_df, graph_data_dir)

    # Building the dataset (external test set only)
    test2013_set = GraphDataset(graph_data_dir, test2013_df, graph_type='Atom_Graph2', create=False)
    test2016_set = GraphDataset(graph_data_dir, test2016_df, graph_type='Atom_Graph2', create=False)
    test2019_set = GraphDataset(graph_data_dir, test2019_df, graph_type='Atom_Graph2', create=False)

    # Build the DataLoader (using num_workers=0 to avoid insufficient shared memory issues).
    test2013_loader = DataLoader(test2013_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test2016_loader = DataLoader(test2016_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test2019_loader = DataLoader(test2019_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)

    device = torch.device('cuda:0')

    # ============================================
    # The file path for the weight files from the three repeated experiments (please fill in)
    # ============================================
    model_paths = [
        '/data/model/model_weight_final/pdbs_pretrain_model.pt',
        '/data/model/model_weight_final/pdbs_pretrain_model.pt',
        '/data/model/model_weight_final/pdbs_pretrain_model.pt'
    ]

    # Store the results of each experiment.
    results = {
        'test2013_rmse': [], 'test2013_pr': [],
        'test2016_rmse': [], 'test2016_pr': [],
        'test2019_rmse': [], 'test2019_pr': []
    }

    # The experiment was repeated three times.
    for i, model_path in enumerate(model_paths):
        print(f"\n{'='*60}")
        print(f"Repeat {i}: {model_path}")
        print('='*60)

        # Initialize the model
        model = MSIGN(node_feat_size=35, edge_feat_size=17, hidden_feat_size=256, layer_num=3).to(device)
        load_model_dict(model, model_path)

        # External test set evaluation
        test2013_rmse, test2013_pr, test2013_pred, test2013_label = val(model, test2013_loader, device)
        test2016_rmse, test2016_pr, test2016_pred, test2016_label = val(model, test2016_loader, device)
        test2019_rmse, test2019_pr, test2019_pred, test2019_label = val(model, test2019_loader, device)

        # Save the prediction results to a CSV file.
        test2013_result = pd.DataFrame({
            'pdbid': test2013_df['BindingDB MonomerID'].values,
            'label': test2013_label,
            'pred': test2013_pred
        })
        test2013_result.to_csv(f'test2013_predictions_repeat{i}.csv', index=False)

        test2016_result = pd.DataFrame({
            'pdbid': test2016_df['BindingDB MonomerID'].values,
            'label': test2016_label,
            'pred': test2016_pred
        })
        test2016_result.to_csv(f'test2016_predictions_repeat{i}.csv', index=False)

        test2019_result = pd.DataFrame({
            'pdbid': test2019_df['BindingDB MonomerID'].values,
            'label': test2019_label,
            'pred': test2019_pred
        })
        test2019_result.to_csv(f'test2019_predictions_repeat{i}.csv', index=False)

        print(f"Predictions saved to test2013/2016/2019_predictions_repeat{i}.csv")

        # Store the results.
        results['test2013_rmse'].append(test2013_rmse)
        results['test2013_pr'].append(test2013_pr)
        results['test2016_rmse'].append(test2016_rmse)
        results['test2016_pr'].append(test2016_pr)
        results['test2019_rmse'].append(test2019_rmse)
        results['test2019_pr'].append(test2019_pr)

        # Print single result
        msg = "test2013_rmse-%.4f, test2013_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f, test2019_rmse-%.4f, test2019_pr-%.4f" \
              % (test2013_rmse, test2013_pr, test2016_rmse, test2016_pr, test2019_rmse, test2019_pr)
        print(msg)

    # Calculate the mean and standard deviation.
    print(f"\n{'='*60}")
    print("Summary (Mean ± Std)")
    print('='*60)

    for dataset in ['test2013', 'test2016', 'test2019']:
        rmse_key = f'{dataset}_rmse'
        pr_key = f'{dataset}_pr'

        rmse_mean = np.mean(results[rmse_key])
        rmse_std = np.std(results[rmse_key])
        pr_mean = np.mean(results[pr_key])
        pr_std = np.std(results[pr_key])

        print(f"{dataset}: RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}, PR = {pr_mean:.4f} ± {pr_std:.4f}")

# %%
