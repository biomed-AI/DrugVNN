import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime
import random
import os
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr,spearmanr
from tqdm import tqdm
import pickle
from sklearn import metrics
def train(model, loader, criterion, opt, scheduler, norm_factor, device):
    model.train()
    # for idx, data in enumerate(tqdm(loader, desc='Iteration', disable=False)):
    all_loss = 0
    mean = norm_factor[0]
    std = norm_factor[1]
    for idx, data in enumerate(tqdm(loader, disable=False)):
        drug, cell, label= data
        if isinstance(cell, list):
            drug, cell, label= drug.to(device), [feat.to(device) for feat in cell], label.to(device)
        else:
            drug, cell, label= drug.to(device), cell.to(device), label.to(device)
        output = model(drug, cell)
        del drug, cell
        label = label.view(-1, 1).float()
        loss = criterion(output, (label - mean) / std)
        # loss = loss.mean()
        all_loss += loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        # scheduler.step()

    return all_loss
def train_classify(model, loader, criterion, opt, scheduler, norm_factor, device):
    model.train()
    # for idx, data in enumerate(tqdm(loader, desc='Iteration', disable=False)):
    all_loss = 0
    mean = norm_factor[0]
    std = norm_factor[1]
    for idx, data in enumerate(tqdm(loader, disable=False)):
        drug, cell, label, binary_ic50 = data
        if isinstance(cell, list):
            drug, cell, label, binary_ic50= drug.to(device), [feat.to(device) for feat in cell], label.to(device),binary_ic50.to(device)
        else:
            drug, cell, label, binary_ic50= drug.to(device), cell.to(device), label.to(device),binary_ic50.to(device)
        output = model(drug, cell)
        del drug, cell
        binary_ic50=binary_ic50.view(-1,1).float()
        loss=criterion(output,binary_ic50)
        # loss = loss.mean()
        all_loss += loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        # scheduler.step()

    return all_loss

def validate(model, loader, norm_factor, device):
    model.eval()

    y_true = []
    y_pred = []
    
    mean = norm_factor[0]
    std = norm_factor[1]

    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc='Iteration', disable=False):
            drug, cell, label= data
            if isinstance(cell, list):
                drug, cell, label= drug.to(device), [feat.to(device) for feat in cell], label.to(device)
            else:
                drug, cell, label= drug.to(device), cell.to(device), label.to(device)
            output = model(drug, cell)
            del drug, cell
            output = output * std + mean
            total_loss += F.mse_loss(output, label.view(-1, 1).float(), reduction='sum')
            y_true.append(label.view(-1, 1))
            y_pred.append(output)

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    rmse = torch.sqrt(total_loss/ len(loader.dataset)).item()
    mae = mean_absolute_error(y_true.cpu(), y_pred.cpu())
    r2 = r2_score(y_true.cpu(), y_pred.cpu())
    r = pearsonr(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())[0]
    r_sp=spearmanr(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())[0]
    return {"RMSE":rmse, "MAE":mae, "R2":r2,"PCC":r,"SPCC":r_sp},y_pred.cpu().flatten().tolist(),y_true.cpu().flatten().tolist()

def validate_classify(model, loader, norm_factor, device):
    model.eval()

    y_true = []
    y_pred = []
    
    mean = norm_factor[0]
    std = norm_factor[1]

    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc='Iteration', disable=False):
            drug, cell, label,binary_ic50 = data
            if isinstance(cell, list):
                drug, cell, label, binary_ic50= drug.to(device), [feat.to(device) for feat in cell], label.to(device),binary_ic50.to(device)
            else:
                drug, cell, label, binary_ic50= drug.to(device), cell.to(device), label.to(device),binary_ic50.to(device)
            output = model(drug, cell)
            del drug, cell
            binary_ic50=binary_ic50.view(-1,1).float()
            y_true.append(binary_ic50)
            y_pred.append(output)

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    auROC_all = metrics.roc_auc_score(y_true.cpu(),y_pred.cpu())
    fpr,tpr,thred,= metrics.roc_curve(y_true.cpu(),y_pred.cpu())
    precision,recall,_, = metrics.precision_recall_curve(y_true.cpu(),y_pred.cpu())
    auPR_all = -np.trapz(precision,recall)
    return {"AUROC": auROC_all,"AUPR":auPR_all,},y_pred.cpu().flatten().tolist(),y_true.cpu().flatten().tolist()

class MyDataset2(Dataset):
    def __init__(self, drug_dict, cell_dict, IC, drug2thred):
        super(MyDataset2, self).__init__()
        self.drug, self.cell = drug_dict, cell_dict
        self.IC = IC
        IC.reset_index(drop=True, inplace=True) 
        self.drug_id = IC['DRUG_ID']
        self.cell_COSMIC= IC['COSMIC_ID']
        self.value = IC['LN_IC50']
        self.drug2thred=drug2thred
    def __len__(self):
        return len(self.value)
    
    def mean(self):
        return float(self.value.mean())
    
    def std(self):
        return float(self.value.std())

    def __getitem__(self, index):
        drug = self.drug[self.drug_id[index]]
        cell = torch.tensor(self.cell[self.cell_COSMIC[index]], dtype=torch.float).transpose(1,0)
        # cell = cell[:, 0].unsqueeze(-1)
        label = self.value[index]
        binary_IC50=1 if label<self.drug2thred[self.drug_id[index]] else 0 
        return drug, cell, label,binary_IC50


def _collate2(samples):
    drugs, cells, labels,binary_ic50 = map(list, zip(*samples))
    batched_drug = Batch.from_data_list(drugs)
    batched_cell = torch.stack(cells, dim=0)
    list_cell = batched_cell
    return batched_drug, list_cell, torch.tensor(labels),torch.tensor(binary_ic50)


class MyDataset(Dataset):
    def __init__(self, drug_dict, cell_dict, IC):
        super(MyDataset, self).__init__()
        self.drug, self.cell = drug_dict, cell_dict
        self.IC = IC
        IC.reset_index(drop=True, inplace=True) 
        self.drug_id = IC['DRUG_ID']
        self.cell_COSMIC= IC['COSMIC_ID']
        self.value = IC['LN_IC50']
    def __len__(self):
        return len(self.value)
    
    def mean(self):
        return float(self.value.mean())
    
    def std(self):
        return float(self.value.std())

    def __getitem__(self, index):
        drug = self.drug[self.drug_id[index]]
        cell = torch.tensor(self.cell[self.cell_COSMIC[index]], dtype=torch.float).transpose(1,0)
        # cell = cell[:, 0].unsqueeze(-1)
        label = self.value[index]
        return drug, cell, label


def _collate(samples):
    drugs, cells, labels= map(list, zip(*samples))
    batched_drug = Batch.from_data_list(drugs)
    batched_cell = torch.stack(cells, dim=0)
    list_cell = batched_cell
    return batched_drug, list_cell, torch.tensor(labels)


def load_data(IC, drug_dict, cell_dict, drug2thred, args):
    if args.setup == 'known': 
        if args.classify:
            with open("./data/cla_dataset.pkl",'rb') as f:
                data_dict=pickle.load(f)
                train_set, val_set, test_set = data_dict["train"], data_dict["valid"], data_dict["test"]   
        else:
            with open("./data/known_dataset.pkl",'rb') as f:
                data_dict=pickle.load(f)
                train_set, val_set, test_set = data_dict["train"], data_dict["valid"], data_dict["test"]       
    elif args.setup == 'leave_drug_out':
        with open("./data/leave_drug.pkl",'rb') as f:
            data_dict=pickle.load(f)
            train_set, val_set, test_set = data_dict["train"], data_dict["valid"], data_dict["test"]    
    elif args.setup == 'leave_cell_out':
        with open("./data/leave_cell.pkl",'rb') as f:
            data_dict=pickle.load(f)
            train_set, val_set, test_set = data_dict["train"], data_dict["valid"], data_dict["test"]   
    else:
        raise ValueError
        
    # mean, std = train_dataset.mean(), train_dataset.std()
    if args.classify:
        Dataset = MyDataset2
        collate_fn = _collate2
        train_dataset = Dataset(drug_dict, cell_dict, train_set, drug2thred)
        val_dataset = Dataset(drug_dict, cell_dict, val_set, drug2thred)
        test_dataset = Dataset(drug_dict, cell_dict, test_set, drug2thred)
    else:
        Dataset = MyDataset
        collate_fn = _collate
        train_dataset = Dataset(drug_dict, cell_dict, train_set)
        val_dataset = Dataset(drug_dict, cell_dict, val_set)
        test_dataset = Dataset(drug_dict, cell_dict, test_set)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=args.num_workers
                              )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=args.num_workers
                            )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                             num_workers=args.num_workers)

    return train_loader, val_loader, test_loader#, mean, std


class EarlyStopping():
    """
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
    """

    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            dt = datetime.datetime.now()
            folder = os.path.join(os.getcwd(), 'results')
            if not os.path.exists(folder):
                os.makedirs(folder)
            print("model_name: early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
                dt.date(), dt.hour, dt.minute, dt.second))
            filename = os.path.join(folder, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        """Update based on a new score.
        The new score is typically model performance on the validation set
        for a new epoch.
        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        '''Load the latest checkpoint
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cell_line_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    
    np.random.seed(seed)

    cell_counts = dataset['COSMIC_ID'].value_counts()
    cell_counts = dict(cell_counts.items())
    
    cells = np.random.permutation(list(cell_counts.keys()))
    
    data_len = len(dataset)
    train_cutoff = int(frac_train * data_len)
    valid_cutoff = int(frac_valid * data_len)
    train_idx, valid_idx, test_idx = [], [], []
    train_len, valid_len, test_len = 0, 0, 0
    for c in cells:
        if train_len + cell_counts[c] > train_cutoff:
            if valid_len + cell_counts[c] > valid_cutoff:
                test_idx.append(c)
                test_len += cell_counts[c]
            else:
                valid_idx.append(c)
                valid_len += cell_counts[c]
        else:
            train_idx.append(c)
            train_len += cell_counts[c]
            
    train_set = dataset[dataset['COSMIC_ID'].isin(train_idx)]
    val_set = dataset[dataset['COSMIC_ID'].isin(valid_idx)]
    test_set = dataset[dataset['COSMIC_ID'].isin(test_idx)]
    
    return train_set, val_set, test_set


def drug_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    
    np.random.seed(seed)
    
    cell_counts = dataset['DRUG_ID'].value_counts()
    cell_counts = dict(cell_counts.items())
    
    cells = np.random.permutation(list(cell_counts.keys()))
    
    data_len = len(dataset)
    train_cutoff = int(frac_train * data_len)
    valid_cutoff = int(frac_valid * data_len)
    train_idx, valid_idx, test_idx = [], [], []
    train_len, valid_len, test_len = 0, 0, 0
    for c in cells:
        if train_len + cell_counts[c] > train_cutoff:
            if valid_len + cell_counts[c] > valid_cutoff:
                test_idx.append(c)
                test_len += cell_counts[c]
            else:
                valid_idx.append(c)
                valid_len += cell_counts[c]
        else:
            train_idx.append(c)
            train_len += cell_counts[c]
            
    train_set = dataset[dataset['DRUG_ID'].isin(train_idx)]
    val_set = dataset[dataset['DRUG_ID'].isin(valid_idx)]
    test_set = dataset[dataset['DRUG_ID'].isin(test_idx)]
    print(test_idx)
    return train_set, val_set, test_set
    
