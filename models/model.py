import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import torch.nn.functional as F
from models.GNN_drug import GNN_drug
from models.VNN_cell import FullyNet, VNN_cell_PLUS



class  Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        #drug
        self.drug_gnn=args.drug_gnn
        self.drug_layer = args.drug_layer
        self.drug_hidden_dim = args.drug_hidden_dim
        self.drug_dim = args.drug_output_dim 
        #cell line
        self.num_feature = args.num_feature
        self.vnn_mode=args.vnn_mode
        self.cell_dim = args.cell_output_dim
        self.vnn_dropout_ratio=args.vnn_dropout_ratio
        self.child_neuron_ratio=args.child_neuron_ratio
        self.use_average_neuron_n=args.use_average_neuron_n
        self.only_combine_child_gene_group=args.only_combine_child_gene_group
        self.dropout_ratio = args.dropout_ratio
        self.classify=args.classify
        
        self.GNN_drug = GNN_drug(self.drug_layer, self.drug_dim, self.drug_hidden_dim, self.drug_gnn) 
        if self.vnn_mode=='ref' or self.vnn_mode=="random":
            model_class=VNN_cell_PLUS
        elif self.vnn_mode=='full':
            model_class=FullyNet
        else:
            print("Run mode error!")
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=4,kernel_size=(3, 3))
        self.ln = nn.Linear(self.drug_dim, 3)
        self.VNN_cell = model_class(input_dim=self.num_feature, 
                                    output_dim=self.cell_dim,
                                    drug_dim=self.drug_dim,
                                    only_combine_child_gene_group=self.only_combine_child_gene_group,
                                    neuron_ratio=self.child_neuron_ratio,
                                    use_average_neuron_n=self.use_average_neuron_n,
                                    run_mode=self.vnn_mode,
                                    dropout_p=self.vnn_dropout_ratio) 
        
        self.regression = nn.Sequential(
            nn.Linear(self.drug_dim + self.cell_dim, 256),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(256, 1)
        )
        self.sigmid=nn.Sigmoid()
        
    def forward(self, drug, cell):
        x_drug = self.GNN_drug(drug)
        cell = cell.transpose(1, 0)
        x_cell = self.VNN_cell(cell, x_drug)
        x = torch.cat([x_drug, x_cell], -1)
        x = self.regression(x)
        if self.classify:
            print(1)
            x=self.sigmid(x)
        return x



