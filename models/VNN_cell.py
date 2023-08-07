import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from scipy.optimize import fsolve
from time import time
from .VNN_utilis import VNN_preprocess
import pickle
import logging

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class VNN_cell_PLUS(nn.Module): 
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 drug_dim,
                 dropout_p=0.1,
                 only_combine_child_gene_group=True,
                 neuron_min=10, 
                 neuron_ratio=0.2,#是否需要保存
                 use_average_neuron_n=True,
                 run_mode='ref',
                 gene_feat=3): 
        
        super(VNN_cell_PLUS, self).__init__() 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drug_dim = drug_dim
        self.dropout_p = dropout_p
        self.only_combine_child_gene_group = only_combine_child_gene_group        
        self.gene_feat = gene_feat
        self.run_mode=run_mode
        self.total_num=0
        with open('./data/CCLE/gene_list_full_clean.pkl', 'rb') as f:
            self.gene_name_list = pickle.load(f)  
        self.level_neuron_ct = dict()
        self.com_layers = nn.ModuleDict()
        self.bn_layers = nn.ModuleDict()
        self.output_layers = nn.ModuleDict()
        self.drug_layers = nn.ModuleDict()
        if self.dropout_p > 0:
            self.dropout_layers = nn.ModuleDict()        
        self._preprocess()
        self._set_layer_names()
        self.build_order = []
        self.neuron_min = neuron_min
        self.neuron_ratio = neuron_ratio
        self._set_layers()
        self.act_func=Mish()
        self.sigmoid = nn.Sigmoid()
        self.output = [None] * len(self.build_order)
        if self.only_combine_child_gene_group:
            print("{} gene groups do not combine gene features".format(len(self.only_combine_gene_group_dict)))
        self.norm = nn.LayerNorm(541, eps=1e-6)
        self.dropout_d = nn.Dropout(0.8)
        self.drug_query = nn.Linear(self.drug_dim, self.gene_feat)
        self.gene_key = nn.Linear(3, self.gene_feat)
        self.gene_value = nn.Linear(3, self.gene_feat)

    def _preprocess(self): 
        preprocess_data=VNN_preprocess(self.gene_name_list)
        vnn_dict=preprocess_data._perform()
        if self.run_mode=='ref' or self.run_mode=='full':
            self.child_map = vnn_dict["mask"] 
        elif self.run_mode=='random':
            self.child_map=vnn_dict["mask_random"]
        self.group_level_dict = vnn_dict['community_level_dict']
        self.level_group_dict= vnn_dict["level_community_dict"]
        self.gene_group_idx = vnn_dict['gene_group_idx'] 
        self.idx_name = vnn_dict['idx_name']
        

    def _set_layers(self):
        self._build_layers()
        self._report_parameter_n()

    def _set_layer_names(self):
        for g in self.gene_group_idx.keys(): 
            self.com_layers[g] = None
            self.drug_layers[g] = None
            self.bn_layers['bn_{}'.format(g)] = None
            self.output_layers['output_{}'.format(g)] = None
            if self.dropout_p > 0:
                self.dropout_layers['drop_{}'.format(g)] = None


    def _build_layers(self):
        neuron_to_build = list(range(len(self.child_map)))
        self.only_combine_gene_group_dict = {}
        neuron_n_dict = dict()
        while len(neuron_to_build) > 0:
            for i in neuron_to_build:
                j = i + self.input_dim
                children = self.child_map[i] 
                child_feat = [z for z in children if z < self.input_dim] 
                child_com = [self.idx_name[z] for z in children if z >= self.input_dim] 
                child_none = [self.com_layers[z] for z in child_com if self.com_layers[z] is None] 
                if len(child_none) > 0:
                    continue
                neuron_name = self.idx_name[j] 
                if self.only_combine_child_gene_group and len(child_com) > 0:
                    child_n = len(child_com) 
                    if i == len(self.child_map) - 1: 
                        child_n = self.output_dim
                    child_feat = []
                    self.only_combine_gene_group_dict[neuron_name] = 1
                else:
                    child_n = len(children)
                    if i == len(self.child_map) - 1: 
                        child_n = self.output_dim 
                #print("Building gene group {} with {} children".format(j, len(children)))
                if i not in neuron_n_dict:
                    neuron_n = np.max([self.neuron_min, int(child_n * self.neuron_ratio)])
                    neuron_n_dict[i] = neuron_n
                else:
                    neuron_n = neuron_n_dict[i]
                level = self.group_level_dict[neuron_name]
                if level not in self.level_neuron_ct.keys():
                    self.level_neuron_ct[level] = neuron_n
                else:
                    self.level_neuron_ct[level] += neuron_n
                total_in = int(len(child_feat)*self.gene_feat + np.sum([self.com_layers[z].out_features for z in child_com])) 
                self.drug_layers[neuron_name] = nn.Linear(self.drug_dim, total_in)
                self.com_layers[neuron_name] = nn.Linear(total_in, neuron_n) 
                self.bn_layers['bn_{}'.format(neuron_name)] = nn.BatchNorm1d(neuron_n)
                self.total_num+=neuron_n
                if self.dropout_p > 0:
                    self.dropout_layers['drop_{}'.format(neuron_name)] = nn.Dropout(self.dropout_p)
                neuron_to_build.remove(i)
                self.build_order.append(i)


    def _report_parameter_n(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total {} parameters and {} are trainable".format(total_params, trainable_params))
        return trainable_params


    def _drug_gene_att(self, features, drug):
        gene_feat = features
        drug_feat = drug
        Q = self.dropout_d(self.drug_query(drug_feat))
        K = self.gene_key(gene_feat)
        V = self.gene_value(gene_feat)
        
        attention_scores = torch.sum(Q*K, dim=-1).unsqueeze(-1) 
        attention_scores = attention_scores.sigmoid()
        gene_feat = V*attention_scores + gene_feat
        
        return gene_feat, attention_scores


    def forward(self, features, drug): 
        gene_feat, att = self._drug_gene_att(features, drug)
        features = [None]*self.input_dim + self.output 

        for i in self.build_order:
            j = i + self.input_dim
            neuron_name = self.idx_name[j]
            
            com_layer = self.com_layers[neuron_name]
            bn_layer = self.bn_layers['bn_{}'.format(neuron_name)]

            children = self.child_map[i]
            if neuron_name in self.only_combine_gene_group_dict:
                children = [z for z in children if z >= self.input_dim]

            if self.group_level_dict[neuron_name] == 1:
                input_mat = gene_feat[children]
                input_mat = torch.flatten(input_mat.transpose(1,0), start_dim=1)
            else:
                input_list = [features[z] for z in children]
                input_mat = torch.cat(input_list, axis=1)
                
                
            state = com_layer(input_mat)
            state = bn_layer(self.act_func(state))

            if self.dropout_p > 0:
                drop_layer = self.dropout_layers['drop_{}'.format(neuron_name)]
                features[j] = drop_layer(state)
            else:
                features[j] = state

        out = features[-1]
        return out
    
    
class FullyNet(VNN_cell_PLUS):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 drug_dim,
                 dropout_p=0.1,
                 only_combine_child_gene_group=True, 
                 neuron_min=10, 
                 neuron_ratio=0.2,
                 use_average_neuron_n=True, 
                 run_mode='ref',
                 gene_feat=3):
                
        super(FullyNet, self).__init__(input_dim,output_dim,drug_dim,dropout_p,only_combine_child_gene_group,
                                    neuron_min,neuron_ratio, use_average_neuron_n,run_mode,gene_feat) 

        self.use_average_neuron_n = use_average_neuron_n
        parameter_n = self._report_parameter_n()
        self._build_layers_fully(parameter_n)
        self._report_parameter_n()
    def _solve_neuron_n(self,layer_n,parameter_n):
        def func(i):
            x = i[0]
            return [self.input_dim * x + (layer_n-1) * (x ** 2 + x) - parameter_n]

        r = fsolve(func, [0])
        return int(r[0])

    def _build_layers_fully(self, parameter_n=39974, layer_n=13):
        self.com_layers = None
        self.fully_layers = nn.ModuleDict()
        self.bn_layers = nn.ModuleDict()
        self.output_layers = nn.ModuleDict()
        if self.dropout_p > 0:
            self.dropout_layers = nn.ModuleDict()
        self.build_order = []
        total_n = parameter_n // self.input_dim 
        if total_n / float(layer_n) < 1:  
            self.build_order.append(0)
            self.fully_layers['fully_0'] = nn.Linear(self.input_dim, total_n)
            self.bn_layers['bn_0'] = nn.BatchNorm1d(total_n)
            if self.dropout_p > 0:
                self.dropout_layers['drop_0'] = nn.Dropout(self.dropout_p)
            print("The fully connected network has 1 layers")
        else:
            # neuron_per_layer = self._solve_neuron_n(layer_n,parameter_n)
            # neuron_per_layer = total_n // layer_n #试一下
            neuron_per_layer=self.total_num//layer_n
            print("neuron_per_layer:",neuron_per_layer)
            for i in range(layer_n):# 不知道怎么写
                self.build_order.append(i)
                if self.use_average_neuron_n:
                    print("Use average_neuron_n:",neuron_per_layer)
                    if i == 0:
                        self.fully_layers['fully_' + str(i)] = nn.Linear(self.input_dim*self.gene_feat, neuron_per_layer)
                        self.bn_layers['bn_' + str(i)] = nn.BatchNorm1d(neuron_per_layer)
                    elif i ==layer_n-1:
                        self.fully_layers['fully_' + str(i)] = nn.Linear(neuron_per_layer, self.output_dim)
                        self.bn_layers['bn_' + str(i)] = nn.BatchNorm1d(self.output_dim)
                    else:
                        self.fully_layers['fully_' + str(i)] = nn.Linear(neuron_per_layer, neuron_per_layer)
                        self.bn_layers['bn_' + str(i)] = nn.BatchNorm1d(neuron_per_layer)
                    
                else:
                    print("Don't use average_neuron_n")
                    if i == 0:
                        in_n = self.input_dim*self.gene_feat
                        out_n = self.level_neuron_ct[i + 1]
                    else:
                        in_n = self.level_neuron_ct[i]
                        out_n = self.level_neuron_ct[i + 1]

                    self.fully_layers['fully_' + str(i)] = nn.Linear(in_n, out_n)
                    print("The fully connected network layer {} has {} neurons".format(i, out_n))
                    self.bn_layers['bn_' + str(i)] = nn.BatchNorm1d(out_n)

                if self.dropout_p > 0:
                    self.dropout_layers['drop_' + str(i)] = nn.Dropout(self.dropout_p)

        self.output = [None] * len(self.build_order)


    def forward(self, features,drug): 
        gene_feat,att=self._drug_gene_att(features,drug)
        features = [None]*self.input_dim + self.output

        for i in range(13):
            j = i + self.input_dim
            neuron_name = i
            fully_layer = self.fully_layers['fully_{}'.format(neuron_name)]
            bn_layer = self.bn_layers['bn_{}'.format(neuron_name)]
            if i == 0:
                input_list = [gene_feat[z] for z in range(self.input_dim)]
                input_mat = torch.cat(input_list, axis=1)
            else:
                input_mat = features[j - 1]
            state = fully_layer(input_mat)
            state= bn_layer( self.act_func(state) )

            if self.dropout_p > 0:
                drop_layer = self.dropout_layers['drop_{}'.format(neuron_name)]
                features[j] = drop_layer(state)
            else:
                features[j] = state

        return features[-1]

