import yaml
import os
import re
import pandas as pd
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict as ddict
import matplotlib.pyplot as plt
import random
import sys


class VNN_preprocess():
    def __init__(self, gene_name_list):
        self.path = "./"    
        self.community_hierarchy_path=os.path.join(self.path,"data/Reactome/ReactomePathwaysRelation.txt")
        self.community_path = os.path.join(self.path,"data/Reactome/ReactomePathways.gmt")
        self.result_dir="./data/"
        self.gene_name_list=gene_name_list
     
    def _prepare_data(self):
        self._load_communities() 
        self._filter_communities()
        self._build_hierarchy()

    def _perform(self):
        self._prepare_data()
        self._save_data()
        return self.community_hierarchy_dicts_all

    def _load_communities(self):
        lines = open('{}'.format(self.community_path)).readlines()
        if 'pathway' in self.community_path.lower(): 
            ind_key = 1
            ind_gene = 3
        elif self.community_path.lower().endswith('.gmt'):
            ind_key = 1
            ind_gene = 3
        else:
            ind_key = 0
            ind_gene = 1
        self.community_genes = set()
        self.community_dict = {}
        self.gene_community_dict = ddict(list)
        self.community_size_dict = {}
        for line in lines:
            line = line.strip().split('\t')
            self.community_dict[line[ind_key]] = line[ind_gene:]
            self.community_size_dict[line[ind_key]] = len(line[ind_gene:])
            self.community_genes |= set(line[ind_gene:])
            for g in line[ind_gene:]:
                self.gene_community_dict[g].append(line[ind_key])

    def _load_known_genes(self):
        self.cell_line_metadata = pd.read_csv(os.path.join(self.path,"data/CCLE/Model.csv"))
        self.cell_line_metadata = self.cell_line_metadata.set_index('ModelID')
        self.cell_line_id_mapping = self.cell_line_metadata['COSMICID'].to_dict()
        self.cell_line_id_mapping = ddict(lambda: None, self.cell_line_id_mapping)

    def _filter_communities(self,community_affected_size_min=5,community_affected_size_max=999999):
        com_to_drop = []
        modeled_com_genes = set()
        modeled_genes = set()

        modeled_genes |= set(self.gene_name_list)
        for com, members in self.community_dict.items(): 
            self.community_dict[com] = sorted(list(set(modeled_genes).intersection(members)))
            if len(self.community_dict[com]) < community_affected_size_min:
                com_to_drop.append(com)
            elif len(self.community_dict[com]) > community_affected_size_max:
                com_to_drop.append(com)
            elif len(set(members) & set(self.gene_name_list)) < 1:
                com_to_drop.append(com)
            else:
                modeled_com_genes |= set(self.community_dict[com])
        for com in com_to_drop:
            self.community_dict.pop(com, None)


    def _build_hierarchy(self):
        leaf_communities, df = self._load_leaf_communities() 
        child = leaf_communities
        level = 1      
        self.community_level_dict = dict()
        self.level_community_dict = dict()
        count_dict = ddict(int)

        for x in child:
            self.community_level_dict[x] = level
            count_dict[x] += 1
        self.level_community_dict[level] = child

        while 1:
            df_level = df.loc[df[1].isin(child)] 
            if df_level.shape[0] == 0:
                break
            level += 1
            parent = sorted(list(set(df_level[0])))
            
            for parent_group in parent:
                self.community_level_dict[parent_group] =  level  
                count_dict[parent_group] += 1
            self.level_community_dict[level] = parent
            child = parent

        self.level_community_dict = ddict(list) 
        for g, level in  self.community_level_dict.items():
            self.level_community_dict[level].append(g)
        for level, groups in sorted(self.level_community_dict.items()):
            print("Layer {} has {} gene groups".format(level, len(groups)))

        gene_groups_all = sorted(list(self.community_dict.keys())) +['root']
        genes_all=self.gene_name_list 
        entity_all = genes_all + gene_groups_all

        self.idx_name = {i: k for i, k in enumerate(entity_all)}
        name_idx = ddict(list)
        for k, v in self.idx_name.items():
            name_idx[v].append(k)
        self.genes_idx = {x: min(name_idx[x]) for x in genes_all}
        self.gene_group_idx = {x: name_idx[x][0] for x in gene_groups_all}

        
        gene_pool = sorted(list(set(genes_all) ))
        self.child_map_all = [] 
        self.child_map_all_random = []

        self.community_dict_random = {}
        random_hierarchy = pd.DataFrame()
        self.gene_community_dict_random = ddict(list)

        idx_gene_pool = {i: g for i, g in enumerate(gene_pool)}
        partially_shuffled_membership = None
        partially_shuffled_relation = None
        idx_gene_group = {i: g for g, i in self.gene_group_idx.items()}

        min_group_idx = min(self.gene_group_idx.values())
        prng = np.random.RandomState(134)

        for group, idx in sorted(self.gene_group_idx.items()):
            if group in self.community_dict:
                genes = self.community_dict[group]
                gene_idx = self._genes_to_idx(genes)
                if partially_shuffled_membership is not None:         
                    genes_random_idx = partially_shuffled_membership[idx - min_group_idx].nonzero()[0]
                    genes_random = sorted([idx_gene_pool[x] for x in genes_random_idx])
                else:
                    genes_num=random.randint(1,len(genes))
                    genes_random = sorted(list(prng.choice(gene_pool, genes_num, replace=False))) 
                
                self.community_dict_random[group] = genes_random 
                for g in genes_random:
                    self.gene_community_dict_random[g].append(group)

                genes_random = set(genes_random) & set(self.genes_idx.keys())
                gene_idx_random = [self.genes_idx[x] for x in genes_random]
            else:
                gene_idx = []
                gene_idx_random = []

            child = sorted(df.loc[df[0] == group, 1].tolist())
            child_idx = sorted([self.gene_group_idx[x] for x in child if x in self.gene_group_idx])        
            self.child_map_all.append(sorted(gene_idx + child_idx))
            
            if len(self.child_map_all[-1]) == 0:
                print("Gene group {} does not have children".format(group))

            if partially_shuffled_relation is not None:
                child_idx_random = partially_shuffled_relation[idx - min_group_idx, :].nonzero()[0]
                child_idx_random = [x + min_group_idx for x in child_idx_random]
                child_random = sorted([idx_gene_group[x] for x in child_idx_random])
            else:
                child_idx_random = []
                child_random = []
                for c in child:
                    child_level = self.community_level_dict[c]
                    c_random = prng.choice(self.level_community_dict[child_level], 1, replace=False)[0] # 从child c 所在的level里random一个
                    child_random.append(c_random)
                try:
                    childgroup_num=random.randint(1,len(child_random))
                except ValueError:
                    childgroup_num=len(child_random)
                else:
                    child_random=random.sample(child_random,childgroup_num)

                for child in child_random:
                    idx_random = self.gene_group_idx[child]
                    child_idx_random.append(idx_random)

            for rc in sorted(child_random):
                random_hierarchy = pd.concat([random_hierarchy, pd.DataFrame([group, rc]).T], axis=0)
            self.child_map_all_random.append(sorted(gene_idx_random + child_idx_random))

            try:
                assert len(gene_idx) == len(gene_idx_random), "Random gene number does not match"
            except AssertionError:
                # print("Don't match!")
                pass

        self._save_communities(self.community_dict_random,"random")
        random_hierarchy.to_csv(os.path.join(self.result_dir, 'random_group_hierarchy.tsv'),index=None, sep='\t', header=None)
        self.community_hierarchy_dicts_all = {'idx_name':self.idx_name,
                                            'genes_idx': self.genes_idx,
                                            'gene_group_idx': self.gene_group_idx,
                                            'community_level_dict':self.community_level_dict,
                                            "level_community_dict":self.level_community_dict,
                                            'mask':self.child_map_all,
                                            'mask_random':self.child_map_all_random}

    def _load_leaf_communities(self):
        f = self.community_hierarchy_path
        df = pd.read_csv(f, sep='\t', header=None)
        if 'Reactome' in f:
            df = df.loc[df[0].str.contains('HSA')]

        df_root = pd.DataFrame(columns=df.columns)
        for x in set(df[0]) - set(df[1]):
            if x in self.community_dict or 'GO:' in x:
                df_root = pd.concat([df_root, pd.DataFrame(['root', x]).T])
        df = df.loc[df[1].isin(self.community_dict.keys()) & df[0].isin(self.community_dict.keys())]
        df = pd.concat([df, df_root])
        leaf_communities = sorted(list((set(df[1]) - set(df[0])) & set(self.community_dict.keys())))
        return leaf_communities, df

    def _genes_to_idx (self,genes):
        feat_genes = set(genes) & set(self.genes_idx.keys())
        feat_gene_idx = [self.genes_idx[x] for x in feat_genes]
        return feat_gene_idx

    def _save_communities(self,community_dict,d=None):
        if d is None:
            fout = open(os.path.join(self.result_dir, 'community_list.tsv'), 'w')
            d = community_dict
            s = ''
        else:
            fout = open(os.path.join(self.result_dir, 'random_community_list.tsv'), 'w')
            s = '_random'
        for k, v in community_dict.items():
            fout.write('{}\n'.format('\t'.join([k + s] + v)))
        fout.close()

    def _save_data(self):
        with open(os.path.join(self.result_dir,"vnn_preprocess_data.pkl"),"wb") as f:
            pickle.dump(self.community_hierarchy_dicts_all,f)
        print("vnn_preprocess_data.pkl Successfully save!")


