import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv,SAGEConv,global_mean_pool
from torch_scatter import scatter
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from models.CMPNN import cmpnn
from torch_geometric.utils import add_self_loops


class Att_pooling(nn.Module):
    def __init__(self, hidden_dim, drug_dim, num_heads):
        super(Att_pooling, self).__init__()
        
        self.num_heads = num_heads

        self.ATFC = nn.Sequential(
                                nn.Linear(hidden_dim, 64)
                                ,nn.LeakyReLU()
                                ,nn.LayerNorm(64, eps=1e-6)
                                ,nn.Linear(64, num_heads)
                                )
        self.output_block = nn.Sequential(
                                        nn.Linear(num_heads*hidden_dim, hidden_dim)
                                        ,nn.LeakyReLU()
                                        ,nn.LayerNorm(hidden_dim, eps=1e-6)
                                        ,nn.Linear(hidden_dim, drug_dim)
                                        ,nn.LeakyReLU()
                                        ,nn.LayerNorm(drug_dim, eps=1e-6)
                                        ,nn.Dropout(p=0.1),
                                        )


    def forward(self, feat, batch):
        size = int(batch.max().item() + 1)
        
        att = self.ATFC(feat)    # [L, att_heads]

        att = torch.exp(att)
        s = scatter(att, batch, dim=0, dim_size=size, reduce='add')
        s = s[batch]
        att = att / s    
        
        out = att.unsqueeze(-1)@feat.unsqueeze(1)
        out = scatter(out, batch, dim=0, dim_size=size, reduce='add')
        out = torch.flatten(out, start_dim=1)
        
        out = self.output_block(out)
        return out


class GNN_drug(torch.nn.Module):
    def __init__(self, drug_layer, drug_output_dim, drug_hidden_dim, gnn='gat'):
        super().__init__()
        self.layer_drug = drug_layer
        self.dim_drug = drug_output_dim
        self.hidden_dim = drug_hidden_dim
        self.gnn = gnn
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()
        self.dropout_ratio = 0.1

        for i in range(self.layer_drug):
            input_dim = self.hidden_dim
            if self.gnn == 'GIN':
                block = nn.Sequential(nn.Linear(input_dim, 2*self.hidden_dim), nn.ReLU(),
                                        nn.Linear(2*self.hidden_dim, self.hidden_dim))
                conv = GINConv(block)
            elif self.gnn == 'GCN':
                conv = GCNConv(in_channels=input_dim, out_channels=self.hidden_dim)
            elif self.gnn == 'SAGE':
                conv = SAGEConv(in_channels=input_dim, out_channels=self.hidden_dim)
            elif self.gnn == 'CMPNN':
                conv = cmpnn(in_channels=input_dim, hidden_dim=self.hidden_dim, out_channels=self.hidden_dim, edge_dim=input_dim)
   
            self.convs_drug.append(conv)
            bn = torch.nn.BatchNorm1d(self.hidden_dim)
            self.bns_drug.append(bn)
            
        if self.gnn != 'CMPNN':
            self.drug_emb = nn.Sequential(
                nn.Linear(self.hidden_dim, self.dim_drug),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_ratio),
            )
                
        self.att_pooling = Att_pooling(self.hidden_dim, self.dim_drug, 4)
        self.atom_encoder = AtomEncoder(self.hidden_dim)
        self.edge_encoder = BondEncoder(self.hidden_dim)
        

    def forward(self, drug):
        x, edge_index, edge_attr, batch = drug.x, drug.edge_index, drug.edge_attr, drug.batch
        
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))[0]
        self_loop_attr = torch.zeros(x.size(0), 3)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)
        
        x = self.atom_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        edge_index = edge_index.to(torch.long)
        edge_attr += x[edge_index[0]]
        
        x_drug_list = []
        for i in range(self.layer_drug):
            if self.gnn == 'CMPNN':
                x, edge_attr = self.convs_drug[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs_drug[i](x, edge_index)
            x = self.bns_drug[i](torch.relu(x))
            if i != self.layer_drug - 1:
                x = F.dropout(x, self.dropout_ratio)
                if self.gnn == 'CMPNN':
                    edge_attr = F.dropout(edge_attr, self.dropout_ratio)
            x_drug_list.append(x)

        node_representation = x_drug_list[-1]
        if self.gnn == 'CMPNN':
            x_drug = self.att_pooling(node_representation, batch)
        else:
            x_drug = global_mean_pool(node_representation, batch)
            x_drug = self.drug_emb(x_drug)
        return x_drug

