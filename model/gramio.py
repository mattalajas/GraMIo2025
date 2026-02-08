import random
import math

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import LayerNorm
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn.models import GCN
from tsl.nn.layers.graph_convs import DiffConv
from tsl.nn import utils
from tsl.nn.layers.base import MultiHeadAttention
from tsl.nn.models.base_model import BaseModel
from utils import closest_distances_unweighted

EPSILON = 1e-8
ACTIVATIONS = {
    'relu': F.relu,
    'leakyrelu': F.leaky_relu,
    'elu': F.elu,
    'tanh': torch.tanh,
    'gelu': F.gelu,
    'selu': F.selu,
    'sigmoid': torch.sigmoid,
    'softplus': F.softplus
}

class SpatioTemporalTransformerLayer(nn.Module):
    r"""A :class:`~tsl.nn.blocks.encoders.TransformerLayer` which attend both
    the spatial and temporal dimensions by stacking two
    :class:`~tsl.nn.layers.base.MultiHeadAttention` layers.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        ff_size (int): Units in the MLP after self attention.
        n_heads (int, optional): Number of parallel attention heads.
        causal (bool, optional): If :obj:`True`, then causally mask attention
            scores in temporal attention.
            (default: :obj:`True`)
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout probability.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size=None,
                 n_heads=1,
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(SpatioTemporalTransformerLayer, self).__init__()
        self.temporal_att = MultiHeadAttention(embed_dim=hidden_size,
                                               qdim=input_size,
                                               kdim=input_size,
                                               vdim=input_size,
                                               heads=n_heads,
                                               axis='time',
                                               causal=causal)

        self.spatial_att = MultiHeadAttention(embed_dim=hidden_size,
                                              qdim=hidden_size,
                                              kdim=hidden_size,
                                              vdim=hidden_size,
                                              heads=n_heads,
                                              axis='nodes',
                                              causal=False)

        self.skip_conn = nn.Linear(input_size, hidden_size)

        self.norm1 = LayerNorm(input_size)
        self.norm2 = LayerNorm(hidden_size)

        self.mlp = nn.Sequential(LayerNorm(hidden_size),
                                 nn.Linear(hidden_size, ff_size),
                                 utils.get_layer_activation(activation)(),
                                 nn.Dropout(dropout),
                                 nn.Linear(ff_size, hidden_size),
                                 nn.Dropout(dropout))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        """"""
        # x: [batch, steps, nodes, features]
        x = self.skip_conn(x) + self.dropout(
            self.temporal_att(self.norm1(x))[0])
        x = x + self.dropout(
            self.spatial_att(self.norm2(x), attn_mask=mask)[0])
        x = x + self.mlp(x)
        return x
    
class Gramio(BaseModel):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 adj,
                 gcn_layers,
                 psd_layers,
                 cro_layers=None,
                 activation='tanh',
                 horizon=24,
                 cmd_sample_ratio=1.,
                 tra_sample_ratio=1.,
                 k=5,
                 att_heads=8,
                 dropout=0.1,
                 full_cadj=True,
                 sampling='partition'):
        super(Gramio, self).__init__()

        self.full_cadj = full_cadj
        self.horizon = horizon
        self.cmd_ratio = cmd_sample_ratio
        self.tra_ratio = tra_sample_ratio
        self.k = k
        self.att_heads = att_heads

        assert sampling in ['partition', 'random', 'empty', 'half', 'no_part']
        self.sampling = sampling

        if not cro_layers:
            cro_layers = gcn_layers
        
        self.activation = ACTIVATIONS[activation]

        self.init_emb = nn.Linear(input_size, hidden_size)
        self.init_emb_tr = nn.Linear(input_size, hidden_size)

        self.full_tras = SpatioTemporalTransformerLayer(input_size=hidden_size,
                                                       hidden_size=hidden_size,
                                                       ff_size=hidden_size,
                                                       n_heads=att_heads,
                                                       causal=False,
                                                       activation=activation,
                                                       dropout=dropout)

        self.layernorm00 = LayerNorm(hidden_size)
        self.layernorm01 = LayerNorm(hidden_size)
        self.layernorm0 = LayerNorm(hidden_size)
        self.layernorm1 = LayerNorm(hidden_size)
        self.layernorm2 = LayerNorm(hidden_size)
        self.layernorm3 = LayerNorm(hidden_size)
        
        self.gcn_air = GCN(in_channels=hidden_size,
                        hidden_channels=hidden_size,
                        num_layers=psd_layers,
                        out_channels=hidden_size,
                        norm='LayerNorm',
                        add_self_loops=None,
                        act=activation)
        self.gcn_cr = GCN(in_channels=hidden_size,
                        hidden_channels=hidden_size,
                        num_layers=psd_layers,
                        out_channels=hidden_size,
                        norm='LayerNorm',
                        add_self_loops=None,
                        act=activation)

        self.gcn_tr = DiffConv(in_channels=hidden_size,
                            out_channels=hidden_size,
                            k=cro_layers,
                            root_weight=True,
                            activation=activation)
        
        
        self.gcn2 = DiffConv(in_channels=hidden_size,
                            out_channels=hidden_size,
                            k=gcn_layers,
                            root_weight=True,
                            activation=activation)

        self.readout1 = nn.Linear(hidden_size, output_size)
        self.readout2 = nn.Linear(hidden_size*2, output_size)

        self.adj = adj

    def forward(self,
                x,
                x_exog,
                mask,
                split,
                known_set,
                masked_set=[],
                seened_set=[],
                sub_entry_num=0,
                edge_weight=None,
                training=False,
                reset=False,
                transform=None):
        # x: [batches steps nodes features]
        b, t, og_n, _ = x.size()
        device = x.device

        full_adj = torch.tensor(self.adj).to(device)
        t_adj = full_adj[split:, :]
        t_adj = t_adj[:, split:]

        if seened_set != []:
            o_adj = full_adj[seened_set, :]
            o_adj = o_adj[:, seened_set]

            c_adj = full_adj[seened_set, split:]
        else:
            o_adj = full_adj[known_set, :]
            o_adj = o_adj[:, known_set]

            c_adj = full_adj[known_set, split:]

        # Check if nodes arent connected to anything, 
        # if so add self loops
        zero_inds = torch.where((o_adj.sum(0) + o_adj.sum(1)) == 0)[0]
        o_adj[zero_inds, zero_inds] = 1.

        edge_index, edge_weight = dense_to_sparse(o_adj)
        a_fwd = self.init_emb(x)

        # ========================================
        # Get cross module embeddings
        # ========================================
        # Sample traffic nodes
        N_t = t_adj.shape[0]
        if self.tra_ratio < 1.0:
            n_tra = int(N_t*self.tra_ratio)
            tr_indx = torch.multinomial(torch.ones(N_t), n_tra, replacement=False).to(device)
        else:
            tr_indx = torch.arange(N_t).to(device)
            n_tra = N_t

        t_adj_sam = t_adj[tr_indx, :]
        t_adj_sam = t_adj_sam[:, tr_indx]
        x_exog = x_exog[:, :, tr_indx]

        # Get traffic embeddings
        traf_adj = dense_to_sparse(t_adj_sam)
        t_fwd = self.init_emb_tr(x_exog)

        t_fwd_caus = self.gcn_tr(t_fwd, traf_adj[0], traf_adj[1]) + t_fwd
        t_fwd = self.layernorm0(t_fwd_caus)

        # Intiialise cross embeddings
        cr_embs = torch.cat((a_fwd, t_fwd), dim=2)
        c_adj = c_adj[:, tr_indx]

        ar_adj = torch.zeros((o_adj.shape[0], o_adj.shape[0])).to(device)
        tr_adj = torch.zeros((tr_indx.shape[0], tr_indx.shape[0])).to(device)

        if self.full_cadj:
            cr_adj = torch.ones((o_adj.shape[0], tr_indx.shape[0])).to(device)
        else:
            cr_adj = c_adj.ne(0).to(c_adj.dtype)
        
        rows = [
            [ar_adj, cr_adj],
            [cr_adj.T, tr_adj]
        ]

        cross_adj = torch.cat(
            [torch.cat(row, dim=1) for row in rows],
            dim=0
        )

        # ========================================
        # Calculating spatiotemporal attention for both embeddings
        # ========================================
        a_fwd = self.layernorm00(self.full_tras(a_fwd, o_adj) + a_fwd)
        cr_embs = self.layernorm01(self.full_tras(cr_embs, cross_adj) + cr_embs)

        output_air = a_fwd
        output_cro = cr_embs

        # ========================================
        # Create new adjacency matrix 
        # ========================================
        if seened_set != []:
            arrange = seened_set + masked_set

            o_adj = full_adj[arrange, :]
            o_adj = o_adj[:, arrange]

            c_adj = full_adj[arrange, split:]

            if self.full_cadj:
                c_adj = torch.ones_like(c_adj[:, tr_indx]).to(device)
            else:
                c_adj = c_adj[:, tr_indx].ne(0).to(c_adj.dtype)

        if training:
            # inductive
            if reset:
                numpy_graph = nx.from_numpy_array(o_adj.cpu().numpy())
                target_nodes = list(range(o_adj.shape[0]))[:len(seened_set)]
                source_nodes = list(range(o_adj.shape[0]))[len(seened_set):]

                init_hops = closest_distances_unweighted(numpy_graph, source_nodes, target_nodes)
                adj_aug, level_hops, c_adj = self.get_new_adj(o_adj, self.k, n_add=sub_entry_num, cross_adj=c_adj, init_hops=init_hops)

            else:
                adj_aug = o_adj

                numpy_graph = nx.from_numpy_array(adj_aug.cpu().numpy())
                target_nodes = list(range(adj_aug.shape[0]))[:len(known_set)]
                source_nodes = list(range(adj_aug.shape[0]))
                level_hops = closest_distances_unweighted(numpy_graph, source_nodes, target_nodes)

            adj = adj_aug
        else:
            arrange = known_set + masked_set

            n_adj = full_adj[arrange, :]
            n_adj = n_adj[:, arrange]

            c_adj = full_adj[arrange, split:]

            if self.full_cadj:
                c_adj = torch.ones_like(c_adj[:, tr_indx]).to(device)
            else:
                c_adj = c_adj[:, tr_indx].ne(0).to(c_adj.dtype)

            numpy_graph = nx.from_numpy_array(n_adj.cpu().numpy())
            target_nodes = list(range(n_adj.shape[0]))[:len(known_set)]
            source_nodes = list(range(n_adj.shape[0]))
            level_hops = closest_distances_unweighted(numpy_graph, source_nodes, target_nodes)

            adj = n_adj

        b, t, _, d = output_air.shape
        if seened_set != []:
            add_nodes = len(masked_set) + sub_entry_num
        else:
            add_nodes = sub_entry_num

        if add_nodes != 0:
            sub_entry = torch.zeros(b, t, add_nodes, d).to(device)

            xh_air = torch.cat([output_air, sub_entry], dim=2)  # b t n2 d
            xh_cro = torch.cat([output_cro[:, :, :og_n], sub_entry, output_cro[:, :, og_n:]], dim=2)
        else:
            xh_air = output_air
            xh_cro = output_cro

        # ========================================
        # Multimodal Masked Propagation
        # ========================================

        # Get the paritions of each index
        threshold = self.k
        grouped = {label: [] for label in list(range(self.k+1))}
        for key, value in level_hops.items():
            if value < threshold:
                if value in grouped:
                    grouped[value].append(key)
            else:
                grouped[threshold].append(key)

        # [batch, time, node, node]
        gcn_adj = dense_to_sparse(adj.to(torch.float32))
        
        xh_air_2 = torch.zeros_like(xh_air).to(device=device)
        xh_cro_2 = torch.zeros_like(xh_cro).to(device=device)

        cur_indices_tensor = torch.tensor(grouped[0], dtype=torch.long, device=device)
        cur_ind_exp = cur_indices_tensor[None, None, :, None].expand(b, t, -1, xh_air.size(-1))
        xh_air_2 = xh_air_2.scatter(2, cur_ind_exp, xh_air[:, :, grouped[0], :])

        cro_inds = grouped[0] + list(range(xh_air_2.shape[2], xh_cro_2.shape[2]))
        cur_indices_tensor = torch.tensor(cro_inds, dtype=torch.long, device=device)
        cur_ind_exp = cur_indices_tensor[None, None, :, None].expand(b, t, -1, xh_air.size(-1))
        xh_cro_2 = xh_cro_2.scatter(2, cur_ind_exp, xh_cro[:, :, cro_inds, :])

        rep_indices = []
        cur_indices = []
        for i in range(self.k+1):
            rep_indices += grouped[i]
        for i in range(1, self.k+1):
            cur_indices += grouped[i]

        rep_air = xh_air_2
        rep_cro = xh_cro_2

        air_adj = dense_to_sparse(adj.to(torch.float32))
        cro_adj, _ = dense_to_sparse(c_adj.T.to(torch.float32))
        cro_adj[0] += c_adj.shape[0]

        xh_air_0 = self.gcn_air(rep_air, air_adj[0], air_adj[1])
        xh_air_1 = self.layernorm1(xh_air_0)

        xh_cro_0 = self.gcn_cr(rep_cro, cro_adj)
        xh_cro_1 = self.layernorm1(xh_cro_0)

        cur_indices_tensor = torch.tensor(cur_indices, dtype=torch.long, device=device)
        cur_ind_exp = cur_indices_tensor[None, None, :, None].expand(b, t, -1, xh_air_1.size(-1))

        xh_air_2 = xh_air_2.scatter(2, cur_ind_exp, xh_air_1[:, :, -len(cur_indices):, :])
        xh_cro_2 = xh_cro_2.scatter(2, cur_ind_exp, xh_cro_1[:, :, len(grouped[0]):len(grouped[0])+len(cur_indices), :])
        xh_cro_2 = xh_cro_2[:, :, :xh_air_2.shape[2]]

        # ========================================
        # Final Message Passing
        # ========================================
        xh_air_2 = self.gcn2(xh_air_2, gcn_adj[0], gcn_adj[1]) + xh_air_2
        xh_air_2 = self.layernorm3(xh_air_2)

        xh_cro_2 = self.gcn2(xh_cro_2, gcn_adj[0], gcn_adj[1]) + xh_cro_2
        xh_cro_2 = self.layernorm3(xh_cro_2)

        xh_air_3 = xh_air_2
        xh_cro_3 = xh_cro_2

        finpreds = self.readout1(xh_air_3)

        # ========================================
        # Get cosine similarity for each embedding and get mask 
        # ========================================
        N_a = xh_air_3.shape[2]
        if self.cmd_ratio < 1.0:
            n_air = int(N_a*self.cmd_ratio)
            ar_indx = torch.multinomial(torch.ones(N_a), n_air, replacement=False).to(device)
        else:
            ar_indx = torch.arange(N_a).to(device)
            n_air = N_a
        
        det_mask = torch.zeros_like(xh_air_3).to(dtype=bool, device=device)
        det_mask[:, :, :len(known_set)] = 1
        xh_air_3 = torch.where(det_mask, xh_air_3.detach(), xh_air_3) 

        det_mask = torch.zeros_like(xh_cro_3).to(dtype=bool, device=device)
        det_mask[:, :, :len(known_set)] = 1
        xh_cro_3 = torch.where(det_mask, xh_cro_3.detach(), xh_cro_3) 

        air_nodes = xh_air_3[:, :, ar_indx]
        air_nodes = rearrange(air_nodes, 'b t n d -> t b n d')

        cro_nodes = xh_cro_3[:, :, ar_indx]
        cro_nodes = rearrange(cro_nodes, 'b t n d -> t b n d')

        finsim = [air_nodes, cro_nodes]
        fincross = self.readout1(xh_cro_3)

        if not training:
            return finpreds, fincross
        else:
            return finpreds, fincross, finsim

    def get_new_adj(self, adj, k, n_add, cross_adj, scale=1.0, init_hops={}):
        current_adj = adj.clone()
        current_c_adj = cross_adj.clone()
        t_nodes = cross_adj.shape[1]

        n_current = current_adj.shape[0]
        prev_cur = 0

        if self.sampling == 'partition' or self.sampling == 'half':
            partitions = np.random.exponential(scale, k)
            partitions = partitions / partitions.sum() * n_add
            partitions = np.round(partitions).astype(int)
            partitions[-1] += n_add - partitions.sum()
            partitions = np.sort(partitions)[::-1]
        else:
            partitions = [n_add]

        # Get partitions
        partitions = np.random.exponential(scale, k)
        partitions = partitions / partitions.sum() * n_add
        partitions = np.round(partitions).astype(int)
        partitions[-1] += n_add - partitions.sum()
        partitions = np.sort(partitions)[::-1]

        if partitions[-1] < 0:
            partitions[0] += partitions[-1]
            partitions[-1] = 0

        levels = {i:0 for i in range(n_current+n_add)} | init_hops
        for _, part in enumerate(partitions):
            for _ in range(part):
                n = current_adj.shape[0]

                # Initialize new (n+1)x(n+1) matrix
                expanded = torch.zeros(size=(n + 1, n + 1)).to(device=adj.device, dtype=torch.int)
                expanded[:n, :n] = current_adj

                c_expanded = torch.zeros(size=(n + 1, t_nodes)).to(device=adj.device)
                c_expanded[:n, :] = current_c_adj

                # Select random anchor
                anchor = random.randint(prev_cur, n_current - 1)
                levels[n] = max(levels[anchor] + 1, 1)
                    
                # Connect to anchor
                expanded[anchor, n] = 1
                expanded[n, anchor] = 1

                # Optionally connect to anchor's neighbors
                neigh_list = [anchor]

                if self.sampling == 'random':
                    neighbors = torch.empty(0)
                    tot_n = int(torch.sum(current_adj[anchor, :n_current]).item())

                    if tot_n > 0:
                        neighbors = torch.multinomial(torch.ones(n_current), tot_n, replacement=False)
                        neighbors = neighbors[neighbors != anchor]
                else:
                    neighbors = torch.nonzero(current_adj[anchor, :n_current]).squeeze(-1)

                # print(anchor, neighbors)
                for neighbor in neighbors:
                    connect_prob = np.random.rand(1)
                    if np.random.rand(1) < connect_prob and levels[n] >= levels[neighbor.item()]:
                        expanded[neighbor, n] = 1.
                        expanded[n, neighbor] = 1.

                        if levels[n] > levels[neighbor.item()]:
                            levels[n] = max(levels[neighbor.item()] + 1, 1)

                        neigh_list.append(neighbor.item())

                # Update current_adj and current_c_adj
                c_expanded[n, :] = current_c_adj[neigh_list, :].sum(0).ne(0).float()

                current_adj = expanded
                current_c_adj = c_expanded
            prev_cur = n_current
            n_current += part

        # Create matrices for both n1 and n2
        adj_aug_n1 = torch.rand((current_adj.shape)).to(device = adj.device)  # n2, n2
        adj_aug_n1 = 0.9*adj_aug_n1 + 0.1
        adj_aug_n1 = torch.triu(adj_aug_n1) + torch.triu(adj_aug_n1, 1).T

        # preserve original observed parts in newly-created adj
        adj_aug_n1 = adj_aug_n1.fill_diagonal_(0)
        adj_aug_n1 *= current_adj
        adj_aug_n1[:adj.shape[0], :adj.shape[0]] = adj

        if adj_aug_n1.shape[0] > adj.shape[0] + n_add:
            print('error')
        
        if self.full_cadj:
            new_c_adj = torch.ones((adj_aug_n1.shape[0], t_nodes)).to(device=adj.device)
        else:
            new_c_adj = current_c_adj.to(device=adj.device)

        return adj_aug_n1, levels, new_c_adj