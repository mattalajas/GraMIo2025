import math
import os
from functools import partial
from typing import (Iterable, List, Literal, Mapping, Optional, Sequence,
                    Tuple, Type, Union)

import networkx as nx
import numpy as np
import pandas as pd
import torch
import tsl
from numpy import ndarray
from pytorch_lightning import LightningDataModule
from torch import Generator, Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data.storage import recursive_apply
from torch_geometric.utils import dense_to_sparse
from torch_scatter import scatter
from tqdm import tqdm
from tsl.data import ImputationDataset
from tsl.data.datamodule.splitters import Splitter, disjoint_months
from tsl.data.loader import StaticGraphLoader
from tsl.data.preprocessing import Scaler, ScalerModule
from tsl.data.spatiotemporal_dataset import SpatioTemporalDataset
from tsl.data.synch_mode import HORIZON
from tsl.datasets import TabularDataset
from tsl.datasets.prototypes import DatetimeDataset, TabularDataset
from tsl.datasets.prototypes.casting import to_pandas_freq
from tsl.metrics import numpy as numpy_metrics
from tsl.nn.layers.graph_convs.gpvar import GraphPolyVAR
from tsl.ops.connectivity import parse_connectivity
from tsl.ops.imputation import to_missing_values_dataset
from tsl.ops.pattern import broadcast, outer_pattern, take
from tsl.typing import Index, SparseTensArray, TensArray
from tsl.utils.casting import torch_to_numpy
from tsl.utils.python_utils import foo_signature

StageOptions = Literal['fit', 'validate', 'test', 'predict']


def zeros_to_one_(scale):
    """Set to 1 scales of near constant features, detected by identifying
    scales close to machine precision, in place.
    Adapted from :class:`sklearn.preprocessing._data._handle_zeros_in_scale`
    """
    if np.isscalar(scale):
        return 1.0 if np.isclose(scale, 0.) else scale
    eps = 10 * np.finfo(scale.dtype).eps
    zeros = np.isclose(scale, 0., atol=eps, rtol=eps)
    scale[zeros] = 1.0
    return scale


def fit_wrapper(fit_function):

    def fit(obj: "Scaler", x, *args, **kwargs) -> "Scaler":
        x_type = type(x)
        x = np.asarray(x)
        fit_function(obj, x, *args, **kwargs)
        if x_type is Tensor:
            obj.torch()
        return obj

    return fit

def closest_distances_unweighted(G, source_nodes, target_nodes):
    result = {}
    target_set = set(target_nodes)
    
    for source in source_nodes:
        lengths = nx.single_source_shortest_path_length(G, source)
        distances = [lengths[t] for t in target_set if t in lengths]
        result[source] = min(distances) if distances else float('inf')
    
    return result

def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    sum_of_diff_square = ((x1-x2)**2).sum(-1) + 1e-8
    return sum_of_diff_square.sqrt()

def moment_diff(sx1, sx2, k, og_batch, coarse_batch):
    """
    difference between moments
    """
    ss1 = scatter(sx1**k, og_batch, dim=0, dim_size=None, reduce='mean')
    ss2 = scatter(sx2**k, coarse_batch, dim=0, dim_size=None, reduce='mean')
    return l2diff(ss1,ss2)

def cmd(x1, x2, og_batch, coarse_batch, n_moments=2):
    """
    central moment discrepancy (cmd)
    - Zellinger, Werner et al. "Robust unsupervised domain adaptation
    for neural networks via moment alignment," arXiv preprint arXiv:1711.06114,
    2017.
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    #print("input shapes", x1.shape, x2.shape)
    mx1 = scatter(x1, og_batch, dim=0, dim_size=None, reduce='mean')
    mx2 = scatter(x2, coarse_batch, dim=0, dim_size=None, reduce='mean')
    #print("mx* shapes should be same (batch_szie, dim)", mx1.shape, mx2.shape)
    sx1 = x1 - mx1.repeat_interleave(torch.unique(og_batch, return_counts=True)[1], dim=0)
    sx2 = x2 - mx2.repeat_interleave(torch.unique(coarse_batch, return_counts=True)[1], dim=0)
    #print("sx1, sx2 should be same size as input", sx1.shape, sx2.shape)
    dm = l2diff(mx1, mx2)
    #print("dm should have shape (batch_size,)", dm.shape)
    scms = dm
    for i in range(n_moments-1):
        # moment diff of centralized samples
        scms = scms + moment_diff(sx1, sx2, i+2, og_batch, coarse_batch)
    return scms

def get_agg_feature_distance_community(adj_matrix,
                                       feature,
                                       train_idx,
                                       test_idx,
                                       group_num=5):
    
    adj_matrix = torch.from_numpy(adj_matrix).to_sparse()
    num_nodes = adj_matrix.shape[0]

    feature = feature.reshape(feature.shape[0]*feature.shape[1], feature.shape[2]).T

    A = adj_matrix  # torch_sparse.tensor.SparseTensor
    # print("complete A")
    A = A + torch.sparse_coo_tensor(
        [[i for i in range(num_nodes)], [i for i in range(num_nodes)]], [1] * num_nodes)
    # print("complete A+I")
    D_diag = list(torch.sparse.sum(A, dim=1))
    # print("complete D_diag")
    D_1 = [1 / x for x in D_diag]
    D_1 = torch.sparse_coo_tensor(
        [[i for i in range(num_nodes)], [i for i in range(num_nodes)]], D_1)
    # print("complete D_1")

    agg = torch.sparse.mm(D_1, A)
    # print("complete mm1")
    agg = torch.sparse.mm(agg, D_1)
    # print("complete mm2")
    agg = torch.sparse.mm(agg, A).to_dense().numpy()
    # print("complete mm3")
    agg = np.matmul(agg, feature)
    # print("complete mm4")

    agg_distance = {}
    train_idx = list(train_idx)
    for k in range(len(test_idx)):
        i = test_idx[k]
        # if k % 100 == 0:
        #     print(k)
        agg_distance[i] = float('inf')
        for j in train_idx:
            agg_distance[i] = min(agg_distance[i],
                                  np.linalg.norm(agg[i] - agg[j]))

    sort_res = list(
        map(lambda x: x[0], sorted(agg_distance.items(), key=lambda x: x[1])))
    node_num_group = len(sort_res) // group_num
    return [
        sort_res[i:i + node_num_group + 1]
        for i in range(0, len(sort_res), node_num_group + 1)
    ]

def test_wise_eval(y_hat, y_true, mask, known_nodes, adj, mode, num_groups=4, alpha = 0.20, features=None):
    try:
        adj = adj.numpy()
    except:
        pass

    numpy_graph = nx.from_numpy_array(adj)
    k_nodes = np.array(known_nodes)
    u_nodes = np.array([i for i in range(adj.shape[0]) if i not in known_nodes])
    m_adj = (adj > 0).astype(float)
    group_size = u_nodes.shape[0] // num_groups

    # LPS
    n = adj.shape[-1]

    A_hat = m_adj
    idx = np.arange(n)
    A_hat[idx, idx] = 1
    D = np.diag(np.sum(A_hat, axis=1))

    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    I = np.eye(n)

    P = np.linalg.inv((I - (1-alpha)*A_norm))
    T = np.zeros((n, n))
    T[k_nodes, k_nodes] = 1
    ones = np.ones((n,))

    LPS = P @ T @ ones

    sorted_lps = sorted(u_nodes, key=lambda i: LPS[i])
    lps_gr = [sorted_lps[i*group_size : (i+1)*group_size] for i in range(num_groups)]
    remainder = len(sorted_lps) % num_groups
    if remainder:
        lps_gr[-1].extend(sorted_lps[-remainder:])

    # AGG
    agg_gr = get_agg_feature_distance_community(adj, features, k_nodes, u_nodes, group_num=num_groups)

    # CC
    closeness = nx.closeness_centrality(numpy_graph)
    closeness = {node: score for node, score in closeness.items() if score > 0}

    sorted_cls = sorted([i for i in u_nodes if i in closeness], key=lambda i: closeness[i])
    # sorted_cls = [x for x, _ in sorted_cls]
    cls_gr = [sorted_cls[i*group_size : (i+1)*group_size] for i in range(num_groups)]
    remainder = len(sorted_cls) % num_groups
    if remainder:
        cls_gr[-1].extend(sorted_cls[-remainder:])

    # KHR
    khr_grouped = closest_distances_unweighted(numpy_graph, u_nodes, k_nodes.tolist())
    khr_grouped = {node: score for node, score in khr_grouped.items() if score < 1e9}
    khr_gr = [[] for _ in range(num_groups)]

    for key, pos in khr_grouped.items():
        value = pos-1
        if value < num_groups:
            khr_gr[value].append(key)
        else:
            khr_gr[num_groups-1].append(key)

    # Evaluate
    if features is not None:
        group_dict = {'LPS': lps_gr,
                    'CC': cls_gr,
                    'KHR': khr_gr,
                    'AGG': agg_gr}
    else:
        group_dict = {'LPS': lps_gr,
                    'CC': cls_gr,
                    'KHR': khr_gr}
        
    res = {f'{mode}_mae': numpy_metrics.mae(y_hat, y_true, mask),
               f'{mode}_mre': numpy_metrics.mre(y_hat, y_true, mask),
               f'{mode}_rmse': numpy_metrics.rmse(y_hat, y_true, mask)}

    for key, groups in group_dict.items():
        results = {'mae':[], 'mre':[], 'rmse':[]}
        for group in groups:
            node_mask = np.zeros_like(mask, dtype=bool)
            if len(node_mask.shape) == 4:
                node_mask[:, :, group] = True
            elif len(node_mask.shape) == 3:
                node_mask[:, group] = True
            else:
                raise 'node_mask dim only 3 or 2'
            
            masked_adj = mask * node_mask

            results['mae'].append(numpy_metrics.mae(y_hat, y_true, masked_adj))
            results['mre'].append(numpy_metrics.mre(y_hat, y_true, masked_adj))
            results['rmse'].append(numpy_metrics.rmse(y_hat, y_true, masked_adj))

        for metric, val in results.items():
            res[f'max_{metric}_{key}_{mode}'] = max(val)
            res[f'min_{metric}_{key}_{mode}'] = min(val)
    
        if num_groups == 5:
            res[f'all_mae_{key}_{mode}'] = results['mae']
            res[f'all_mre_{key}_{mode}'] = results['mre']
            res[f'all_rmse_{key}_{mode}'] = results['rmse']
    
    return res    

def add_missing_sensors(dataset: TabularDataset,
                       p_noise=0.05,
                       p_fault=0.01,
                       min_seq=1,
                       max_seq=10,
                       seed=None,
                       inplace=True,
                       masked_sensors = [],
                       connect = None,
                       spatial_shift = False, 
                       order = 0,
                       node_features = 'CC',
                       mode='road'):
    if seed is None:
        seed = np.random.randint(1e9)
    # Fix seed for random mask generation
    random = np.random.default_rng(seed)

    # Compute evaluation mask
    shape = (dataset.length, dataset.n_nodes, dataset.n_channels)
    if masked_sensors is None:
        if spatial_shift:
            eval_mask = shift_mask(shape, feature=node_features, order=order, 
                                   adj=dataset.get_connectivity(**connect, layout='dense'),
                                   p_noise=p_noise)
            dataset.seed = seed
        else:
            eval_mask = sample_mask(shape,
                                p=p_fault,
                                p_noise=p_noise,
                                mode=mode,
                                adj=dataset.get_connectivity(**connect, layout='dense'))
            
            dataset.p_fault = p_fault
            dataset.p_noise = p_noise
            dataset.min_seq = min_seq
            dataset.max_seq = max_seq
            dataset.seed = seed
            dataset.random = random

        # mask = rearrange(eval_mask, "b n 1 -> b n")
        mask_sum = eval_mask.sum(0)  # n
        masked_sensors = (np.where(mask_sum > 0)[0]).tolist()
    else:
        masked_sensors = list(masked_sensors)
        eval_mask = np.zeros_like(dataset.mask)
        eval_mask[:, masked_sensors] = dataset.mask[:, masked_sensors]

    # Convert to missing values dataset
    dataset = to_missing_values_dataset(dataset, eval_mask, inplace)

    # Store evaluation mask params in dataset
    return dataset, masked_sensors

def shift_mask(shape, feature, order, adj, p_noise=0.05):
    mask = np.zeros(shape).astype(bool)
    
    try:
        adj = adj.numpy()
    except:
        pass

    G = nx.from_numpy_array(adj)
    parts = math.ceil(adj.shape[0]*p_noise)

    if feature == 'CC':
        # Compute closeness centrality
        closeness = nx.closeness_centrality(G)
        nonzero_c = {node: score for node, score in closeness.items() if score > 0}

        # Sort nodes by closeness centrality in descending order
        sorted_nodes = sorted(nonzero_c.items(), key=lambda x: x[1])
        ord_nodes = [x for x, _ in sorted_nodes]

        f_nodes = ord_nodes[parts*order:parts*(order+1)]
        f_nodes_mask = np.zeros(shape).astype(bool)
        f_nodes_mask[:, f_nodes] = True
        mask |= f_nodes_mask
        
    elif feature == 'ND':
        degree = dict(nx.degree(G))
        nonzero_d = {node: score for node, score in degree.items() if score > 0}

        # Sort nodes by node degree in descending order
        sorted_nodes = sorted(nonzero_d.items(), key=lambda x: x[1])
        ord_nodes = [x for x, _ in sorted_nodes]

        f_nodes = ord_nodes[parts*order:parts*(order+1)]
        f_nodes_mask = np.zeros(shape).astype(bool)
        f_nodes_mask[:, f_nodes] = True
        mask |= f_nodes_mask
    else:
        raise f"{feature} not implemented"
    
    return mask.astype('uint8')

def sample_mask(shape, p=0.002, p_noise=0., mode="random", adj=None):
    assert mode in ["random", "road", "mix"], "The missing mode must be 'random' or 'road' or 'mix'."
    rand = np.random.random
    mask = np.zeros(shape).astype(bool)
    if mode == "random" or mode == "mix":
        mask = mask | (rand(mask.shape) < p)
    if mode == "road" or mode == "mix":
        road_shape = mask.shape[1]
        rand_mask = rand(road_shape) < p_noise
        road_mask = np.zeros(shape).astype(bool)
        road_mask[:, rand_mask] = True
        mask |= road_mask
    return mask.astype('uint8')

class TrafAirSplitter(Splitter):
    def __init__(self,
                 val_len: int = None,
                 test_months: Sequence = (3, 6, 9, 12)):
        super(TrafAirSplitter, self).__init__()
        self._val_len = val_len
        self.test_months = test_months

    def fit(self, dataset):
        nontest_idxs, test_idxs = disjoint_months(dataset,
                                                  months=self.test_months,
                                                  synch_mode=HORIZON)
        # take equal number of samples before each month of testing
        val_len = self._val_len
        if val_len < 1:
            val_len = int(val_len * len(nontest_idxs))
        val_len = val_len // len(self.test_months)
        # get indices of first day of each testing month
        delta = np.diff(test_idxs)
        delta_idxs = np.flatnonzero(delta > delta.min())
        end_month_idxs = test_idxs[1:][delta_idxs]
        if len(end_month_idxs) < len(self.test_months):
            end_month_idxs = np.insert(end_month_idxs, 0, test_idxs[0])
        # expand month indices
        month_val_idxs = [
            np.arange(v_idx - val_len, v_idx) - dataset.window
            for v_idx in end_month_idxs
        ]
        val_idxs = np.concatenate(month_val_idxs) % len(dataset)
        # remove overlapping indices from training set
        ovl_idxs, _ = dataset.overlapping_indices(nontest_idxs,
                                                  val_idxs,
                                                  synch_mode=HORIZON,
                                                  as_mask=True)
        train_idxs = nontest_idxs[~ovl_idxs]
        self.set_indices(train_idxs, val_idxs, test_idxs)

class AirCross(DatetimeDataset):
    similarity_options = {"precomputed"}

    def __init__(self,
                 root: str = None,
                 test_months: Sequence = (3, 6, 9, 12),
                 years: Sequence = (),
                 imputation_mode: Literal["nearest", "zero", None] = "zero",
                 freq: str = "h",
                 include_exog: bool = False,
                 exog: str = 'humd'):
        # set root path
        self.root = root
        self.years = years
        self.imputation_mode = imputation_mode
        self.test_months = test_months
        self.include_exog = include_exog
        self.exog = exog

        assert imputation_mode in ["nearest", "zero", None]
        assert exog in ['traffic', 'temp', 'humd', None]

        # Set dataset frequency here to resample when loading
        if freq is not None:
            freq = to_pandas_freq(freq)
        self.freq = freq

        # load dataset
        readings, mask, adj, air_metadata, tra_metadata, modality = self.load()
        self.tra_metadata = tra_metadata
        self.air_metadata = air_metadata
        self.modality = modality
        covariates = {"adj": (adj, 'n n')}
        
        super().__init__(target=readings,
                         freq=freq,
                         mask=mask,
                         covariates=covariates,
                         similarity_score="precomputed",
                         temporal_aggregation="mean",
                         spatial_aggregation="mean",
                         default_splitting_method='trafair',
                         name='AirCross')

    def load_raw(self):
        # load sensors information
        air_metadata = pd.read_csv(os.path.join(self.root_dir, 'air_metadata.csv'))
        self.air_max_nodes = len(air_metadata)

        readings = pd.read_csv(os.path.join(self.root_dir, f'full_data_{self.exog}.csv'), index_col=0, parse_dates=['Time'])
        if len(self.years) != 0:
            readings = readings[readings.index.year.isin(self.years)]

        modality = np.zeros((len(readings.columns), 1))
        modality[self.air_max_nodes:] = 1

        if not self.include_exog:
            readings = readings.iloc[:, :self.air_max_nodes]
            modality = modality[:self.air_max_nodes]

        # resample here to aggregate only valid observations and
        # align to authors' preprocessing
        if self.freq is not None:
            readings = readings.apply(pd.to_numeric, errors='coerce')
            readings = readings.resample(self.freq).mean()

        # load adjacency
        ar_edge_index, ar_edge_weight = np.load(os.path.join(self.root_dir, 'air_adj.npz')).values()
        ar_adj = np.eye(self.air_max_nodes, dtype=np.float32)
        ar_adj[tuple(ar_edge_index)] = ar_edge_weight

        # Get adj for exogenous modality 
        tra_metadata = pd.DataFrame()
        if self.include_exog:
            tra_metadata = pd.read_csv(os.path.join(self.root_dir, f'{self.exog}_metadata.csv'))
            self.tra_max_nodes = len(tra_metadata)

            tr_edge_index, tr_edge_weight = np.load(os.path.join(self.root_dir, f'{self.exog}_adj.npz')).values()
            cr_edge_index, cr_edge_weight = np.load(os.path.join(self.root_dir, f'cross_adj_{self.exog}.npz')).values()
            # build square adj from coo to add adj as covariate

            tr_adj = np.eye(self.tra_max_nodes, dtype=np.float32)
            tr_adj[tuple(tr_edge_index)] = tr_edge_weight

            cr_adj = np.zeros((self.air_max_nodes, self.tra_max_nodes), dtype=np.float32)
            cr_adj[tuple(cr_edge_index)] = cr_edge_weight

            # cross_adj = pd.read_csv(os.path.join(self.root_dir, f'cross_dist_{self.exog}.csv'))
            adj = np.block([
                [ar_adj,  cr_adj],
                [cr_adj.T, tr_adj]
            ])
        else:
            adj = ar_adj

        return readings, adj, air_metadata, tra_metadata, modality
    
    def get_splitter(self, method: Optional[str] = None, **kwargs):
        if method == 'trafair':
            val_len = kwargs.get('val_len')
            return TrafAirSplitter(test_months=self.test_months,
                                    val_len=val_len)

    def load(self):
        readings, adj, air_metadata, tra_metadata, modality = self.load_raw()
        # impute missing observations using last observed values
        # in authors' code: readings = readings.fillna(0)
        mask = ~readings.isna().values
        if self.imputation_mode == "nearest":
            readings = readings.ffill().bfill()
        elif self.imputation_mode == "zero":
            readings = readings.fillna(0)
        return readings, mask, adj, air_metadata, tra_metadata, modality

    def compute_similarity(self, method: str, **kwargs):
        if method == "precomputed":
            # load precomputed adjacency matrix based on road distance
            return self.adj

class StandardScalerSplit(Scaler):
    """Apply standardization to data by removing mean and scaling to unit
    variance.

    Args:
        axis (int): dimensions of input to fit parameters on.
            (default: 0)
    """

    def __init__(self, split: int, axis: Union[int, Tuple] = 0):
        super(StandardScalerSplit, self).__init__()
        self.axis = axis
        self.split = split

    @fit_wrapper
    def fit(self, x: TensArray, mask=None, keepdims=True):
        r"""Fit scaler's parameters `bias` :math:`\mu` and `scale`
        :math:`\sigma` as the mean and the standard deviation of :obj:`x`,
        respectively.

        Args:
            x: array-like input
            mask (optional): boolean mask to denote elements of :obj:`x` on
                which to fit the parameters.
                (default: :obj:`None`)
            keepdims (bool): whether to keep the same dimensions as :obj:`x` in
                the parameters.
                (default: :obj:`True`)
        """
        if mask is not None:
            x = np.where(mask, x, np.nan)
            t, n, f = x.shape

            first_half = x[:, :self.split, :] 
            second_half = x[:, self.split:, :]  

            first = np.nanmean(first_half.astype(np.float32),
                                axis=self.axis,
                                keepdims=keepdims).astype(x.dtype)
            second = np.nanmean(second_half.astype(np.float32),
                                axis=self.axis,
                                keepdims=keepdims).astype(x.dtype)

            filled_first = np.tile(first, (1, self.split, 1))
            filled_second = np.tile(second, (1, n - self.split, 1))

            self.bias = np.concatenate([filled_first, filled_second], axis=1)

            first = np.nanstd(first_half.astype(np.float32),
                                axis=self.axis,
                                keepdims=keepdims).astype(x.dtype)
            second = np.nanstd(second_half.astype(np.float32),
                                axis=self.axis,
                                keepdims=keepdims).astype(x.dtype)

            filled_first = np.tile(first, (1, self.split, 1))
            filled_second = np.tile(second, (1, n - self.split, 1))

            self.scale = np.concatenate([filled_first, filled_second], axis=1)
        else:
            t, n, f = x.shape

            first_half = x[:, :self.split, :] 
            second_half = x[:, self.split:, :]  

            first = first_half.mean(axis=(0, 1), keepdims=True)  # (1, 1, f)
            second = second_half.mean(axis=(0, 1), keepdims=True)  # (1, 1, f)

            filled_first = np.tile(first, (1, self.split, 1))
            filled_second = np.tile(second, (1, n - self.split, 1))
            
            self.bias = np.concatenate([filled_first, filled_second], axis=1)

            first = first_half.std(axis=(0, 1), keepdims=True)  # (1, 1, f)
            second = second_half.std(axis=(0, 1), keepdims=True)  # (1, 1, f)

            filled_first = np.tile(first, (1, self.split, 1))
            filled_second = np.tile(second, (1, n - self.split, 1))

            self.scale = np.concatenate([filled_first, filled_second], axis=1)
        self.scale = zeros_to_one_(self.scale)
        return self

    def transform(self, x: TensArray):
        r"""Apply transformation :math:`f(x) = (x - \mu) / \sigma`."""
        return (x - self.bias) / (self.scale + tsl.epsilon)

    def inverse_transform(self, x: TensArray):
        r"""Apply inverse transformation
        :math:`f(x) = (x \cdot \sigma) + \mu`."""
        return x * (self.scale + tsl.epsilon) + self.bias

    def fit_transform(self, x: TensArray, *args, **kwargs):
        r"""Fit scaler's parameters using input :obj:`x` and then transform
        :obj:`x`."""
        self.fit(x, *args, **kwargs)
        return self.transform(x)
    
class ScalerSplitModule(ScalerModule):
    def __init__(self,
                 scaler: Optional[Union["Scaler", "ScalerModule"]] = None,
                 *,
                 bias: Union[Tensor, float] = 0.,
                 scale: Union[Tensor, float] = 1.,
                 pattern: Optional[str] = None):
        super().__init__(scaler, bias=bias, scale=scale, pattern=pattern)
        self.bias_list = torch.unique(scaler.bias)
        self.scale_list = torch.unique(scaler.scale)

    def _get_name(self):
        return self.__class__.__name__

    def transform_tensor(self, x: Tensor, split = None) -> Tensor:
        if split:
            temp_bias = torch.zeros_like(x).to(x.device)
            temp_bias[:, :, :split] = self.bias_list[0]
            temp_bias[:, :, split:] = self.bias_list[1]

            temp_scale = torch.zeros_like(x).to(x.device)
            temp_scale[:, :, :split] = self.scale_list[0]
            temp_scale[:, :, split:] = self.scale_list[1]

            return (x - temp_bias) / temp_scale + tsl.epsilon
        else:
            return (x - self.bias) / self.scale + tsl.epsilon

    def inverse_transform_tensor(self, x: Tensor, split = None) -> Tensor:
        if split:
            temp_bias = torch.zeros_like(x).to(x.device)
            temp_bias[:, :, :split] = self.bias_list[0]
            temp_bias[:, :, split:] = self.bias_list[1]

            temp_scale = torch.zeros_like(x).to(x.device)
            temp_scale[:, :, :split] = self.scale_list[0]
            temp_scale[:, :, split:] = self.scale_list[1]

            return x * (temp_scale + tsl.epsilon) + temp_bias
        else:
            return x * (self.scale + tsl.epsilon) + self.bias

    def transform(self, x, split=None):
        split_trans_tensor = partial(self.transform_tensor, split=split)
        return recursive_apply(x, split_trans_tensor)

    def inverse_transform(self, x, split=None):
        split_invtr_tensor = partial(self.inverse_transform_tensor, split=split)
        return recursive_apply(x, split_invtr_tensor)
    
    def slice(self,
              time_index: Union[List, Tensor] = None,
              node_index: Union[List, Tensor] = None):
        if self.pattern is None:
            raise RuntimeError("You are trying to slice a scaler with no "
                               "pattern.")
        # move to new object
        scaler = ScalerSplitModule(self)
        # shortcut for when scaler is time-unvarying and node_index is None
        if time_index is None and node_index is None:
            return scaler

        # if time-unvarying scaler, just apply unsqueezing indexing
        new_axes, pattern = None, scaler.pattern
        if time_index is not None and time_index.ndim == 2:
            new_axes = torch.zeros(1, 1, dtype=torch.long)
            pattern = 'b ' + scaler.pattern

        # compute actual slicing for each param
        t, n = self.t_axis, self.n_axis  # axis of time and node dimensions
        ti_bias = ti_scale = time_index
        ni_bias = ni_scale = node_index
        if self.t_axis is not None:
            ti_bias = time_index if self.bias.size(t) > 1 else new_axes
            ti_scale = time_index if self.scale.size(t) > 1 else new_axes
        if self.n_axis is not None:
            ni_bias = node_index if self.bias.size(n) > 1 else None
            ni_scale = node_index if self.scale.size(n) > 1 else None

        # slice params
        scaler.bias = take(scaler.bias,
                           self.pattern,
                           time_index=ti_bias,
                           node_index=ni_bias)
        scaler.scale = take(scaler.scale,
                            self.pattern,
                            time_index=ti_scale,
                            node_index=ni_scale)
        # update pattern
        scaler.pattern = pattern

        return scaler

    # you can also override other methods if needed

class CrossSpatioTemporalDataset(ImputationDataset):
    def __init__(self,
                 target,
                 eval_mask,
                 index = None,
                 mask = None,
                 connectivity  = None,
                 covariates = None,
                 input_map = None,
                 target_map = None,
                 auxiliary_map = None,
                 scalers = None,
                 trend = None,
                 transform = None,
                 window: int = 12,
                 stride: int = 1,
                 window_lag: int = 1,
                 precision: Union[int, str] = 32,
                 name: Optional[str] = None):
        # call parent constructor
        super().__init__(target=target,
                         eval_mask=eval_mask,
                         index=index,
                         mask=mask,
                         connectivity=connectivity,
                         covariates=covariates,
                         input_map=input_map,
                         target_map=target_map,
                         auxiliary_map=auxiliary_map,
                         scalers=scalers,
                         trend=trend,
                         transform=transform,
                         window=window,
                         stride=stride,
                         window_lag=window_lag,
                         precision=precision,
                         name=name)

    def expand_scaler(self, key: str, pattern: Optional[str] = None,
                      time_index: Union[List, Tensor] = None,
                      node_index: Union[List, Tensor] = None) \
            -> Optional[ScalerSplitModule]:
        # check if there is a scaler
        if key not in self.keys:
            raise KeyError(f"{key} not in {self.name}.")
        elif key not in self.scalers:
            return None
        # convert indices
        time_index = self._get_time_index(time_index, layout='index')
        node_index = self._get_time_index(node_index, layout='index')
        # get params
        if pattern is None:
            return self.scalers[key]
        # if there is an out-pattern, create new scaler
        scaler = ScalerSplitModule(self.scalers[key], pattern=pattern)
        pattern = self.patterns[key] + ' -> ' + pattern
        scaler.bias = broadcast(scaler.bias,
                                pattern,
                                backend=torch,
                                time_index=time_index,
                                node_index=node_index)
        scaler.scale = broadcast(scaler.scale,
                                 pattern,
                                 backend=torch,
                                 time_index=time_index,
                                 node_index=node_index)
        return scaler

    def get_tensor(self, key: str, preprocess: bool = False,
                   time_index: Union[List, Tensor] = None,
                   node_index: Union[List, Tensor] = None) \
            -> Tuple[Tensor, Optional[ScalerSplitModule]]:
        # get dataset item
        if key not in self.keys:
            raise KeyError(f"{key} not in dataset {self.name}.")

        # convert indices
        time_index = self._get_time_index(time_index, layout='index')
        node_index = self._get_time_index(node_index, layout='index')
        x = take(getattr(self, key),
                 self.patterns[key],
                 backend=torch,
                 time_index=time_index,
                 node_index=node_index)
        try:
            test = x[:, :, :23, :]
        except:
            pass

        # get scaler (if any)
        scaler = None
        if key in self.scalers is not None:
            scaler = self.scalers[key].slice(time_index=time_index,
                                             node_index=node_index)
            if preprocess:  # transform tensor
                x = scaler.transform(x)
        return x, scaler

    def collate_item_elem(self, key: str,
                          time_index: Union[List, Tensor] = None,
                          node_index: Union[List, Tensor] = None) \
            -> Tuple[Tensor, Optional[ScalerSplitModule]]:
        # get batch item
        if key in self.input_map:
            itm = self.input_map[key]
        elif key in self.target_map:
            itm = self.target_map[key]
        else:
            raise KeyError(f"{key} not in any batch map of {self.name}.")

        # expand and concatenate tensors
        x = torch.cat([
            self.expand_tensor(k, itm.pattern, time_index, node_index)
            for k in itm.keys
        ],
                      dim=itm.cat_dim)

        # get scaler (if any)
        scaler = None
        if key in self._batch_scalers:
            scaler = self._batch_scalers[key].slice(time_index=time_index,
                                                    node_index=node_index)
            if itm.preprocess:  # transform tensor
                x = scaler.transform(x)
        return x, scaler

    def collate_keys(self,
                     keys: Iterable,
                     preprocess: bool = False,
                     time_index: Union[List, Tensor] = None,
                     node_index: Union[List, Tensor] = None,
                     cat_dim: Optional[int] = None,
                     return_pattern: bool = False):
        if any([key not in self.keys for key in keys]):
            unmatch = set(keys).difference(self.keys)
            raise KeyError(f"{unmatch} not in {self.name}.")
        pattern = outer_pattern([self.patterns[key] for key in keys])
        tensors, scalers = list(), list()
        for key in keys:
            tensor = self.expand_tensor(key, pattern, time_index, node_index)
            scaler = self.expand_scaler(key, pattern, time_index, node_index)
            if preprocess and scaler is not None:
                tensor = scaler(tensor)
            tensors.append(tensor)
            scalers.append(scaler)
        if len(tensors) == 1:
            if return_pattern:
                return tensors[0], scalers[0], pattern
            return tensors[0], scalers[0]
        if cat_dim is not None:
            scalers = ScalerSplitModule.cat(scalers,
                                       dim=cat_dim,
                                       sizes=[t.size() for t in tensors])
            tensors = torch.cat(tensors, dim=cat_dim)
        if return_pattern:
            return tensors, scalers, pattern
        return tensors, scalers

    def get_mask(self, dtype: Union[type, str, np.dtype] = None) -> Tensor:
        mask = self.mask if self.has_mask else ~torch.isnan(self.target)
        if dtype is not None:
            assert dtype in ['bool', 'uint8', bool, torch.bool, torch.uint8]
            mask = mask.to(dtype)
        return mask

    def add_scaler(self, key: str, scaler: Union[Scaler, ScalerSplitModule]):
        r"""Add a :class:`tsl.data.preprocessing.Scaler` for the object indexed
        by :obj:`key` in the dataset.

        Args:
            key (str): The name of the variable associated to the scaler. It
                must be a temporal variable, i.e., :obj:`data` or an exogenous.
            scaler (Scaler): The :class:`~tsl.data.preprocessing.Scaler`.
        """
        if key not in self.keys:
            raise KeyError(f"{key} not in {self.name}.")
        # copy to ScalerModule
        scaler = ScalerSplitModule(scaler)
        pattern = self.patterns[key]
        self._check_pattern(scaler.bias,
                            pattern,
                            name=f"scaler ({key})",
                            allow_broadcasting=True)
        self._check_pattern(scaler.scale,
                            pattern,
                            name=f"scaler ({key})",
                            allow_broadcasting=True)
        if key == 'target' and self.trend is not None:
            self.__target_bias = scaler.bias
            scaler.bias = scaler.bias + self.trend
        scaler.pattern = pattern
        self.scalers[key] = scaler
        # cache batch scaler if target tensor is in a multi-key batch item
        for bm in [self.input_map, self.target_map, self.auxiliary_map]:
            for bm_key, bm_item in bm.items():
                if key in bm_item.keys and len(bm_item.keys) > 1:
                    tensor, scaler = self.collate_keys(bm_item.keys,
                                                       cat_dim=bm_item.cat_dim,
                                                       return_pattern=False)
                    self._batch_scalers[bm_key] = scaler


class SpatioTemporalDataModule(LightningDataModule):
    r"""Base :class:`~pytorch_lightning.core.LightningDataModule` for
    :class:`~tsl.data.SpatioTemporalDataset`.

    Args:
        dataset (SpatioTemporalDataset): The complete dataset.
        scalers (dict, optional): Named mapping of
            :class:`~tsl.data.preprocessing.scalers.Scaler`
            to be used for data rescaling after splitting. Every scaler is given
            as input the attribute of the dataset named as the scaler's key.
            If :obj:`None`, no scaling is performed.
            (default :obj:`None`)
        mask_scaling (bool): If :obj:`True`, then compute statistics for
            :obj:`dataset.target` scaler (if any) by considering only valid
            values (according to :obj:`dataset.mask`).
            (default :obj:`True`)
        splitter (Splitter, optional): The
            :class:`~tsl.data.datamodule.splitters.Splitter` to be used for
            splitting :obj:`dataset` into train/validation/test sets.
            (default :obj:`None`)
        batch_size (int): Size of the mini-batches for the dataloaders.
            (default :obj:`32`)
        workers (int): Number of workers to use in the dataloaders.
            (default :obj:`0`)
        pin_memory (bool): If :obj:`True`, then enable pinned GPU memory for
            :meth:`~tsl.data.datamodule.SpatioTemporalDataModule.train_dataloader`.
            (default :obj:`False`)
    """

    def __init__(self,
                 dataset: SpatioTemporalDataset,
                 scalers: Optional[Mapping] = None,
                 mask_scaling: bool = True,
                 splitter: Optional[Splitter] = None,
                 batch_size: int = 32,
                 workers: int = 0,
                 pin_memory: bool = False,
                 generator: Generator = None):
        super(SpatioTemporalDataModule, self).__init__()
        self.torch_dataset = dataset
        # splitting
        self.splitter = splitter
        self.trainset = self.valset = self.testset = None
        self.generator = generator
        # scaling
        if scalers is None:
            self.scalers = dict()
        else:
            self.scalers = scalers
        self.mask_scaling = mask_scaling
        # data loaders
        self.batch_size = batch_size
        self.workers = workers
        self.pin_memory = pin_memory

    def __getattr__(self, item):
        ds = self.__dict__.get('torch_dataset')
        if ds is not None and hasattr(ds, item):
            return getattr(ds, item)
        else:
            raise AttributeError(item)

    def __repr__(self):
        return "{}(train_len={}, val_len={}, test_len={}, " \
               "scalers=[{}], batch_size={})" \
            .format(self.__class__.__name__,
                    self.train_len, self.val_len, self.test_len,
                    ', '.join(self.scalers.keys()), self.batch_size)

    @property
    def trainset(self):
        return self._trainset

    @property
    def valset(self):
        return self._valset

    @property
    def testset(self):
        return self._testset

    @trainset.setter
    def trainset(self, value):
        self._add_set('train', value)

    @valset.setter
    def valset(self, value):
        self._add_set('val', value)

    @testset.setter
    def testset(self, value):
        self._add_set('test', value)

    @property
    def train_len(self):
        return len(self.trainset) if self.trainset is not None else None

    @property
    def val_len(self):
        return len(self.valset) if self.valset is not None else None

    @property
    def test_len(self):
        return len(self.testset) if self.testset is not None else None

    @property
    def train_slice(self):
        return self._train_slice if hasattr(self, '_train_slice') else None

    @property
    def val_slice(self):
        return self._val_slice if hasattr(self, '_val_slice') else None

    @property
    def test_slice(self):
        return self._test_slice if hasattr(self, '_test_slice') else None

    def _add_set(self, split_type, _set):
        assert split_type in ['train', 'val', 'test']
        split_type = '_' + split_type
        name = split_type + 'set'
        if _set is None or isinstance(_set, Dataset):
            setattr(self, name, _set)
        else:
            indices = _set
            assert isinstance(indices, Index.__args__), \
                f"type {type(indices)} of `{name}` is not a valid type. " \
                "It must be a dataset or a sequence of indices."
            _set = Subset(self.torch_dataset, indices)
            _slice = self.torch_dataset.expand_indices(_set.indices,
                                                       merge=True)
            setattr(self, name, _set)
            slice_name = split_type + '_slice'  # e.g. trainset > _train_slice
            setattr(self, slice_name, _slice)

    def setup(self, stage: StageOptions = None):
        # splitting
        if self.splitter is not None:
            self.splitter.split(self.torch_dataset)
            self.trainset = self.splitter.train_idxs
            self.valset = self.splitter.val_idxs
            self.testset = self.splitter.test_idxs

        for key, scaler, in self.scalers.items():
            if key not in self.torch_dataset:
                raise RuntimeError("Cannot find a tensor to scale matching "
                                   f"key '{key}'.")
            # set scalers
            if stage == 'predict':
                tsl.logger.info(f'Set scaler for {key}: {scaler}')
            else:  # fit scalers before training
                data = getattr(self.torch_dataset, key)
                # get only training slice
                if 't' in self.torch_dataset.patterns[key]:
                    data = data[self.train_slice]

                mask = None
                if key == 'target' and self.mask_scaling:
                    if self.torch_dataset.mask is not None:
                        mask = self.torch_dataset.get_mask()[self.train_slice]

                scaler = scaler.fit(data, mask=mask, keepdims=True)
                tsl.logger.info(f'Fit and set scaler for {key}: {scaler}')
            self.torch_dataset.add_scaler(key, scaler)

    def get_dataloader(self, split: Literal['train', 'val', 'test'] = None,
                       shuffle: bool = False,
                       batch_size: Optional[int] = None) \
            -> Optional[DataLoader]:
        if split is None:
            dataset = self.torch_dataset
        elif split in ['train', 'val', 'test']:
            dataset = getattr(self, f'{split}set')
        else:
            raise ValueError("Argument `split` must be one of "
                             "'train', 'val', or 'test'.")
        if dataset is None:
            return None
        pin_memory = self.pin_memory if split == 'train' else None
        return StaticGraphLoader(dataset,
                                 batch_size=batch_size or self.batch_size,
                                 shuffle=shuffle,
                                 drop_last=split == 'train',
                                 num_workers=self.workers,
                                 pin_memory=pin_memory,
                                 generator=self.generator)

    def train_dataloader(self, shuffle: bool = True,
                         batch_size: Optional[int] = None) \
            -> Optional[DataLoader]:
        """"""
        return self.get_dataloader('train', shuffle, batch_size)

    def val_dataloader(self, shuffle: bool = False,
                       batch_size: Optional[int] = None) \
            -> Optional[DataLoader]:
        """"""
        return self.get_dataloader('val', shuffle, batch_size)

    def test_dataloader(self, shuffle: bool = False,
                        batch_size: Optional[int] = None) \
            -> Optional[DataLoader]:
        """"""
        return self.get_dataloader('test', shuffle, batch_size)




class CrossGaussianNoiseSyntheticDataset(TabularDataset):
    r"""A generator of synthetic datasets from an input model and input graph.

    The input model must be implemented as a :class:`torch.nn.Module` and must
    return the observation at the next step and (optionally) the hidden state
    for the next step. Gaussian noise will be added to the output of the model
    at each step.

    Args:
        num_features (int): Number of features in the generated dataset.
        num_nodes (int): Number of nodes in the graph.
        num_steps (int): Number of steps to generate.
        connectivity (SparseTensArray): Connectivity of the underlying graph.
        model (torch.nn.Module): Model used to generate data. If :obj:`None`,
            it will attempt to create model from ``model_class`` and
            ``model_kwargs``.
        model_class (type, optional): Class of the model used to generate the
            data.
            (default: :obj:`None`)
        model_kwargs (dict, optional): Keyword arguments needed to initialize
            the model.
            (default: :obj:`None`)
        sigma_noise (float): Standard deviation of the noise.
            (default: :obj:`0.2`)
        name (str, optional): Name for the generated dataset.
            (default: :obj:`None`)
        seed (int, optional): Seed for the random number generator.
            (default: :obj:`None`)
    """

    seed: int = None

    def __init__(self,
                 num_features: int,
                 num_nodes: int,
                 split: int,
                 num_steps: int,
                 connectivity: SparseTensArray,
                 min_window: int = 1,
                 o_model: nn.Module = None,
                 o_model_class: Type = None,
                 o_model_kwargs: Mapping = None,
                 o_sigma_noise: float = .2,
                 e_model: nn.Module = None,
                 e_model_class: Type = None,
                 e_model_kwargs: Mapping = None,
                 e_sigma_noise: float = .2,
                 cor_rat: float = 0.5,
                 include_exog: bool = True,
                 name: str = None,
                 seed: int = 42,
                 **kwargs):
        self.name = name
        self._num_nodes = num_nodes
        self._num_features = num_features
        self._num_steps = num_steps
        self._min_window = min_window
        self._include_exog = include_exog
        self.cor = cor_rat
        if seed is not None:
            self.seed = seed

        if o_model is not None:
            self.o_model = o_model
        else:
            self.o_model = o_model_class(**o_model_kwargs)

        self._model_forward_signature = foo_signature(o_model.forward)

        self.o_sigma_noise = o_sigma_noise

        if e_model is not None:
            self.e_model = e_model
        else:
            self.e_model = e_model_class(**e_model_kwargs)

        self._model_forward_signature = foo_signature(e_model.forward)

        self.e_sigma_noise = e_sigma_noise

        if connectivity is not None:
            self.connectivity = parse_connectivity(connectivity,
                                                   target_layout='edge_index',
                                                   num_nodes=num_nodes)
        else:
            self.connectivity = None
        self._main_num = split
        self._exog_num = connectivity.shape[1] - split

        target, optimal_pred, mask, modality = self.load()
        self.modality = modality
        super().__init__(target=target, mask=mask, name=name, **kwargs)

        self.add_covariate('optimal_pred', optimal_pred, 't n f')

    def load_raw(self, *args, **kwargs):
        return self.generate_data(self.seed)

    # @property
    # def mae_optimal_model(self):
    #     r""":math:`\mathbb{E}[|\mathbf{X}|]` of a Gaussian
    #     :math:`\mathbf{X} \sim \mathcal{N}(0, \sigma^2)`, computed as
    #     :math:`\varepsilon = \sqrt{\frac{2}{\pi}}\sigma`.
    #     """
    #     return math.sqrt(2.0 / math.pi) * self.o_sigma_noise

    def _filter_forward_kwargs(self, kwargs):
        if not self._model_forward_signature['has_kwargs']:
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in self._model_forward_signature['signature']
            }
        return kwargs

    def _model_forward(self, *args, **kwargs):
        kwargs = self._filter_forward_kwargs(kwargs)
        out = self.o_model(*args, **kwargs)
        if len(out) != 2:
            return out, None
        # Assumes that if the output has length 2,
        # then it will contain [output, hidden_state].
        return out

    def _e_model_forward(self, *args, **kwargs):
        kwargs = self._filter_forward_kwargs(kwargs)
        out = self.e_model(*args, **kwargs)
        if len(out) != 2:
            return out, None
        # Assumes that if the output has length 2,
        # then it will contain [output, hidden_state].
        return out

    def generate_data(self, seed=None):
        """"""
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        # initialize with noise
        x = torch.empty(
            (self._num_steps + self._min_window, self._num_nodes,
             self._num_features)).normal_(generator=rng) * self.o_sigma_noise

        y_opt = torch.empty(
            (self._num_steps, self._num_nodes, self._num_features))

        if self.connectivity is None:
            edge_index = edge_weight = None
        else:
            edge_index, edge_weight = self.connectivity

            if edge_weight is None:
                edge_weight = torch.ones(edge_index.shape[1])

            adj = torch.eye(self._num_nodes, dtype=torch.float32)
            adj[tuple(edge_index)] = edge_weight

            o_edge_index, o_edge_weight = dense_to_sparse(adj[:self._main_num, :self._main_num]) # N N
            e_edge_index, e_edge_weight = dense_to_sparse(adj[self._main_num:, self._main_num:]) # M M

            c_adj = adj[:self._main_num, self._main_num:]
            # c_edge_index, c_edge_weight = dense_to_sparse(c_adj) # N M

        with torch.no_grad():
            eh_t = None
            oh_t = None
            for t in tqdm(range(self._min_window,
                                self._min_window + self._num_steps),
                          desc=f"Generating {self.__class__.__name__} data"):
                # ft modelling 
                e_t, eh_t = self._e_model_forward(x[None, t - self._min_window:t, self._main_num:],
                                               h=eh_t,
                                               t=t,
                                               edge_index=e_edge_index,
                                               edge_weight=e_edge_weight)
                f_t = e_t + torch.zeros_like(e_t).normal_(generator=rng) * self.e_sigma_noise
                x[t:t + 1, self._main_num:] = f_t[0]

                Uf_t = c_adj @ f_t

                # Adding to original graph
                o_t, oh_t = self._model_forward(x[None, t - self._min_window:t, :self._main_num],
                                               h=oh_t,
                                               t=t,
                                               edge_index=o_edge_index,
                                               edge_weight=o_edge_weight)
                x_t = torch.tanh(((self.cor * o_t) + (1 - self.cor) * Uf_t))
                y_opt[t - self._min_window:t + 1 - self._min_window, :self._main_num] = x_t[0]
                # add noise
                x_t = x_t + torch.zeros_like(x_t).normal_(
                    generator=rng) * self.o_sigma_noise
                x[t:t + 1, :self._main_num] = x_t[0]

        x = torch_to_numpy(x[self._min_window:])
        y_opt = torch_to_numpy(y_opt)

        modality = np.zeros((self._num_nodes, 1))
        modality[self._main_num:] = 1

        # Just take the original graph if not including exogeneous data
        if not self._include_exog:
            if self.connectivity is not None:
                self.connectivity = parse_connectivity(o_edge_index,
                                                    target_layout='edge_index',
                                                    num_nodes=self._main_num)
            else:
                self.connectivity = None
            
            x = x[:, :self._main_num]
            y_opt = y_opt[:, :self._main_num]

        return x, y_opt, np.ones_like(x), modality

    def get_connectivity(self, layout: str = 'edge_index', **kwargs):
        """"""
        if self.connectivity is not None:
            return parse_connectivity(connectivity=self.connectivity,
                                      target_layout=layout,
                                      num_nodes=self.n_nodes)
        return None

class _GPVAR(GraphPolyVAR):
    def forward(self, x, edge_index, edge_weight=None):
        out = super(_GPVAR, self).forward(x, edge_index, edge_weight)
        return torch.tanh(out)

# SMALL DATASET
SIZES_X_sm = [30, 30, 30, 30]
PROB_X_sm = [[0.30, 0.01, 0.01, 0.01],
          [0.01, 0.30, 0.01, 0.01],
          [0.01, 0.01, 0.30, 0.01],
          [0.01, 0.01, 0.01, 0.30]]

SIZES_Y_sm = [25, 25, 25]
PROB_Y_sm = [[0.30, 0.01, 0.01],
          [0.01, 0.30, 0.01],
          [0.01, 0.01, 0.30]]

# Cross-layer bipartite SBM
# Y has 2 blocks, X has 2 blocks
SIZES_XY_sm = ([25, 25, 25], [30, 30, 30, 30])
PROB_XY_sm = [[0.25, 0.00, 0.00],
            [0.10, 0.10, 0.00],
            [0.00, 0.10, 0.10],
            [0.00, 0.00, 0.25]]

# LARGE DATASET
SIZES_X_lr = [100, 100, 100, 100, 100]
PROB_X_lr = [[0.30, 0.01, 0.01, 0.01, 0.01],
          [0.01, 0.30, 0.01, 0.01, 0.01],
          [0.01, 0.01, 0.30, 0.01, 0.01],
          [0.01, 0.01, 0.01, 0.30, 0.01],
          [0.01, 0.01, 0.01, 0.01, 0.30]]

SIZES_Y_lr = [100, 100, 100, 100]
PROB_Y_lr = [[0.30, 0.01, 0.01, 0.01],
          [0.01, 0.30, 0.01, 0.01],
          [0.01, 0.01, 0.30, 0.01],
          [0.01, 0.01, 0.01, 0.30]]

# Cross-layer bipartite SBM
# Y has 2 blocks, X has 2 blocks
SIZES_XY_lr = ([100, 100, 100, 100], [100, 100, 100, 100, 100])
PROB_XY_lr = [[0.05, 0.00, 0.00, 0.00],
              [0.01, 0.05, 0.00, 0.00],
              [0.00, 0.01, 0.05, 0.00],
              [0.00, 0.00, 0.01, 0.05],
              [0.02, 0.00, 0.00, 0.02]]

class CrossGPVARDataset(CrossGaussianNoiseSyntheticDataset):
    """Generator for synthetic datasets from a graph polynomial VAR filter on
    triangular community graphs as shown in the paper `"AZ-whiteness test: a
    test for uncorrelated noise on spatio-temporal graphs"
    <https://arxiv.org/abs/2204.11135>`_ (Zambon et al., NeurIPS 22).

    Args:
        num_communities (int): Number of communities (triangles) in the graph.
        num_steps (int): Length of the generated sequence.
        filter_params (iterable): Parameters of the graph polynomial filter
            used to generate the dataset.
        sigma_noise (float): Standard deviation of the noise.
        norm (str): The normalization used for edges and edge weights. The
            available options are: :obj:`'gcn'`, :obj:`'asym'` and
            :obj:`'none'`.
            (default: :obj:`'none'`)
        name (optional, str): Name of the dataset.
    """

    def __init__(self,
                 num_steps: int,
                 o_filter_params: Union[List, Tensor, ndarray],
                 e_filter_params: Union[List, Tensor, ndarray],
                 o_sigma_noise: float = .2,
                 o_norm: str = 'none',
                 e_sigma_noise: float = .2,
                 e_norm: str = 'none',
                 cor_rat: float = 0.5,
                 o_trim_thresh: float = 0.0,
                 e_trim_thresh: float = 0.8,
                 size: str = 'small',
                 include_exog: bool = True,
                 seed=42,
                 name: str = None):
        np.random.seed(seed)

        if name is None:
            name = "GP-VAR"

        if size == 'small':
            sbm_params = {'sizes_x': SIZES_X_sm, 'prob_x': PROB_X_sm,
                          'sizes_y': SIZES_Y_sm, 'prob_y': PROB_Y_sm,
                          'sizes_xy': SIZES_XY_sm, 'prob_xy': PROB_XY_sm}
            o_trim_thresh = 0.7
            e_trim_thresh = 0.4
        elif size == 'large':
            sbm_params = {'sizes_x': SIZES_X_lr, 'prob_x': PROB_X_lr,
                          'sizes_y': SIZES_Y_lr, 'prob_y': PROB_Y_lr,
                          'sizes_xy': SIZES_XY_lr, 'prob_xy': PROB_XY_lr}
            o_trim_thresh = 0.95
            e_trim_thresh = 0.85

        print(o_trim_thresh, e_trim_thresh)
        Gx, Gy, Gxy, A_aug = generate_multiplex_sbm(
            **sbm_params,
            seed=seed
        )
        self.air_max_nodes = len(Gx.nodes)
        # Trim edges below threshold
        A_key = A_aug[:self.air_max_nodes, :self.air_max_nodes].copy()
        A_exo = A_aug[self.air_max_nodes:, self.air_max_nodes:].copy()

        A_key = ((A_key * np.random.rand(*A_key.shape)) > o_trim_thresh).astype(float)
        A_exo = ((A_exo * np.random.rand(*A_exo.shape)) > e_trim_thresh).astype(float)

        A_aug[:self.air_max_nodes, :self.air_max_nodes] = A_key
        A_aug[self.air_max_nodes:, self.air_max_nodes:] = A_exo
        # A_aug = ((A_aug * np.random.rand(*A_aug.shape)) > trim_thresh).astype(float)

        self.A_aug = A_aug
        num_nodes = A_aug.shape[0]
        

        edge_index = []
        edge_weight = []

        for i, row in enumerate(A_aug):
            for j, el in enumerate(row):
                if el > 0 and i != j and np.isfinite(el):
                    edge_index.append([i, j])
                    edge_weight.append(el)

        edge_index = np.array(edge_index).T
        edge_weight = np.array(edge_weight)

        # Calculate the filters
        if not isinstance(o_filter_params, Tensor):
            o_filter_params = torch.as_tensor(o_filter_params, dtype=torch.float32)
        if not isinstance(e_filter_params, Tensor):
            e_filter_params = torch.as_tensor(e_filter_params, dtype=torch.float32)

        o_filter = _GPVAR.from_params(filter_params=o_filter_params,
                                    norm=o_norm,
                                    cached=True)
        e_filter = _GPVAR.from_params(filter_params=e_filter_params,
                                    norm=e_norm,
                                    cached=True)  

        super(CrossGPVARDataset, self).__init__(num_features=1,
                                           num_nodes=num_nodes,
                                           num_steps=num_steps,
                                           split=self.air_max_nodes,
                                           connectivity=edge_index,
                                           min_window=o_filter.temporal_order,
                                           o_model=o_filter,
                                           o_sigma_noise=o_sigma_noise,
                                           e_model=e_filter,
                                           e_sigma_noise=e_sigma_noise,
                                           include_exog=include_exog,
                                           cor_rat=cor_rat,
                                           name=name)

def generate_multiplex_sbm(
    sizes_x,
    prob_x,
    sizes_y,
    prob_y,
    sizes_xy,
    prob_xy,
    seed=None
):
    """
    Generate a multiplex graph with:
    - Layer X: SBM(sizes_x, prob_x)  → placed in UPPER block
    - Layer Y: SBM(sizes_y, prob_y)  → placed in LOWER block
    - Cross-layer edges via bipartite SBM(sizes_xy, prob_xy)

    Returns:
        Gx, Gy, Gxy, A_aug
    """

    rng = np.random.default_rng(seed)

    # -----------------------------
    # 1. Layer X (SBM) – goes to upper-left
    # -----------------------------
    Gx = nx.stochastic_block_model(
        sizes_x, prob_x, seed=seed
    )
    n_x = sum(sizes_x)

    # -----------------------------
    # 2. Layer Y (SBM) – goes to lower-right
    # -----------------------------
    Gy = nx.stochastic_block_model(
        sizes_y, prob_y, seed=(seed + 1 if seed else None)
    )
    n_y = sum(sizes_y)

    # -----------------------------
    # 3. Cross-layer bipartite SBM
    # -----------------------------
    sizes_y_xy, sizes_x_xy = sizes_xy
    B_xy = prob_xy

    Gxy = nx.Graph()

    # X nodes: block labels
    x_blocks = []
    for b, size in enumerate(sizes_x_xy):
        x_blocks += [b] * size

    # Y nodes: block labels
    y_blocks = []
    for b, size in enumerate(sizes_y_xy):
        y_blocks += [b] * size

    # Add bipartite node sets:
    #  - X nodes: 0 .. n_x - 1  (bipartite = 0)
    #  - Y nodes: n_x .. n_x+n_y-1 (bipartite = 1)

    for i in range(n_x):
        Gxy.add_node(i, bipartite=0, block=x_blocks[i])

    for j in range(n_y):
        Gxy.add_node(j + n_x, bipartite=1, block=y_blocks[j])

    # Sample bipartite edges
    for i in range(n_x):
        for j in range(n_y):
            bi = x_blocks[i]
            bj = y_blocks[j]
            if rng.random() < B_xy[bi][bj]:
                Gxy.add_edge(i, j + n_x)

    # -----------------------------
    # 4. Build supra-adjacency
    # -----------------------------
    A_aug = np.zeros((n_x + n_y, n_x + n_y))

    # Layer X (upper-left)
    Ax = nx.to_numpy_array(Gx)
    A_aug[:n_x, :n_x] = Ax

    # Layer Y (lower-right)
    Ay = nx.to_numpy_array(Gy)
    A_aug[n_x:, n_x:] = Ay

    # Cross-layer adjacency
    Axy = nx.to_numpy_array(Gxy)

    # X → Y block (upper-right)
    A_aug[:n_x, n_x:] = Axy[:n_x, n_x:]

    # Y → X block (lower-left)
    A_aug[n_x:, :n_x] = Axy[n_x:, :n_x]

    return Gx, Gy, Gxy, A_aug

def add_missing_sensors_cross(dataset: AirCross | CrossGPVARDataset,
                              p_noise=0.05,
                              p_fault=0.01,
                              min_seq=1,
                              max_seq=10,
                              seed=None,
                              inplace=True,
                              masked_sensors = [],
                              connect = None,
                              spatial_shift = False, 
                              order = 0,
                              node_features = 'CC',
                              mode='road'):
    if seed is None:
        seed = np.random.randint(1e9)
    # Fix seed for random mask generation
    random = np.random.default_rng(seed)

    # Compute evaluation mask
    shape = (dataset.length, dataset.air_max_nodes, dataset.n_channels)
    adj = dataset.get_connectivity(**connect, layout='dense')
    air_adj = adj[:dataset.air_max_nodes, :dataset.air_max_nodes]  
    eval_mask = np.zeros_like(dataset.mask)

    if masked_sensors is None:
        if spatial_shift:
            tmp_mask = shift_mask(shape, feature=node_features, order=order, 
                                   adj=air_adj, p_noise=p_noise)
            dataset.seed = seed
        else:
            tmp_mask = sample_mask(shape,
                                    p=p_fault,
                                    p_noise=p_noise,
                                    mode=mode,
                                    adj=air_adj)
            
            dataset.p_fault = p_fault
            dataset.p_noise = p_noise
            dataset.min_seq = min_seq
            dataset.max_seq = max_seq
            dataset.seed = seed
            dataset.random = random

        # mask = rearrange(eval_mask, "b n 1 -> b n")
        mask_sum = tmp_mask.sum(0)  # n
        masked_sensors = (np.where(mask_sum > 0)[0]).tolist()
        eval_mask[:, :dataset.air_max_nodes] = tmp_mask
    else:
        masked_sensors = list(masked_sensors)
        eval_mask = np.zeros_like(dataset.mask)
        eval_mask[:, masked_sensors] = dataset.mask[:, masked_sensors]

    # Convert to missing values dataset
    dataset = to_missing_values_dataset(dataset, eval_mask, inplace)

    # Store evaluation mask params in dataset
    return dataset, masked_sensors
