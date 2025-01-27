r"""
Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism <https://arxiv.org/abs/2201.12987>`_.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import is_undirected
from torch_sparse import transpose

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor, DGINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor, DvGINFeatExtractor
from .GCNs import DGCNFeatExtractor, GCNFeatExtractor


@register.model_register
class GMT(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GMT, self).__init__(config)
        self.gnn = GINFeatExtractor(config)
        self.extractor = ExtractorMLP(config)

        self.classifier = Classifier(config)
        self.learn_edge_att = True
        self.sampling_method = config.ood.extra_param[0]
        self.sampling_rounds = config.ood.extra_param[3]
        self.config = config

    def forward(self, *args, **kwargs):
        r"""
        The GSAT model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            Label predictions and other results for loss calculations.

        """
        data = kwargs.get('data')
        batch_size = data.batch[-1].item() + 1
        emb = self.gnn(*args, without_readout=True, **kwargs)
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)

        self.causal_adj = generate_adjacency_matrices(data.edge_index, att_log_logits.squeeze(), batch_size).clone()

        att = self.sampling(att_log_logits, self.training)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                nodesize = data.x.shape[0]
                edge_att = (att + transpose(data.edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)

        sampling_logits = []
        sampling_trials = self.sampling_rounds
        while len(sampling_logits)<sampling_trials:
            b = torch.bernoulli(edge_att)
            cur_edge_att = (b - edge_att).detach() + edge_att  # straight-through estimator
            set_masks(cur_edge_att, self)
            x = self.gnn(*args, **kwargs)
            logits = self.classifier(x)
            clear_masks(self)
            # clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=cur_edge_att)
            sampling_logits.append(self.classifier(x))
        logits = torch.stack(sampling_logits).mean(dim=0)
        # loss, loss_dict = self.__loss__(att_log_logits.sigmoid(), clf_logits, data.y, epoch,training)

        # set_masks(edge_att, self)
        # x, diff_loss = self.gnn(*args, **kwargs)
        # logits = self.classifier(x)
        # clear_masks(self)
        return logits, att, edge_att

    def sampling(self, att_log_logits, training):
        if self.sampling_method =="normal":
            att = self.normal_sample(att_log_logits, temp=1.0, training=training)
        elif self.sampling_method =="bern":
            att = self.concrete_sample(att_log_logits, temp=1.0, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    @staticmethod
    def normal_sample(logits, temp, training):
        if training:
            random_noise = torch.randn_like(logits)
            att_normal = ((logits + random_noise * temp).sigmoid())
        else:
            att_normal = logits.sigmoid()
        return att_normal

    @staticmethod
    def gumbel_softmax_sample(logits, temp, training):
        random_noise = torch.empty_like(logits).uniform_(1e-10, 1 - 1e-10)
        gumbel_noise = -torch.log(-torch.log(random_noise))
        y = (logits + gumbel_noise) / temp
        return torch.softmax(y, dim=-1)

    @staticmethod
    def logistic_sample(logits, temp, training):
        if training:
            random_noise = torch.empty_like(logits).uniform_(1e-10, 1 - 1e-10)
            logistic_noise = torch.log(random_noise) - torch.log(1 - random_noise)
            att_logistic = ((logits + logistic_noise) / temp).sigmoid()
        else:
            att_logistic = logits.sigmoid()
        return att_logistic


@register.model_register
class GMTvGIN(GMT):
    r"""
    The GIN virtual node version of GSAT.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GMTvGIN, self).__init__(config)
        self.gnn = vGINFeatExtractor(config)


@register.model_register
class GMTGCN(GMT):
    r"""
    The GIN virtual node version of GSAT.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GMTGCN, self).__init__(config)
        self.gnn = GCNFeatExtractor(config)


class ExtractorMLP(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__()
        hidden_size = config.model.dim_hidden
        self.learn_edge_att = config.ood.extra_param[0]  # learn_edge_att
        dropout_p = config.model.dropout_rate

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                if batch.shape[0] == 0:
                    inputs = inputs
                else:
                    inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)


def set_masks(mask: Tensor, model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module._apply_sigmoid = False
            module.__explain__ = True
            module._explain = True
            module.__edge_mask__ = mask
            module._edge_mask = mask


def clear_masks(model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module._explain = False
            module.__edge_mask__ = None
            module._edge_mask = None


def generate_adjacency_matrices(edge_index, edge_weights, bz, num_nodes=100):
    """
    Generate an adjacency matrix from edge indices and edge weights.

    Parameters:
    - edge_index (torch.Tensor): A tensor of shape [2, num_edges] containing the indices of the edges.
    - edge_weights (torch.Tensor): A tensor of shape [num_edges] containing the weights of the edges.
    - num_nodes (int): The number of nodes in the graph.

    Returns:
    - adjacency_matrix (torch.Tensor): The adjacency matrix of shape [num_nodes, num_nodes].
    """
    adjacency_matrix = torch.zeros((bz * num_nodes, bz * num_nodes), dtype=edge_weights.dtype).to(edge_weights.device)
    adjacency_matrix[edge_index[0], edge_index[1]] = edge_weights

    # split the adjacency matrix into multiple adjacency matrices
    adjacency_matrices = [adjacency_matrix[i * num_nodes: (i + 1) * num_nodes, i * num_nodes: (i + 1) * num_nodes] for i in range(bz)]
    return torch.stack(adjacency_matrices).to(edge_weights.device)
