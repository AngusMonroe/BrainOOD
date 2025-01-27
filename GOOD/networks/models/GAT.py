r"""The Graph Attention Network from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper.
"""
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, Size
from torch_sparse import SparseTensor

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier


@register.model_register
class GAT(GNNBasic):
    r"""
    The Graph Attention Network from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feat_encoder = GATFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The GAT model implementation.

        Args:
            *args (list): argument list for the use of arguments_read.
            **kwargs (dict): key word arguments for the use of arguments_read.

        Returns (Tensor):
            label predictions
        """
        out_readout = self.feat_encoder(*args, **kwargs)
        out = self.classifier(out_readout)
        return out


class GATFeatExtractor(GNNBasic):
    r"""
    GAT feature extractor using the :class:`~GATEncoder`.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GATFeatExtractor, self).__init__(config)
        self.encoder = GATEncoder(config)
        self.edge_feat = False

    def forward(self, *args, **kwargs):
        r"""
        GAT feature extractor using the :class:`~GATEncoder`.

        Args:
            *args (list): argument list for the use of arguments_read.
            **kwargs (dict): key word arguments for the use of arguments_read.

        Returns (Tensor):
            node feature representations
        """
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        out_readout = self.encoder(x, edge_index, edge_weight, batch)
        return out_readout


class DGATFeatExtractor(GNNBasic):
    r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.dataset_type`)
    """
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(DGATFeatExtractor, self).__init__(config)
        num_layer = config.model.model_layer
        if config.dataset.dataset_type == 'mol':
            raise NotImplementedError
        else:
            self.encoder = GATEncoder(config, **kwargs)
            self.hp_encoder = HPGATEncoder(config, **kwargs)
            # self.fuse = nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)
            # self.fuse = nn.Linear(config.model.dim_hidden, config.model.dim_hidden)
            self.edge_feat = False

        # self.diffusion = DenoiseModel(config.model.dim_hidden, config.model.dim_hidden)

    def forward(self, *args, **kwargs):
        r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            node feature representations
        """
        x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
        kwargs.pop('batch_size', 'not found')
        # x, loss = self.diffusion(x)
        loss = 0.0
        out_readout = self.encoder(x, edge_index, batch, batch_size, **kwargs)
        if kwargs.get('without_readout'):
            post_diffusion = self.hp_encoder(out_readout, edge_index, batch, batch_size, **kwargs)
            dim = post_diffusion.shape[-1]
            post_diffusion = post_diffusion.view(-1, dim, dim)
            inner_product = torch.bmm(post_diffusion, post_diffusion.transpose(1, 2))  # Shape: (bz, n, n)
            result_x = torch.tanh(inner_product).view(-1, dim)
            loss = self.mse_loss(result_x, x)
            # loss = self.mse_loss(post_diffusion, x)
        # res = self.fuse(torch.cat([out_readout, hp_out_readout], dim=-1))
        # res = out_readout + hp_out_readout
        # res = self.fuse(res)
        return out_readout, loss

    def mse_loss(self, x_reconstructed, x):
        criterion = nn.MSELoss()
        return criterion(x_reconstructed, x)


class GATEncoder(BasicEncoder):
    r"""
    The GAT encoder using the :class:`~GATConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GATEncoder, self).__init__(config)
        num_layer = config.model.model_layer
        heads = 1 #config.model.attention_heads

        self.conv1 = GATConv(config.dataset.dim_node, config.model.dim_hidden, heads=heads)
        self.convs = nn.ModuleList(
            [
                GATConv(config.model.dim_hidden * heads, config.model.dim_hidden, heads=heads)
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        r"""
        The GAT encoder.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator

        Returns (Tensor):
            node feature representations
        """
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        if kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout


class GATConv(gnn.GATConv):
    r"""The graph attention operator from the `"Graph Attention Networks"
        <https://arxiv.org/abs/1710.10903>`_ paper.

    Args:
        *args (list): argument list for the use of arguments_read.
        **kwargs (dict): Additional key word arguments for the use of arguments_read.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        r"""
        The GAT graph attention operator.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_weight (Tensor): edge weights

        Returns (Tensor):
            node feature representations
        """
        x = self.lin_l(x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        if self.bias is not None:
            out += self.bias
        return out


class HPGATEncoder(BasicEncoder):
    r"""
    The high-pass GAT encoder using the :class:`~GATConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(HPGATEncoder, self).__init__(config)
        num_layer = config.model.model_layer

        self.conv1 = HPGATConv(config.dataset.dim_node, config.model.dim_hidden, 1)
        self.convs = nn.ModuleList(
            [
                HPGATConv(config.model.dim_hidden * 1, config.model.dim_hidden, 1)
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x: Tensor, edge_index: Adj, batch: Tensor, batch_size: int, **kwargs):
        r"""
        The high-pass GAT encoder.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator

        Returns (Tensor):
            node feature representations
        """
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        if kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout


class HPGATConv(GATConv):
    """
    High-pass GAT layer using GATConv as the base class.
    This layer highlights the differences between the original node features
    and the aggregated features from neighbors, acting as a high-pass filter.
    """

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """
        Args:
            x (Tensor): Node features
            edge_index (Tensor): Edge indices
            edge_weight (Tensor, optional): Edge weights

        Returns:
            Tensor: Node feature representations after applying high-pass filtering
        """

        # Perform the standard GAT convolution to get the aggregated features
        x_agg = super().forward(x, edge_index, edge_weight)

        # Subtract the aggregated features from the original features (high-pass filtering)
        x_high_pass = x - x_agg

        return x_high_pass
