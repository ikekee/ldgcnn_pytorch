import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP
from torch_geometric.nn import EdgeConv
from torch_geometric.utils import scatter


# TODO: Possibly can be replaced with pool.KNNIndex (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.KNNIndex.html#torch_geometric.nn.pool.KNNIndex)
def knn(x: torch.Tensor, k: int):
    """Performs K-NN operation on an input tensor.

    Args:
        x: Input tensor.
        k: Number of nearest neighbours.

    Returns:
        Tensor size (x.size(0), x.size(2), k) - (batch_size, num_points, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x: torch.Tensor, k=30) -> torch.Tensor:
    """Applies K-NN to an input and gets each node's features.

    Args:
        x: Input tensor size of (batch_size, num_dims, num_points) to perform operations on.
        k: Number of nearest neighbours for forming graph.

    Returns:
        Tensor size of (batch_size, num_points, k, num_dims)
         with features for each found nearest neighbour.
    """
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    # x = x.view(batch_size, -1, num_points)
    # idx is (batch_size, num_points, k)
    idx = knn(x, k=k)
    device = torch.device('cuda')

    # idx_base is (batch_size, 1, 1)
    # each element contains num_points value multiplication by index of first dim,
    # e.g. [ [ [0] ], [ [2048] ], ...]
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    # add to each element of idx row wise element of idx_base
    idx = idx + idx_base

    idx = idx.view(-1)
    # x will be (batch_size, num_points, num_dims)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    # x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    #
    # feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    #
    # return feature  # (batch_size, 2*num_dims, num_points, k)
    return feature

    return feature  # (batch_size, 2*num_dims, num_points, k)

class LDGCNNSegmentor(nn.Module):
    """
    Attributes:
        ...
    """
    def __init__(self, out_channels: int, k=30, aggr='max'):
        """

        Args:
            num_points
            out_channels:
            k:
            aggr:
        """
        super().__init__()

        self.k = k

        # Extracting global features
        self.edge_conv1 = EdgeConv(MLP([2 * 6, 64, 64]), aggr)
        self.edge_conv2 = EdgeConv(MLP([2 * 64, 64, 64]), aggr)
        self.edge_conv3 = EdgeConv(MLP([2 * 64, 64, 64]), aggr)
        self.edge_conv4 = EdgeConv(MLP([2 * 64, 128, 128]), aggr)

        self.fe_mlp = MLP([128, 1024, 1024])

        # MLP for prediction segmentation scores
        self.mlp = MLP([1347, 256, 256, 128, out_channels], dropout=0.5, norm=None)

    def forward(self, data):
        """

        Args:
            data:

        Returns:

        """
        x, pos, batch = data.x, data.pos, data.batch
        x = torch.unsqueeze(x, dim=-2)
        x0 = torch.cat([x, pos], dim=-1)

        num_points = x0.size(2)

        x1 = get_graph_feature(x0, k=self.k)
        x1 = self.edge_conv1(x1, batch)

        x2 = get_graph_feature(x1, k=self.k)
        # x0 will be (batch_size, num_dims, 1, num_points)
        # it is needed for concatenation with knn result
        x0 = torch.unsqueeze(x0, dim=-2)
        link_1 = x0 + x2
        x2 = self.edge_conv2(link_1, batch)

        # input point cloud + result of first EdgeConv + result of second EdgeConv
        x3 = get_graph_feature(x2, k=self.k)
        x2 = torch.unsqueeze(x2, dim=-2)
        link_2 = x0 + x2 + x3
        x3 = self.edge_conv3(link_2, batch)

        # input point cloud + result of first EdgeConv + result of second EdgeConv
        x4 = get_graph_feature(x3, k=self.k)
        x3 = torch.unsqueeze(x3, dim=-2)
        link_3 = x0 + x2 + x3 + x4
        x4 = self.edge_conv4(link_3, batch)

        link_4 = torch.cat([x0, x1, x2, x3, x4], dim=-1)

        x5 = self.fe_mlp(link_4)
        # x6 is (batch_size, 1024)
        # x6 is a global feature tensor
        x6, _ = torch.max(x5, dim=1)
        x6_repeated = x6.repeat(1, 1, num_points)

        # tensor is (batch_size)
        local_global_features = torch.cat([link_4, x6_repeated], axis=1)

        out = self.mlp(local_global_features)
        return nn.functional.log_softmax(out, dim=1)
