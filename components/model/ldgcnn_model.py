import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP
from torch_geometric.nn import EdgeConv
from torch_geometric.utils import scatter
from torch_cluster import knn_graph


def apply_knn(x: torch.Tensor, batch: torch.Tensor, k=30) -> torch.Tensor:
    """Applies knn for making graph.

    Args:
        x: Node feature matrix.
        batch: Batch vector, which assigns each node to a specific example.
        k: The number of neighbors.

    Returns:
        Tensor with graph edges for EdgeConv.
    """
    idx = knn_graph(x, k=k, batch=batch)
    return idx


class LDGCNNSegmentor(nn.Module):
    """Class with implementation of segmentation part of LDGCNN.

    Attributes:
        k: Number of nearest neighbours for creating graph using KNN.
        edge_conv1: First Edge convolution layer. It uses only model input.
        edge_conv2: Second Edge convolution layer.
         It uses model input + edge_conv1 result (in_channels + 64 features).
        edge_conv3: Third Edge convolution layer.
         It uses model input + edge_conv1 result + edge_conv2 result
         (in_channels + 64 + 64 features).
        edge_conv4: Fourth Edge convolution layer.
         It uses model input + edge_conv1 result + edge_conv2 result + edge_conv3 result
         (in_channels + 64 + 64 + 64 features).
        fe_mlp: MLP that uses extracted local features for creating global features vector.
        mlp: MLP that uses concatenated local and global features vectors
         for predicting segmentation scores.
    """
    def __init__(self, in_channels: int, out_channels: int, k=30, aggr='max'):
        """Creates an instance of the class.

        Args:
            out_channels: Number of output segmentation classes.
            k: Number of nearest neighbours for creating graph using KNN.
            aggr: Aggregation function for using in EdgeConv.
        """
        super().__init__()

        self.k = k

        # Extracting global features
        self.edge_conv1 = EdgeConv(MLP([2 * in_channels, 64, 64]), aggr)
        self.edge_conv2 = EdgeConv(MLP([2 * (64 + in_channels), 64, 64]), aggr)
        self.edge_conv3 = EdgeConv(MLP([2 * (64 + 64 + in_channels), 64, 64]), aggr)
        self.edge_conv4 = EdgeConv(MLP([2 * (64 + 64 + 64 + in_channels), 128, 128]), aggr)

        self.fe_mlp = MLP([in_channels + 64 + 64 + 64 + 128, 1024, 1024])

        # MLP for prediction segmentation scores
        self.mlp = MLP([in_channels + 64 + 64 + 64 + 128 + 1024, 256, 256, 128, out_channels],
                       dropout=0.5,
                       norm=None)

    def forward(self, data) -> torch.Tensor:
        """Performs forward propagation.

        Args:
            data: DataBatch.

        Returns:
            Model output for desired input as a torch.Tensor.
        """
        x, pos, batch = data.x, data.pos, data.batch
        num_points = batch.size(0)
        # x0 is (num_points, in_channels)
        x0 = torch.cat([x, pos], dim=-1)

        edge_index = apply_knn(x0, batch, k=self.k)
        # (num_points, in_channels) -> (num_points, 64)
        x1 = self.edge_conv1(x0, edge_index)

        edge_index = apply_knn(x1, batch, k=self.k)
        link_1 = torch.cat([x0, x1], dim=1)
        # (num_points, in_channels + 64) -> (num_points, 64)
        x2 = self.edge_conv2(link_1, edge_index)

        edge_index = apply_knn(x2, batch, k=self.k)
        link_2 = torch.cat([x0, x1, x2], dim=1)
        # (num_points, in_channels + 64 + 64) -> (num_points, 64)
        x3 = self.edge_conv3(link_2, edge_index)

        edge_index = apply_knn(x2, batch, k=self.k)
        link_3 = torch.cat([x0, x1, x2, x3], dim=1)
        # (num_points, in_channels + 64 + 64 + 64) -> (num_points, 128)
        x4 = self.edge_conv4(link_3, edge_index)

        link_4 = torch.cat([x0, x1, x2, x3, x4], dim=-1)
        # (num_points, in_channels + 64 + 64 + 64 + 128) -> (num_points, 1024)
        x5 = self.fe_mlp(link_4)

        # x6 is a global feature tensor
        # (num_points, 1024) -> (1, 1024)
        global_features, _ = torch.max(x5, dim=0, keepdim=True)
        # (1, 1024) -> (num_points, 1024)
        global_features_repeated = global_features.repeat(num_points, 1)
        # (num_points, in_channels + 64 + 64 + 64 + 128) + (num_points, 1024)
        # -> (num_points, in_channels + 64 + 64 + 64 + 128 + 1024)
        local_global_features = torch.cat([link_4, global_features_repeated], axis=1)
        # (num_points, in_channels + 64 + 64 + 64 + 128 + 1024) -> (num_points, out_channels)
        out = self.mlp(local_global_features)
        return nn.functional.log_softmax(out, dim=1)
