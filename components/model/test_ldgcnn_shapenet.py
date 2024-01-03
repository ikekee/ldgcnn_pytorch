from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torchmetrics.functional.classification import multiclass_jaccard_index

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv
from torch_geometric.utils import scatter

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if ROOT_DIR not in sys.path:
    sys.path.append(str(ROOT_DIR))

from components.model.ldgcnn_model import LDGCNNSegmentor


def train(loader: DataLoader,
          model: LDGCNNSegmentor,
          device: torch.device,
          optimizer: Optimizer):
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for i, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 10 == 0:
            print(f'[{i + 1}/{len(loader)}] Loss: {total_loss / 10:.4f} '
                  f'Train Acc: {correct_nodes / total_nodes:.4f}')
            total_loss = correct_nodes = total_nodes = 0


@torch.no_grad()
def test(loader: DataLoader,
         model: LDGCNNSegmentor,
         device: torch.device):
    model.eval()

    ious, categories = [], []
    y_map = torch.empty(loader.dataset.num_classes, device=device).long()
    for data in loader:
        data = data.to(device)
        outs = model(data)

        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for out, y, category in zip(outs.split(sizes), data.y.split(sizes),
                                    data.category.tolist()):
            category = list(ShapeNet.seg_classes.keys())[category]
            part = ShapeNet.seg_classes[category]
            part = torch.tensor(part, device=device)

            y_map[part] = torch.arange(part.size(0), device=device)

            iou = multiclass_jaccard_index(out[:, part].argmax(dim=-1), y_map[y],
                                           num_classes=part.size(0))
            ious.append(iou)

        categories.append(data.category)

    iou = torch.tensor(ious, device=device)
    category = torch.cat(categories, dim=0)

    mean_iou = scatter(iou, category, reduce='mean')  # Per-category IoU.
    return float(mean_iou.mean())  # Global IoU.


def main():
    category = 'Airplane'  # Pass in `None` to train on all categories.
    path = "data/ShapeNet"
    transform = T.Compose([
        T.RandomJitter(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2)
    ])
    pre_transform = T.NormalizeScale()
    train_dataset = ShapeNet(path, category, split='trainval', transform=transform,
                             pre_transform=pre_transform)
    test_dataset = ShapeNet(path, category, split='test',
                            pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True,
                              num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False,
                             num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LDGCNNSegmentor(in_channels=6, out_channels=train_dataset.num_classes, k=30).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 31):
        train(train_loader, model, device, optimizer)
        iou = test(test_loader)
        print(f'Epoch: {epoch:02d}, Test IoU: {iou:.4f}')


if __name__ == '__main__':
    main()
