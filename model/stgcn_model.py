"""
ST-GCN encoder + FC decoder, reimplemented to match OpenHands (AI4Bharat/OpenHands) exactly:
  - openhands/models/encoder/graph/st_gcn.py  (ConvTemporalGraphical, STGCN_BLOCK, STGCN)
  - openhands/models/decoder/fc.py            (FC)
  - openhands/models/network.py               (Network: encoder + decoder wrapper)

Module names/attribute names are kept identical to OpenHands (`st_gcn_networks`, `gcn`, `tcn`,
`residual`, `edge_importance`, `data_bn`, `classifier`, ...) so that a checkpoint trained with
OpenHands' code would load here unchanged.

Note on input shape: the classic ST-GCN paper (Yan et al. 2018) uses (N, C, T, V, M) with an
explicit person-count axis M. OpenHands drops M entirely (single signer) and its STGCN.forward
takes (N, C, T, V). This module follows OpenHands (4D) since matching its pretrained weights is
the hard constraint; `STGCN.forward` accepts an optional (N, C, T, V, M) input and squeezes a
trailing M=1 axis for convenience/spec-compatibility, but M must be 1.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import Graph


class ConvTemporalGraphical(nn.Module):
    """
    Applies the graph convolution: a 1x1(-ish) temporal conv followed by contraction against
    the (K, V, V) adjacency tensor.

    Shapes:
        Input:  x (N, in_channels, T, V), A (K, V, V)
        Output: (N, out_channels, T, V), A (K, V, V)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))
        return x.contiguous(), A


class STGCNBlock(nn.Module):
    """
    One spatial-temporal graph conv block: spatial graph conv -> BN+ReLU -> temporal conv ->
    BN(+dropout), plus a residual connection, followed by a final ReLU.

    Shapes:
        Input:  x (N, in_channels, T_in, V), A (K, V, V)
        Output: (N, out_channels, T_out, V), A (K, V, V)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class STGCN(nn.Module):
    """
    Spatial-temporal graph convolutional network backbone (Yan et al. 2018), as used by
    OpenHands for pose-based sign language recognition.

    10 stacked STGCNBlocks, channel progression 64,64,64,64,128,128,128,256,256,256 with
    temporal stride-2 downsampling at the two channel-increase transitions (block 4 and block
    7, 0-indexed), followed by global average pooling over (T, V).

    Args:
        in_channels (int): number of input coordinate channels (2 for x,y or 3 for x,y,z).
        graph_args (dict): kwargs for `Graph` (num_nodes, center, inward_edges, ...).
        edge_importance_weighting (bool): learnable per-block, per-edge importance weights.
        n_out_features (int): output embedding dimension. Default 256.
    """

    def __init__(self, in_channels, graph_args, edge_importance_weighting=True, n_out_features=256, **kwargs):
        super().__init__()

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        self.n_out_features = n_out_features
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != "dropout"}
        self.st_gcn_networks = nn.ModuleList(
            (
                STGCNBlock(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
                STGCNBlock(64, 64, kernel_size, 1, **kwargs),
                STGCNBlock(64, 64, kernel_size, 1, **kwargs),
                STGCNBlock(64, 64, kernel_size, 1, **kwargs),
                STGCNBlock(64, 128, kernel_size, 2, **kwargs),
                STGCNBlock(128, 128, kernel_size, 1, **kwargs),
                STGCNBlock(128, 128, kernel_size, 1, **kwargs),
                STGCNBlock(128, 256, kernel_size, 2, **kwargs),
                STGCNBlock(256, 256, kernel_size, 1, **kwargs),
                STGCNBlock(256, self.n_out_features, kernel_size, 1, **kwargs),
            )
        )

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):
        """
        Args:
            x: (N, C, T, V) tensor, or (N, C, T, V, 1) (trailing person-axis M=1 is squeezed).

        Returns:
            (N, n_out_features) embedding.
        """
        if x.dim() == 5:
            assert x.size(4) == 1, "STGCN only supports a single person (M=1)"
            x = x.squeeze(4)

        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # N C T V -> N V C T
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()  # N V C T -> N C T V

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        return x.view(N, -1)


class FC(nn.Module):
    """Fully-connected classifier head, matching openhands/models/decoder/fc.py."""

    def __init__(self, n_features, num_class, dropout_ratio=0.2, batch_norm=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.bn = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(n_features)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        self.classifier = nn.Linear(n_features, num_class)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2.0 / num_class))

    def forward(self, x):
        x = self.dropout(x)
        if self.bn:
            x = self.bn(x)
        return self.classifier(x)


class Network(nn.Module):
    """Encoder + decoder wrapper, matching openhands/models/network.py."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


def build_stgcn(num_class, graph, in_channels=2, dropout_ratio=0.05):
    """
    The ST-GCN model of OpenHands' `examples/configs/wlasl/st_gcn.yaml`, over the given skeleton
    graph (one of the GRAPH_* dicts in pose_config.py) with `num_class` outputs.
    """
    encoder = STGCN(in_channels=in_channels, graph_args=graph, edge_importance_weighting=True)
    decoder = FC(n_features=encoder.n_out_features, num_class=num_class, dropout_ratio=dropout_ratio)
    return Network(encoder, decoder)
