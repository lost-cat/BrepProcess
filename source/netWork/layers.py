import math

import torch
from torch import nn
from torch.nn import Module
from torch_geometric.nn import MessagePassing


class MyGCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index):
        self.propagate(edge_index, x=x)
        pass

    def message(self, x_j):
        print(x_j)
        return x_j


class GraphCNNGlobal(object):
    BN_DECAY = 0.999
    GRAPHCNN_INIT_FACTOR = 1.
    GRAPHCNN_I_FACTOR = 1.0


class GraphEmbeddingLayer(Module):
    """Graph embedding layer for summarizing learned information."""

    def __init__(self, in_features, out_features):
        super().__init__()

        self.num_filters = out_features
        self.weight_decay = 0.0005
        self.W = None
        self.b = None

        self.build(in_features)

    def build(self, in_features):
        self.W = nn.Parameter(torch.zeros(size=(in_features, self.num_filters), dtype=torch.float32))
        print('before', self.W)
        w_stddev = 1.0 / math.sqrt(in_features)
        nn.init.trunc_normal_(self.W, std=w_stddev)

        print('after', self.W)
        self.b = nn.Parameter(torch.zeros([self.num_filters], dtype=torch.float32))
        nn.init.constant_(self.b, 0.1)

    def forward(self, V):
        # mm or bmm?
        output = torch.mm(V, self.W) + self.b
        return output


class GraphEdgeConvLayer(nn.Module):
    """Graph convolutional layer that uses the edge convexity."""

    def __init__(self, in_features, out_features):
        super().__init__()

        self.num_filters = out_features

        self.weight_decay = 0.0005

        self.W_E_1 = None
        self.W_E_2 = None
        self.W_E_3 = None
        self.W_I = None
        self.b = None
        self.build(in_features)

    def build(self, in_features):
        num_features = in_features
        w_dim = [num_features, self.num_filters]
        b_dim = [self.num_filters]

        w_stddev = math.sqrt(1.0 / num_features * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)
        w_i_stddev = math.sqrt(
            GraphCNNGlobal.GRAPHCNN_I_FACTOR / num_features * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)

        self.W_E_1 = nn.Parameter(torch.randn(*w_dim, dtype=torch.float32))
        nn.init.trunc_normal_(self.W_E_1, std=w_stddev)
        self.W_E_2 = nn.Parameter(torch.randn(*w_dim, dtype=torch.float32))
        nn.init.trunc_normal_(self.W_E_2, std=w_stddev)

        self.W_E_3 = nn.Parameter(torch.randn(*w_dim, dtype=torch.float32))
        nn.init.trunc_normal_(self.W_E_3, std=w_stddev)
        self.W_I = nn.Parameter(torch.randn(*w_dim, dtype=torch.float32))
        nn.init.trunc_normal_(self.W_I, std=w_i_stddev)

        self.b = nn.Parameter(torch.randn(*b_dim))
        nn.init.constant_(self.b, 0.1)

    def forward(self, input):
        V, E_1, E_2, E_3 = input
        n_E_1 = torch.mm(E_1, V)
        n_E_2 = torch.mm(E_2, V)
        n_E_3 = torch.mm(E_3, V)

        output = (torch.mm(n_E_1, self.W_E_1) + torch.mm(n_E_2, self.W_E_2) +
                  torch.mm(n_E_3, self.W_E_3) + torch.mm(V, self.W_I) + self.b)
        return output


class GraphCNNLayer(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()

        self.in_feature = in_feature
        self.num_filters = out_feature
        self.weight_decay = 0.0005
        self.W = None
        self.W_I = None
        self.b = None

        self.build(in_feature)

    def build(self, input_shape):
        # num_features = (c)
        # num_nodes = (n)
        # num_filters = (j)
        # W_dim = (c x j)
        # W_I_dim = (c x j)
        # b_dim = (n x j)
        num_features = input_shape
        W_dim = [num_features, self.num_filters]
        W_I_dim = [num_features, self.num_filters]
        b_dim = [self.num_filters]

        W_stddev = math.sqrt(1.0 / num_features * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)
        W_I_stddev = math.sqrt(
            GraphCNNGlobal.GRAPHCNN_I_FACTOR / num_features * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)

        self.W = nn.Parameter(torch.empty(*W_dim, dtype=torch.float32))
        nn.init.trunc_normal_(self.W, std=W_stddev)
        self.W_I = nn.Parameter(torch.empty(*W_I_dim, dtype=torch.float32))
        nn.init.trunc_normal_(self.W_I, std=W_I_stddev)
        self.b = nn.Parameter(torch.empty(*b_dim, dtype=torch.float32))
        nn.init.constant_(self.b, 0.1)

    def forward(self, input):
        V, A = input
        n = torch.mm(A, V)
        # print(A)
        output = torch.mm(n, self.W) + torch.mm(V, self.W_I) + self.b

        return output


class TransferLayer(Module):
    """Transfer layer for passing learned information between graph levels."""

    def __init__(self, v_shape, v_aux_shape):
        super().__init__()
        self.W = None
        self.build(v_shape, v_aux_shape)

    def build(self, v_shape, v_aux_shape):
        w_dim = [v_shape, v_aux_shape]
        w_stddev = math.sqrt(1.0 / (v_shape * 2 * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR))

        self.W = nn.Parameter(torch.randn(*w_dim, dtype=torch.float32))
        nn.init.trunc_normal_(self.W, w_stddev)

    def forward(self, input):
        V, _, A_linkage = input
        n = torch.mm(A_linkage, V)
        output = torch.mm(n, self.W)
        return output
