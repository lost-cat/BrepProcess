import torch
import torch.nn.functional
from torch import nn

from source.netWork.layers import GraphEmbeddingLayer, GraphEdgeConvLayer, TransferLayer, GraphCNNLayer


class HierarchicalCADNet(nn.Module):

    def __init__(self, in_features, units, dropout_rate, num_classes, num_layers=7):
        super().__init__()
        self.num_layers = num_layers
        self.ge_start = GraphEmbeddingLayer(in_features, units)
        self.bn_start = torch.nn.BatchNorm1d(units)
        self.dp_start = torch.nn.Dropout(dropout_rate)

        for i in range(1, self.num_layers + 1):
            setattr(self, f"gcnn_1_{i}", GraphEdgeConvLayer(units, units))
            setattr(self, f"bn_1_{i}", torch.nn.BatchNorm1d(units))
            setattr(self, f"dp_1_{i}", torch.nn.Dropout(dropout_rate))

        for i in range(1, self.num_layers + 1):
            if i == 1:
                setattr(self, f"gcnn_2_{i}", GraphCNNLayer(4, units))
            else:
                setattr(self, f"gcnn_2_{i}", GraphCNNLayer(units, units))

            setattr(self, f"bn_2_{i}", torch.nn.BatchNorm1d(units))
            setattr(self, f"dp_2_{i}", torch.nn.Dropout(dropout_rate))

        self.ge_1 = GraphEmbeddingLayer(units, units)
        self.bn_1 = torch.nn.BatchNorm1d(units)
        self.dp_1 = torch.nn.Dropout(dropout_rate)

        self.ge_2 = GraphEmbeddingLayer(units, units)
        self.bn_2 = torch.nn.BatchNorm1d(units)
        self.dp_2 = torch.nn.Dropout(dropout_rate)

        # Transfer Layers
        self.a3 = TransferLayer(units, units)
        self.bn_a3 = torch.nn.BatchNorm1d(units)
        self.a4 = TransferLayer(units, 4)
        self.bn_a4 = torch.nn.BatchNorm1d(4)

        # Level 1 - Final (Block 5)
        self.ge_final = GraphEmbeddingLayer(units, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data):
        V_1, E_1, E_2, E_3, V_2, A_2, A_3 = data

        V_1 = V_1.cuda()
        V_2 = V_2.cuda()

        E_1 = E_1.cuda()
        E_2 = E_2.cuda()
        E_3 = E_3.cuda()

        A_2 = A_2.cuda()
        A_3 = A_3.cuda()

        x_1 = self.ge_start(V_1)
        x_1 = self.bn_start(x_1)
        x_1 = torch.nn.functional.relu(x_1)
        x_1 = self.dp_start(x_1)

        a_4 = self.a4([x_1, V_2, A_3])
        a_4 = self.bn_a4(a_4)
        a_4 = torch.nn.functional.relu(a_4)
        x_2 = V_2 + a_4

        # print("X2", x_2.shape)
        for i in range(1, self.num_layers + 1):
            r_2 = getattr(self, f"gcnn_2_{i}")([x_2, A_2])
            r_2 = getattr(self, f"bn_2_{i}")(r_2)
            r_2 = torch.nn.functional.relu(r_2)
            r_2 = getattr(self, f"dp_2_{i}")(r_2)

            if i == 1:
                x_2 = r_2
            else:
                x_2 = x_2 + r_2

        x_2 = self.ge_2(x_2)
        x_2 = self.bn_2(x_2)
        x_2 = torch.nn.functional.relu(x_2)
        x_2 = self.dp_2(x_2)

        # A3 => Embedding from Level 2 to Level 1
        a_3 = self.a3([x_2, x_1, torch.transpose(A_3, 0, 1)])
        a_3 = self.bn_a3(a_3)
        a_3 = torch.nn.functional.relu(a_3)
        x_1 = x_1 + a_3

        for i in range(1, self.num_layers + 1):
            r_1 = getattr(self, f"gcnn_1_{i}")([x_1, E_1, E_2, E_3])
            r_1 = getattr(self, f"bn_1_{i}")(r_1)
            r_1 = torch.nn.functional.relu(r_1)
            r_1 = getattr(self, f"dp_1_{i}")(r_1)
            x_1 = x_1 + r_1

        x_1 = self.ge_1(x_1)
        x_1 = self.bn_1(x_1)
        x_1 = torch.nn.functional.relu(x_1)
        x_1 = self.dp_1(x_1)

        # Final 0-hop layer
        x = self.ge_final(x_1)
        x = self.softmax(x)

        return x
