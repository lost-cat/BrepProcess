import numpy as np
import torch
import torchmetrics
from torch import nn
import time
from torch.nn import functional
from source.analysis import analysis_report_mfcadplus
from source.dataProcess.dataLoader import dataloader_edge as dataloader

from source.netWork.network_edge import HierarchicalCADNet


def test_step(x, y):
    model.eval()

    y = y.cuda()
    with torch.set_grad_enabled(False):
        test_logits = model(x)
        loss_value = loss_fn(test_logits, y)

    # one_hot_y = torch.nn.functional.one_hot(y, num_classes=num_classes)
    # y_true = torch.argmax(one_hot_y, 1)
    y_true = y
    y_pred = torch.argmax(test_logits, 1)

    test_loss_metric.update(loss_value)
    test_acc_metric.update(test_logits, y)
    test_precision_metric.update(test_logits, y)
    test_recall_metric.update(test_logits, y)

    return y_true, y_pred


if __name__ == '__main__':

    device = torch.device('cuda')
    num_classes = 25
    num_layers = 6
    units = 512
    learning_rate = 1e-4
    dropout_rate = 0.3
    checkpoint_path = "checkpoint/edge_lvl_6_units_512_epochs_100_date_2023-11-19.ckpt"
    data_path = 'data'

    test_set_path = data_path + "/val_MFCAD++.h5"

    model = HierarchicalCADNet(units=units, dropout_rate=dropout_rate, num_classes=num_classes,
                               num_layers=num_layers, in_features=5)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0005)

    test_loss_metric = torchmetrics.MeanMetric().to(device)
    test_acc_metric = torchmetrics.Accuracy("multiclass", num_classes=num_classes).to(device)
    test_precision_metric = torchmetrics.Precision("multiclass", num_classes=num_classes).to(device)
    test_recall_metric = torchmetrics.Recall('multiclass', num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model = model.cuda()
    test_dataloader = dataloader(test_set_path)
    y_true_total = []
    y_pred_total = []

    start_time = time.time()
    for x_batch_test, y_batch_test in test_dataloader:
        # one_hot_y = torch.nn.functional.one_hot(y_batch_test.cuda(), num_classes=num_classes).cuda()
        y_true, y_pred = test_step(x_batch_test, y_batch_test)

        y_true_total = np.append(y_true_total, y_true.cpu())
        y_pred_total = np.append(y_pred_total, y_pred.cpu())
    print("Time taken: %.2fs" % (time.time() - start_time))
    analysis_report_mfcadplus(y_true_total, y_pred_total)
    test_loss = test_loss_metric.compute()
    test_acc = test_acc_metric.compute()
    test_precision = test_precision_metric.compute()
    test_recall = test_recall_metric.compute()

    test_loss_metric.reset()
    test_acc_metric.reset()
    test_precision_metric.reset()
    test_recall_metric.reset()
    print(f"Test loss={test_loss}, Test acc={test_acc}, Precision={test_precision}, Recall={test_recall}")
