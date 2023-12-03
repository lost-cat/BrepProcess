import datetime as dt

import torch.optim
from torch import nn
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional
from source.netWork.network_edge import HierarchicalCADNet
from source.dataProcess.dataLoader import dataloader_edge as dataloader


def train_step(x, y):
    model.train()
    y = y.cuda()
    with torch.set_grad_enabled(mode=True):
        optimizer.zero_grad()

        logits = model(x)

        loss_value = loss_fn(logits, y)
        loss_value.backward()
        optimizer.step()

    train_loss_metric(loss_value)
    train_acc_metric(logits, y)


def val_step(x, y):
    model.eval()

    y = y.cuda()
    with torch.set_grad_enabled(False):
        val_logits = model(x)
        loss_value = loss_fn(val_logits, y)

    val_loss_metric(loss_value)
    val_acc_metric(val_logits, y)


if __name__ == '__main__':
    import time

    device = torch.device('cuda')
    # User defined parameters.
    num_classes = 25
    num_layers = 6
    units = 512
    num_epochs = 100
    learning_rate = 1e-3
    dropout_rate = 0.0
    weight_decay = 0.0005
    data_path = 'data'
    train_set_path = data_path + "/training_MFCAD++.h5"
    val_set_path = data_path + "/val_MFCAD++.h5"
    save_name = (f'edge_lvl_{num_layers}_units_{units}_epochs_{num_epochs}_date_'
                 f'{dt.datetime.now().strftime("%Y-%m-%d")}')

    model = HierarchicalCADNet(5, units, dropout_rate, num_classes, num_layers)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.LRScheduler(optimizer)
    summary_writer = SummaryWriter(f'./log/{save_name}')

    train_loss_metric = torchmetrics.MeanMetric().to(device)
    train_acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
    val_loss_metric = torchmetrics.MeanMetric().to(device)
    val_acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)

    min_val_loss = 0.0
    min_train_loss = 0.0
    max_train_acc = 0.0
    max_val_acc = 0.0
    max_epoch = 0
    iterations = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} of {num_epochs}")
        start_time = time.time()
        train_dataloader = dataloader(train_set_path)
        val_dataloader = dataloader(val_set_path)

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataloader):
            # one_hot_y = torch.nn.functional.one_hot(y_batch_train.cuda(), num_classes=num_classes).cuda()
            # train_step(x_batch_train, one_hot_y)
            train_step(x_batch_train, y_batch_train)
            iterations += 1
            if step % 2 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(train_loss_metric.compute()))
                    , "Training acc at step %d: %.4f"
                    % (step, float(train_acc_metric.compute()))
                )

        train_loss = train_loss_metric.compute()
        train_acc = train_acc_metric.compute()
        summary_writer.add_scalar('train_loss', train_loss, iterations)
        summary_writer.add_scalar('train_acc', train_acc, iterations)

        train_loss_metric.reset()
        train_acc_metric.reset()
        print(f"Train loss={train_loss}, Train acc={train_acc}")

        # Run validation loop
        for x_batch_val, y_batch_val in val_dataloader:
            val_step(x_batch_val, y_batch_val)

        val_loss = val_loss_metric.compute()
        val_acc = val_acc_metric.compute()

        if val_acc > max_val_acc:
            min_val_loss = float(val_loss)
            min_train_loss = float(train_loss)
            max_train_acc = float(train_acc)
            max_val_acc = float(val_acc)
            optimizer.state_dict()
            torch.save(model.state_dict(), f"checkpoint/{save_name}.ckpt")
            max_epoch = epoch

        summary_writer.add_scalar('val_loss', val_loss, iterations)
        summary_writer.add_scalar('val_acc', val_acc, iterations)
        val_acc_metric.reset()
        val_loss_metric.reset()

        print(f"Val loss={val_loss}, Val acc={val_acc}")
        print("Time taken: %.2fs" % (time.time() - start_time))
    summary_writer.close()
    print(f"Epoch={max_epoch + 1}, Max train acc={max_train_acc}, Max val acc={max_val_acc}")
    print(f"Train loss={min_train_loss}, Val loss={min_val_loss}")
