import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from utils import mnist, plot_graphs, plot_mnist, mnist_transform
import numpy as np
import matplotlib.pyplot as plt

from prepare_datasets import jet_dataset, jet_dataloader
from network import Net


def train(epoch, models, log=None):
    # train_size = len(train_loader.sampler)
    train_size = 10
    for batch_idx, (data_in, data_out) in enumerate(training_loader):
        for model in models.values():
            model.optim.zero_grad()
            output = model(data_in)
            loss = model.loss(output, data_in)
            loss.backward()
            model.optim.step()

        if batch_idx % 150 == 0:
            line = 'Train Epoch: {} [{:05d}/{}] '.format(
                epoch, batch_idx * len(data_in), train_size)
            losses = ' '.join(['{}: {:.6f}'.format(k, m._loss.item()) for k, m in models.items()])
            print(line + losses)

    else:
        batch_idx += 1
        line = 'Train Epoch: {} [{:05d}/{}] '.format(
            epoch, batch_idx * len(data_in), train_size)
        losses = ' '.join(['{}: {:.6f}'.format(k, m._loss.item()) for k, m in models.items()])
        if log is not None:
            for k in models:
                log[k].append(models[k]._loss)
        print(line + losses)



models = {'64-2': Net(64,2), '128-2': Net(128,2), '64-32': Net(64,32), '128-32': Net(128,32)}
train_log = {k: [] for k in models}
test_log = {k: [] for k in models}

#training_set = jet_dataset
training_loader = jet_dataloader

for epoch in range(1, 10):
    for model in models.values():
        model.train()
    train(epoch, models, train_log)
    #for model in models.values():
    #    model.eval()
    #test(models, test_loader, test_log)

