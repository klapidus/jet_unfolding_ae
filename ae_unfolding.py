import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from utils import mnist, plot_graphs, plot_mnist, mnist_transform
import numpy as np
import matplotlib.pyplot as plt

from prepare_datasets import jet_dataloader_train, jet_dataloader_test, jetsPL_test, jetsDL_test
from network import Net

import utils


def train(epoch, models, log=None):
    #train_size = training_loader.__len__()
    train_size = 400
    for batch_idx, (data_in, data_out) in enumerate(training_loader):
        for model in models.values():
            model.optim.zero_grad()
            output = model(data_out)
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


avg_lambda = lambda l: 'loss: {:.4f}'.format(l)
line = lambda i, l: '{}: '.format(i) + avg_lambda(l)

def test(models, loader, log=None):
    test_size = 5560
    #test_size = loader.__len__()
    #print(test_size)
    test_loss = {k: 0. for k in models}
    with torch.no_grad():
        for (data_in, data_out) in loader:
            output = {k: m(data_out) for k, m in models.items()}
            for k, m in models.items():
                test_loss[k] += m.loss(output[k], data_in, reduction='sum').item()  # sum up batch loss

    for k in models:
        test_loss[k] /= (test_size * utils.N_IMAGE_BINS * utils.N_IMAGE_BINS)
        if log is not None:
            log[k].append(test_loss[k])

    lines = '\n'.join([line(k, test_loss[k]) for k in models]) + '\n'
    report = 'Test set:\n' + lines
    print(report)

models = {'64-2': Net(64,2), '128-2': Net(128,2), '64-32': Net(64,32), '128-32': Net(128,32)}
train_log = {k: [] for k in models}
test_log = {k: [] for k in models}

training_loader = jet_dataloader_train
test_loader = jet_dataloader_test

for epoch in range(1, 10):
    for model in models.values():
        model.train()
    train(epoch, models, train_log)
    for model in models.values():
       model.eval()
    test(models, test_loader, test_log)

#bins = np.linspace(-0.4, 0.4, num=utils.N_IMAGE_BINS+1)

#model = models['128-2']
#jetDL = jetsDL_test[200]
#h_dl, _, _ = np.histogram2d(jetDL[:, 1], jetDL[:, 2], bins=bins, weights=jetDL[:, 0])
#jetDL = torch.from_numpy(h_pl.astype(np.float32))
#print( model(jetDL) )


