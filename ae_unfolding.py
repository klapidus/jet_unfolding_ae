import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from utils import mnist, plot_graphs, plot_mnist, mnist_transform
import numpy as np
import matplotlib.pyplot as plt

from prepare_datasets import jet_dataloader_train, jet_dataloader_test, jetsPL_test, jetsDL_test, efps_pl_test
from prepare_datasets import jet_images_pl_test, jet_images_dl_test, BINS
from prepare_datasets import normalization

import network
import network_toy

import utils

def train(epoch, models, log=None):
    #train_size = training_loader.__len__()
    train_size = 400
    for batch_idx, sample_batched in enumerate(training_loader):
        # print('batch_idx', batch_idx)
        for model in models.values():
            model.optim.zero_grad()
            # output = model(data_out)
            output = model(sample_batched['pl'])
            # print('data_in', sample_batched['pl'][10])
            # print('data_out', output[10])
            loss = model.loss(output, sample_batched['pl'])
            loss.backward()
            model.optim.step()

        if batch_idx % 150 == 0:
            line = 'Train Epoch: {} [{:05d}/{}] '.format(
                epoch, batch_idx * len(sample_batched['pl']), train_size)
            losses = ' '.join(['{}: {:.6f}'.format(k, m._loss.item()) for k, m in models.items()])
            print(line + losses)

    else:
        batch_idx += 1
        line = 'Train Epoch: {} [{:05d}/{}] '.format(
            epoch, batch_idx * len(sample_batched['pl']), train_size)
        losses = ' '.join(['{}: {:.6f}'.format(k, m._loss.item()) for k, m in models.items()])
        if log is not None:
            for k in models:
                log[k].append(models[k]._loss)
        print(line + losses)


avg_lambda = lambda l: 'loss: {:.4f}'.format(l)
line = lambda i, l: '{}: '.format(i) + avg_lambda(l)


def test(models, loader, log=None):
    test_size = len(jetsPL_test)
    #test_size = loader.__len__()
    #print(test_size)
    test_loss = {k: 0. for k in models}
    with torch.no_grad():
        for sample_batched in loader:
            #output = {k: m(data_out) for k, m in models.items()}
            output = {k: m(sample_batched['pl']) for k, m in models.items()}
            for k, m in models.items():
                test_loss[k] += m.loss(output[k], sample_batched['pl'], reduction='sum').item()  # sum up batch loss

    for k in models:
        test_loss[k] /= (test_size * utils.N_IMAGE_BINS * utils.N_IMAGE_BINS)
        if log is not None:
            log[k].append(test_loss[k])

    lines = '\n'.join([line(k, test_loss[k]) for k in models]) + '\n'
    report = 'Test set:\n' + lines
    print(report)

# net = network.Net
net = network_toy.Net

# models = {'32-16': net(32,16), '32-8': net(32,8), '32-4': net(32,4)}
# models = {'64-2': net(64,2), '128-2': net(128,2), '64-32': net(64,32), '128-32': net(128,32)}
# models.update({'64-4': net(64,4), '128-4': net(128,4), '64-8': net(64,8), '128-8': net(128,8)})

models = {'32': net(32), '16': net(16), '8': net(8), '4': net(4), '2': net(2)}
# models = {'4': net(4)}
train_log = {k: [] for k in models}
test_log = {k: [] for k in models}

training_loader = jet_dataloader_train
test_loader = jet_dataloader_test

for epoch in range(1, 20):
    for model in models.values():
        model.train()
    train(epoch, models, train_log)
    for model in models.values():
       model.eval()
    test(models, test_loader, test_log)


rec_jet = []
efps_rec = []
# for k, m in models.items():
# model = models['128']
model = models['2']
# for efp in jetsPL_test:
#     with torch.no_grad():
        # print(model(jet).detach().numpy().reshape((utils.N_IMAGE_BINS, utils.N_IMAGE_BINS)))
        # efp_rec = model(efp).detach().numpy().reshape(utils.N_EFP)
        # efps_rec.append(efp_rec)

tensors = torch.stack(jetsPL_test)
with torch.no_grad():
    #print(model(jet).detach().numpy().reshape((utils.N_IMAGE_BINS, utils.N_IMAGE_BINS)))
    efps_rec = model(tensors).detach().numpy().reshape(len(jetsPL_test), utils.N_EFP)
    #efps_rec.append(efp_rec)

efps_pl_test = np.add(efps_pl_test, 1.0)
efps_pl_test = np.divide(efps_pl_test, 2.0)
efps_pl_test = np.multiply(efps_pl_test, normalization)

efps_rec = np.add(efps_rec, 1.0)
efps_rec = np.divide(efps_rec, 2.0)
efps_rec = np.multiply(efps_pl_test, efps_rec)

# print(efps_pl_test.shape)
# print(efps_rec.shape)

fig = plt.figure(figsize=(12.0, 12.0))
# fig = plt.figure()
for idx in range(0, utils.N_EFP):
    histos = (efps_pl_test[:, idx], efps_rec[:, idx])
    # print(efps_pl_test[:, idx].shape)
    ax = fig.add_subplot(6, 6, idx+1)
    # bins = np.linspace(0.0, 0.2, 20)
    # plt.hist(histos, bins=10, range=(0.0, 0.2), label=('INPUT','OUTPUT'), alpha=0.8, color=('g','b'))
    plt.hist(histos, label=('INPUT', 'OUTPUT'), alpha=0.9, color=('b', 'r'))
    # x1, x2, y1, y2 = plt.axis()
    # ax = fig.add_subplot(6, 6, idx + 1)
    # plt.hist(efps_rec[:][idx], bins=40, range=(0.0, 1.0), label='OUTPUT')
    # plt.axis((0.0, 0.05, 0.0, y2))
plt.show()

# for epoch in range(1, 20):
#     decoder_output(models, test_loader, test_log)

#bins = np.linspace(-0.4, 0.4, num=utils.N_IMAGE_BINS+1)

#model = models['128-2']
#jetDL = jetsDL_test[200]
#h_dl, _, _ = np.histogram2d(jetDL[:, 1], jetDL[:, 2], bins=bins, weights=jetDL[:, 0])
#jetDL = torch.from_numpy(h_pl.astype(np.float32))
#print( model(jetDL) )