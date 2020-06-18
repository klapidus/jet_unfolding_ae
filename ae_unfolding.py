import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from utils import mnist, plot_graphs, plot_mnist, mnist_transform
import numpy as np
import matplotlib.pyplot as plt

from prepare_datasets import jet_dataloader_train, jet_dataloader_test, jetsPL_test, jetsDL_test
from prepare_datasets import jet_images_pl_test, jet_images_dl_test, BINS

import network
import network_toy

import utils

# def unfold_images(jet_images):
#     for jetim in jet_images:
#         torch.from_numpy(h_dl.astype(np.float32))
#     torch_inputs = []

def dR(x,y):
    return np.sqrt(x*x + y*y)

def calcX(jetimage):

    jetimage = np.add(jetimage, 1.0)
    jetimage = np.divide(jetimage, 2.0)

    X = 0.0
    norm = 0.0
    for ix, xedge in enumerate(BINS):
        for iy, yedge in enumerate(BINS):
            if ix > utils.N_IMAGE_BINS - 1:
                continue
            if iy > utils.N_IMAGE_BINS - 1:
                continue
            xcenter = xedge + utils.BIN_WIDTH
            ycenter = yedge + utils.BIN_WIDTH
            X += jetimage[ix, iy] * dR(xcenter, ycenter)
            norm += jetimage[ix, iy]
    if norm > 0.0:
        return X/norm
    else:
        return 0


def train(epoch, models, log=None):
    #train_size = training_loader.__len__()
    train_size = 400
    for batch_idx, sample_batched in enumerate(training_loader):
        # print('batch_idx', batch_idx)
        for model in models.values():
            model.optim.zero_grad()
            # output = model(data_out)
            output = model(sample_batched['pl'])
            # print('data_in', sample_batched['pl'][10].view(1,16))
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
    test_size = 5560
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

net = network.Net
# net = network_toy.Net

models = {'64-16': net(64,16), '256-128': net(256,128), '64-32': net(64,32), '128-32': net(128,32)}
# models = {'64-2': net(64,2), '128-2': net(128,2), '64-32': net(64,32), '128-32': net(128,32)}
# models.update({'64-4': net(64,4), '128-4': net(128,4), '64-8': net(64,8), '128-8': net(128,8)})

# models = {'32': net(32), '64': net(64), '128': net(128)}
# models = {'4': net(4)}
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

# for epoch in range(1, 20):
#     decoder_output(models, test_loader, test_log)

#bins = np.linspace(-0.4, 0.4, num=utils.N_IMAGE_BINS+1)

#model = models['128-2']
#jetDL = jetsDL_test[200]
#h_dl, _, _ = np.histogram2d(jetDL[:, 1], jetDL[:, 2], bins=bins, weights=jetDL[:, 0])
#jetDL = torch.from_numpy(h_pl.astype(np.float32))
#print( model(jetDL) )

x_pl = [calcX(jetim) for jetim in jet_images_pl_test]
x_dl = [calcX(jetim) for jetim in jet_images_dl_test]

rec_jet = []
x_pl_rec = []
# for k, m in models.items():
model = models['256-128']
for jet in jetsPL_test:
    with torch.no_grad():
        # print(model(jet).detach().numpy().reshape((utils.N_IMAGE_BINS, utils.N_IMAGE_BINS)))
        image = model(jet).detach().numpy().reshape((utils.N_IMAGE_BINS, utils.N_IMAGE_BINS))
        x_pl_rec.append(calcX(image))
        # print(jet)
        # print(image)
        # print(calcX(jet))
        # print(calcX(image))

hist_x_pl, _ = np.histogram(x_pl)
hist_x_dl, _ = np.histogram(x_dl)
hist_x_pl_rec, _ = np.histogram(x_pl_rec)

fig = plt.figure(figsize=(12.0, 4.0))
# fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.hist(x_pl, bins=40)
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, 0.0, y2))

ax = fig.add_subplot(1, 2, 2)
plt.hist(x_pl_rec, bins=40)
plt.axis((x1, x2, 0.0, y2))

# ax = fig.add_subplot(1, 3, 3)
# plt.hist(x_dl_unfolded)

plt.show()