import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils


class Encoder(nn.Module):
    def __init__(self, latent_size=32, act=torch.sigmoid):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(utils.N_IMAGE_BINS*utils.N_IMAGE_BINS, latent_size)
        self.act = act

    def forward(self, x):
        latent = self.act(self.fc1(x))
        return latent


class Decoder(nn.Module):
    def __init__(self, latent_size=32, act=torch.sigmoid):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, utils.N_IMAGE_BINS*utils.N_IMAGE_BINS)
        self.act = act

    def forward(self, x):
        output = torch.tanh(self.fc1(x))
        # output = torch.relu(self.fc1(x))
        return output


class Net(nn.Module):
        def __init__(self, latent_size=32, loss_fn=F.mse_loss, lr=1.e-3, l2=0.0):
            super(Net, self).__init__()
            self.E = Encoder(latent_size)
            self.D = Decoder(latent_size)
            self.loss_fn = loss_fn
            self._loss = None
            # self.optim = optim.SGD(self.parameters(), lr=lr)
            self.optim = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        def forward(self, x):
            x = x.view(-1, utils.N_IMAGE_BINS*utils.N_IMAGE_BINS)
            h = self.E(x)
            out = self.D(h)
            return out

        def decode(self, h):
            with torch.no_grad():
                return self.D(h)

        def loss(self, x, target, **kwargs):
            target = target.view(-1, utils.N_IMAGE_BINS*utils.N_IMAGE_BINS)
            self._loss = self.loss_fn(x, target, **kwargs)
            return self._loss