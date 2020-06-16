import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, interm_size=32, latent_size=32, act=torch.relu):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(120, interm_size)
        self.fc2 = nn.Linear(interm_size, latent_size)
        self.act = act

    def forward(self, x):
        interm = self.act(self.fc1(x))
        latent = self.act(self.fc2(interm))
        return latent


class Decoder(nn.Module):
    def __init__(self, interm_size=32, latent_size=32, act=torch.relu):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, interm_size)
        self.fc2 = nn.Linear(interm_size, 120)
        self.act = act

    def forward(self, x):
        interm = self.act(self.fc1(x))
        output = torch.tanh(self.fc2(interm))
        return output


class Net(nn.Module):
        def __init__(self, interm_size=32, latent_size=32, loss_fn=F.mse_loss, lr=1e-4, l2=0.):
            super(Net, self).__init__()
            self.E = Encoder(interm_size, latent_size)
            self.D = Decoder(interm_size, latent_size)
            self.loss_fn = loss_fn
            self._loss = None
            self.optim = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

        def forward(self, x):
            x = x.view(-1, 120)
            h = self.E(x)
            out = self.D(h)
            return out

        def decode(self, h):
            with torch.no_grad():
                return self.D(h)

        def loss(self, x, target, **kwargs):
            target = target.view(-1, 120)
            self._loss = self.loss_fn(x, target, **kwargs)
            return self._loss