import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np


class JetPicturesDataset(Dataset):
    def __init__(self, jets_in, jets_out):
        self.jets_in = jets_in
        self.jets_out = jets_out

    def __len__(self):
        return len(self.jets_in)

    def __getitem__(self, idx):
        return self.jets_in[idx], self.jets_out[idx]


#if __name__ == '__main__':
# generate placeholder jets
jets = []
# eta phi pt
for j in range(0, 10000):
    #nconst = np.random.randint(1, 40)
    nconst = 40
    jet = np.random.rand(nconst, 3)
    jet_torch = torch.from_numpy(jet)
    #jet_torch = jet_torch.view(120)
    #print(jet_torch.shape)
    #placeholder
    jet_torch = torch.randn(120)
    jets.append(jet_torch)


jets_output = []
for j in range(0, 10000):
    jet_torch = torch.randn(120)
    jets_output.append(jet_torch)


#prepare jet pictures with given binning
#vector of N*N pt (energy) values

jet_dataset = JetPicturesDataset(jets, jets_output)
#print(len(jet_dataset))
#print(jet_dataset[100])
#print(jet_dataset[122:361])

jet_dataloader = DataLoader(jet_dataset, batch_size=4, shuffle=True, num_workers=2)
#jet_dataloader_output = DataLoader(jet_dataset, batch_size=2, shuffle=True, num_workers=2)
#or i, batch in enumerate(dataloader):
  #  print(i, batch)
  #  if i == 3:
   #     break