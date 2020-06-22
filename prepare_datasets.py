import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import uproot
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/klapidus/emd')

import random

import jetutils

import utils

import energyflow

_mnumber = 1000

def norm_hist_to_max(hist):
    max_ = np.amax(hist)
    if max_ > 0:
        h = np.divide(hist, max_)
        h = np.multiply(hist, 2.0)
        return np.subtract(h, 1.0)
    else:
        return hist

def get_jets(vals, rotate=True):
  jets = []
  start = 0
  for i in vals[3]:
      if i > 0 and i < _mnumber:
          jet_pt = vals[0,start]
          #print("jet pt = ",jet_pt)
          jet = vals[0:3,start+1:start+1+int(i)]
          start = start + 1 + int(i)
          jet = jet[:, (jet[0,:] > 0) & (jet[0,:] < _mnumber)]
          jet = jet.transpose()
          yphi_avg = np.average(jet[:, 1:3], weights=jet[:, 0], axis=0)
          jet[:,1:3] -= yphi_avg
          #normalization here
          #print("const pt before = ", jet1[:,0])
          #print('jet pt',jet_pt)
          # if jet_pt < 90 or jet_pt > 110:
          #    print('error? jet pt =', jet_pt)
          jet[:, 0] /= jet_pt
          # const_max_pt = np.amax(jet[:, 0])
          # jet[:, 0] /= const_max_pt
          #print("const pt after = ", jet1[:,0])
          #proper jet rotation
          #please note: this is not needed, when
          #comparing PL->DL->DLUEBS
          if rotate is True:
              jetutils.align_jet_pc_to_pos_phi(jet)
          #print(jet)
          if jet_pt > 60 and jet_pt < 80:
              jets.append(jet)
          # if len(jets) > 100:
          #     break
  return jets


class JetPicturesDataset(Dataset):
    def __init__(self, jets_in, jets_out):
        self.jets_in = jets_in
        self.jets_out = jets_out

    def __len__(self):
        return len(self.jets_in)

    def __getitem__(self, idx):
        sample = {'pl': self.jets_in[idx], 'dl': self.jets_out[idx]}
        return sample


#if __name__ == '__main__':
# generate placeholder jets

# file = uproot.open("./tree_PbPb_sim.root")
# treePL = file["treePL"]
# treeDL = file["treeDL"]
# eventsPL = treePL.arrays(["pt", "eta", "phi", "jetm"])
# eventsDL = treeDL.arrays(["pt", "eta", "phi", "jetm"])
# valsPL = np.array( list( eventsPL.values() ) )
# valsDL = np.array( list( eventsDL.values() ) )

# _jetsPL = get_jets(valsPL)
# _jetsDL = get_jets(valsDL)

# print('PL size = ', len(_jetsPL))
# print('DL size = ', len(_jetsDL))

# np.save('jet_input.npy', _jetsPL)
_jetsPL = np.load('jet_input.npy', allow_pickle=True)
# _jetsPL = _jetsPL[:2000]
print(len(_jetsPL))

BINS = np.linspace(-0.4, 0.4, num=utils.N_IMAGE_BINS+1)
#print(len(bins))

jetsPL_train = []
# jetsDL_train = []
jetsPL_test = []
# jetsDL_test = []

jet_images_pl_test = []
# jet_images_dl_test = []

efpset = energyflow.EFPSet(('d<=', 4), measure='hadr', beta=0.5)
# masked_X = [x[x[:,0] > 0] for x in _jetsPL]
# X = efpset.compute(_jetsPL[3])
#print(len(X))
# print(X.shape)
efps_pl = efpset.batch_compute(_jetsPL, n_jobs=2)[:, 1:]

normalization = np.max(efps_pl, axis=0)
print(np.max(efps_pl, axis=0))
efps_pl = np.divide(efps_pl, normalization)
print(np.max(efps_pl, axis=0))

efps_pl_train = []
efps_pl_test = []
for efp in efps_pl:

    efp = np.multiply(efp, 2.0)
    efp = np.subtract(efp, 1.0)

    if np.random.randint(3) <= 1:
        jetsPL_train.append(torch.from_numpy(efp.astype(np.float32)))
        # jetsDL_train.append(torch.from_numpy(efp.astype(np.float32)))
        efps_pl_train.append(efp)
    else:
        jetsPL_test.append(torch.from_numpy(efp.astype(np.float32)))
        # jetsDL_test.append(torch.from_numpy(efp.astype(np.float32)))
        efps_pl_test.append(efp)

# eta phi pt
#for j in range(0, 10000):
    #nconst = np.random.randint(1, 40)
    #nconst = 40
    #jet = np.random.rand(nconst, 3)
    #jet_torch = torch.from_numpy(jet)
    #jet_torch = jet_torch.view(120)
    #print(jet_torch.shape)
    #placeholder
    #jet_torch = torch.randn(120)
    #jet = np.random.rand(120)
    #jet_torch = torch.from_numpy(jet.astype(np.float32))

    #distortion = torch.empty(jet_torch.size()).normal_(mean=0.0, std=0.5)
    #jet_mod = jet_torch + distortion

    #jets.append(jet_torch)
    #jets_output.append(jet_mod)


#jets_output = []
# for j in range(0, 10000):
#     jet_torch = torch.randn(120)
#     jets_output.append(jet_torch)


#prepare jet pictures with given binning
#vector of N*N pt (energy) values

#print('len jetsPL_train', len(jetsPL_train))
#print('len jetsPL_test', len(jetsPL_test))

jet_dataset_train = JetPicturesDataset(jetsPL_train, jetsPL_train)
jet_dataset_test = JetPicturesDataset(jetsPL_test, jetsPL_test)
#print(len(jet_dataset))
#print(jet_dataset[100])
#print(jet_dataset[122:361])

jet_dataloader_train = DataLoader(jet_dataset_train, batch_size=400, shuffle=True, num_workers=2)
jet_dataloader_test = DataLoader(jet_dataset_test, batch_size=400, shuffle=True, num_workers=2)

#print(jet_dataloader_test.__len__())
#jet_dataloader_output = DataLoader(jet_dataset, batch_size=2, shuffle=True, num_workers=2)
#or i, batch in enumerate(dataloader):
  #  print(i, batch)
  #  if i == 3:
   #     break