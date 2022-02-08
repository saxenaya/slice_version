import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import os

import argparse
parser = argparse.ArgumentParser()


parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--trajectories', type=str, nargs='+', required=True)
parser.add_argument('--embedding_ckpt_dir', type=str, required=True)
parser.add_argument('--ckpt_dir', type=str, required=True)
parser.add_argument('--embedding_dim', type=int, required=True)
parser.add_argument('--embedding_epoch', type=int, required=True)

args = parser.parse_args()

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

# Set up data loaders
from datasets import PathPreferenceDecisionDataset
pref_datasets = []
for traj_id in args.trajectories:
    pref_datasets.append(PathPreferenceDecisionDataset(os.path.join(args.root_dir, traj_id)))
pref_dataset = torch.utils.data.ConcatDataset(pref_datasets)
batch_size = 1024
kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}
train_size = int(len(pref_dataset) * 0.75)
train_set, test_set = torch.utils.data.dataset.random_split(pref_dataset, (train_size, len(pref_dataset) - train_size))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from network import EmbeddingNet, PreferenceNet, FullPreferenceNet

margin = 1.
embedding_net = EmbeddingNet(args.embedding_dim)
preference_net = PreferenceNet(args.embedding_dim)
model = FullPreferenceNet(embedding_net, preference_net)

if args.embedding_epoch:
    model.load_state_dict(torch.load(os.path.join(args.embedding_ckpt_dir, 'trained_epoch_{}.pth'.format(args.embedding_epoch))), strict=False)
    for param in embedding_net.parameters():
        param.requires_grad = False

if cuda:
    model.cuda()
loss_fn = torch.nn.BCEWithLogitsLoss()
lr = 1e-3
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 5
log_interval = 5

if not os.path.exists(args.ckpt_dir):
    os.mkdir(args.ckpt_dir)

fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, args.ckpt_dir)

