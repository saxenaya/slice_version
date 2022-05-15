import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import os
from torchvision import transforms
# from torchsummary import summary

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', type=str, required=True)
parser.add_argument('--trajectories', type=str, nargs='+', required=True)
parser.add_argument('--ckpt_dir', type=str, required=True)
parser.add_argument('--embedding_dim', type=int, required=True)
parser.add_argument('--data_format', type=str, required=False)
parser.add_argument('--feature_functions_name', type=str, required=False, default="Clearance")
parser.add_argument('--feature_function_thresholds_path', type=str, required=False, default=None)

args = parser.parse_args()

use_high_res_network = False
normalize_imgs = False
crop_center_imgs = False
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

if args.data_format == "minigrid":
    input_img_width = 40 # 40, 100
    input_img_height = 40 # 40, 100
else:
    input_img_width = 224  # 224
    input_img_height = 224 # 224
    crop_center_imgs = False
    normalize_imgs = True
    use_high_res_network = True

transforms_list = []
if crop_center_imgs:
    transforms_list += [transforms.CenterCrop(1100)]

transforms_list += [transforms.Resize((input_img_height, input_img_width)),
                    transforms.ToTensor()]

if normalize_imgs:
    transforms_list += [normalize]

data_transform_input = transforms.Compose(transforms_list)

print("Transformations applied to the input images:")
print(data_transform_input)



from metrics import AverageNonzeroTripletsMetric

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

# Set up data loaders
from datasets import PathPreferenceTripletDataset
from datasets import MiniGridTripletDataset
from datasets_slice import AirSimTripletDataset

pref_datasets = []
if args.data_format == "minigrid":
    triplet_dataset=MiniGridTripletDataset(args.root_dir, args.trajectories, transform_input=data_transform_input)
elif args.data_format == "airsim":
    triplet_dataset=AirSimTripletDataset(args.root_dir, args.trajectories, transform_input=data_transform_input, feature_function_set_name=args.feature_functions_name, feature_function_thresholds_path=args.feature_function_thresholds_path)
else:
    for traj_id in args.trajectories:
        pref_datasets.append(PathPreferenceTripletDataset(os.path.join(args.root_dir, traj_id)))
    triplet_dataset = torch.utils.data.ConcatDataset(pref_datasets)
batch_size = 1 # 1024, 10, 100
kwargs = {'num_workers': 16, 'pin_memory': True} if cuda else {}
train_size = int(len(triplet_dataset) * 0.75)
train_set, test_set = torch.utils.data.dataset.random_split(triplet_dataset, (train_size, len(triplet_dataset) - train_size))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)


# Set up the network and training parameters
from network import EmbeddingNet, TripletNet
from losses import TripletLoss

margin = 1.
embedding_net = EmbeddingNet(args.embedding_dim, high_res=use_high_res_network)
model = TripletNet(embedding_net)

# summary(model, input_size=(3, input_img_height, input_img_width))

if cuda:
    print("Using GPU")
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
n_epochs = 50 # used to be 400 but it takes too long
log_interval = 5

if not os.path.exists(args.ckpt_dir):
    os.mkdir(args.ckpt_dir)

fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, args.ckpt_dir)

