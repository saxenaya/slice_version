import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import os
from torchvision import transforms
import itertools
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from torchvision.utils import make_grid
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--root_dirs', type=str, nargs='+')
parser.add_argument('--model_dir', type=str)
parser.add_argument('--save_dir', default='embeddings', type=str)
parser.add_argument('--epoch', type=str)
parser.add_argument('--embedding_dim', type=int, default=48)
parser.add_argument('--data_format', type=str, required=False)

args = parser.parse_args()

normalize_imgs = False
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
input_img_width = 40
input_img_height = 40

if normalize_imgs:
    data_transform_input = transforms.Compose([
        transforms.Resize((input_img_height, input_img_width)),
        transforms.ToTensor(),
        normalize
    ])
else:
    data_transform_input = transforms.Compose([
        transforms.Resize((input_img_height, input_img_width)),
        transforms.ToTensor(),
    ])


from network import EmbeddingNet, TripletNet

from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import random
from datasets import PathPreferenceTripletDataset
from datasets import MiniGridTripletDataset
batch_size=20

def show_img(img, cluster_idx, save=True, member_count=None, gt=False):
    npimg = img.numpy().astype(np.uint8)
    fig = plt.figure()
    if not gt:
        plt.title('Cluster {}'.format(cluster_idx))
    else:
        plt.title('GT Cluster {}'.format(cluster_idx))
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    if save:
        suffix = 'cluster_{0}.png'.format(cluster_idx)
        plt.imsave(args.save_dir + '/' + suffix, 
                   np.transpose(npimg, (1,2,0)))

def visualize_cluster_images(patches, labels, num=16, gt=False):
    # Visualize the image patches
    print("Num labels: ", len(np.unique(labels)))
    for lab in np.unique(labels):
        ind = np.where(labels == lab)
        
        patch_list = torch.tensor(patches[ind])
        if patch_list.shape[0] > num:
            patch_list = patch_list[np.random.randint(patch_list.shape[0], size=(num))]
        show_img(make_grid(patch_list, nrow=4), lab, 
                member_count=patch_list.shape[0], gt=gt)
        #full_img = transforms.ToPILImage()(full_img[0, :, :, :])
        #full_img = full_img.convert(mode = "RGB")


def plot_embeddings(embeddings, labels, patches, xlim=None, ylim=None):
    fig = plt.figure(figsize=(10,10))
    
    colors = ['#ff3322', '#3322ff', '#33ff22']# ["#%06x" % random.randint(0x333333, 0xFFFFFF) for _ in np.unique(labels)]

    for lab in np.unique(labels):
        ind = np.where(labels == lab)
        plt.gca().scatter(embeddings[ind,0], embeddings[ind,1], alpha=0.5, color=colors[lab],label=lab, picker=True, pickradius=.1)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.gca().legend()
    plt.savefig(args.save_dir + "/" + "embeddings.png")

    visualize_cluster_images(patches, labels)
    plt.show()


def extract_embeddings(dataloader, model, embedding_dim=128):
    with torch.no_grad():
        embeddings = np.zeros((30, batch_size, embedding_dim))
        patches = np.zeros((30, batch_size, 3, 40, 40))
        labels = np.zeros((30, batch_size))
        k = 0
        if model:
            model.eval()
        for data, _ in itertools.islice(dataloader, 10):
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
            
            labels[k] = 1
            labels[k+1] = 1
            labels[k+2] = 0
            for d in data:
              if model:
                emb = model.get_embedding(d).data.cpu().numpy()
                embeddings[k] = emb
              patches[k] = d.cpu().numpy()
              k+= 1
        embeddings = embeddings[:k, : , :]
        patches = patches[:k, :, :, :]
        labels = labels[:k, :]
    return embeddings.reshape(k * batch_size, embedding_dim), labels.reshape(k * batch_size), patches.reshape(k * batch_size, 3, 40, 40)

model=None
if args.model_dir:
    embedding_net = EmbeddingNet(args.embedding_dim)
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()

    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'trained_epoch_{}.pth'.format(args.epoch))))

if args.data_format == "minigrid":
    dir_name = os.path.dirname(os.path.normpath(args.root_dirs[0]))
    session_names = []
    for dir in args.root_dirs:
        base_name = os.path.basename(os.path.normpath(dir))
        curr_dir_name = os.path.dirname(os.path.normpath(dir))
        if dir_name != curr_dir_name:
            print("ERROR: All dataset folders should be under the same directory!")
            exit()
        session_names.append(base_name)

    triplet_test_dataset = MiniGridTripletDataset(dir_name, session_names, transform_input=data_transform_input)
else:
    datasets = []
    for dir in args.root_dirs:
        datasets.append(PathPreferenceTripletDataset(dir))

    triplet_test_dataset = torch.utils.data.ConcatDataset(datasets)
kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

val_embeddings_tl, val_labels_tl, val_patches_tl = extract_embeddings(triplet_test_loader, model, args.embedding_dim)

os.makedirs(args.save_dir, exist_ok=True)

visualize_cluster_images(val_patches_tl, val_labels_tl, num=32, gt=True)

if args.model_dir:
    if (args.embedding_dim > 2):
        clustering = KMeans(3).fit(val_embeddings_tl).labels_
        embedded = TSNE().fit_transform(val_embeddings_tl)
    else:
        embedded = val_embeddings_tl
        # clustering = DBSCAN(eps=0.05, min_samples=10).fit(val_embeddings_tl).labels_
        clustering = val_labels_tl
    plot_embeddings(embedded, clustering, val_patches_tl)
else:
    plt.show()