import torch
import pickle
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
from math import floor
import argparse
import matplotlib.cm as cm
parser = argparse.ArgumentParser()
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
from datasets import AirSimTripletDataset

parser.add_argument('--root_dirs', type=str, nargs='+')
parser.add_argument('--model_dir', type=str)
parser.add_argument('--save_dir', default='embeddings', type=str)
parser.add_argument('--epoch', type=str)
parser.add_argument('--embedding_dim', type=int, default=48)
parser.add_argument('--data_format', type=str, required=False)
parser.add_argument('--cluster_num', type=int, required=False, default=2)
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
    input_img_width_highres = 100
    input_img_height_highres = 100
else:
    input_img_width = 224  # 224
    input_img_height = 224 # 224
    input_img_width_highres = 512
    input_img_height_highres = 512
    crop_center_imgs = True
    normalize_imgs = True
    use_high_res_network = True


CLUSTER_NUM = args.cluster_num
batch_size=15
MAX_DATA_SIZE_TO_CLUSTER = 2000 # Limits the number of data points that are sampled for clustering
ONLY_CLUSTER_SUCCESSFUL_TRAJ=False # Applies clustering only to the trajectories that were finished successfully
ONLY_SAMPLE_FROM_DIST_TAILS=True # Samples data from the distribution tails of the feature function values

RANDOM_SEED = 1 # None
VISUALIZE_HIGHRES = True
AIRSIM_Visualization_Mode=True # Only set to true if visualizing data obtained from the AirSimTripletDataset
CONDITIONS_TO_VISUALIZE=["rain", "snow", "road_snow", "road_wetness", "fog", "leaves", "traffic", "blocked"]

if RANDOM_SEED is not None:
    torch.manual_seed(RANDOM_SEED)


transforms_list = []
transforms_highres_list = []
if crop_center_imgs:
    transforms_list += [transforms.CenterCrop(1100)]
    transforms_highres_list += [transforms.CenterCrop(1100)]

transforms_list += [transforms.Resize((input_img_height, input_img_width)),
                    transforms.ToTensor()]
transforms_highres_list += [transforms.Resize((input_img_height_highres, input_img_width_highres)),
                            transforms.ToTensor()]

if normalize_imgs:
    transforms_list += [normalize]

data_transform_input = transforms.Compose(transforms_list)
data_transform_input_highres = transforms.Compose(transforms_highres_list)



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
        show_img(256.0 * make_grid(patch_list, nrow=4), lab, 
                member_count=patch_list.shape[0], gt=gt)
        #full_img = transforms.ToPILImage()(full_img[0, :, :, :])
        #full_img = full_img.convert(mode = "RGB")


def plot_embeddings(embeddings, labels, patches, patch_info, xlim=None, ylim=None):
    # Plot clustered embeddings
    fig = plt.figure(figsize=(10,10))

    # cmap = cm.get_cmap('tab20', 20)
    # colors = [cmap(i) for i in range(20)]
    
    colors = ['#ff3322', '#3322ff', '#33ff22', '#00ffff', '#ff00ff', '#DE3163', '#CCCCFF', '#9FE2BF', '#DFFF00']# ["#%06x" % random.randint(0x333333, 0xFFFFFF) for _ in np.unique(labels)]

    for lab in np.unique(labels):
        ind = np.where(labels == lab)
        plt.gca().scatter(embeddings[ind,0], embeddings[ind,1], alpha=0.5, color=colors[lab],label=lab, picker=True, pickradius=.1)

    plt.title("K-Means Clustering Results")
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.gca().legend()
    plt.savefig(args.save_dir + "/" + "embeddings.png")

    visualize_cluster_images(patches, labels)
    plt.show()

    # # *************************************************
    # # Plot embeddings colored given their slip condition
    fig = plt.figure(figsize=(10,10))
    
    # colors = ['#ff3322', '#3322ff', '#33ff22', '#00ffff', '#ff00ff', '#DE3163', '#CCCCFF', '#9FE2BF', '#DFFF00']# ["#%06x" % random.randint(0x333333, 0xFFFFFF) for _ in np.unique(labels)]
    
    slip_condition = patch_info["slip_condition"].astype(int)
    for lab in np.unique(slip_condition):
        ind = np.where(slip_condition == lab)
        plt.gca().scatter(embeddings[ind,0], embeddings[ind,1], alpha=0.5, color=colors[2 * lab],label=lab, picker=True, pickradius=.1)

    plt.title("Slip Condition")
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.gca().legend()
    plt.savefig(args.save_dir + "/" + "embeddings_slip_condition.png")

    # # *************************************************
    # # Plot embeddings colored given their fog condition
    fig = plt.figure(figsize=(10,10))
    
    # colors = ['#ff3322', '#3322ff', '#33ff22', '#00ffff', '#ff00ff', '#DE3163', '#CCCCFF', '#9FE2BF', '#DFFF00']# ["#%06x" % random.randint(0x333333, 0xFFFFFF) for _ in np.unique(labels)]
    
    slip_condition = patch_info["fog_condition"].astype(int)
    for lab in np.unique(slip_condition):
        ind = np.where(slip_condition == lab)
        plt.gca().scatter(embeddings[ind,0], embeddings[ind,1], alpha=0.5, color=colors[2 * lab],label=lab, picker=True, pickradius=.1)

    plt.title("Fog Condition")
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.gca().legend()
    plt.savefig(args.save_dir + "/" + "embeddings_fog_condition.png")


    # # *************************************************
    # # Plot embeddings colored given the list of all conditions available (Only applies when in AIRSIM_Visualization_Mode because the minigrid data loader does not provide the required data)
    # colors = ['#ff3322', '#3322ff', '#33ff22', '#00ffff', '#ff00ff', '#DE3163', '#CCCCFF', '#9FE2BF', '#DFFF00']
    if AIRSIM_Visualization_Mode:
        for i in range(len(CONDITIONS_TO_VISUALIZE)):
            condition_name = CONDITIONS_TO_VISUALIZE[i]
            fig = plt.figure(figsize=(10, 10))

            condition = patch_info["conditions"][condition_name].astype(int)
            for lab in np.unique(condition):
                ind = np.where(condition == lab)
                plt.gca().scatter(embeddings[ind, 0], embeddings[ind, 1], alpha=0.5,
                                  color=colors[2 * lab], label=lab, picker=True, pickradius=.1)

            plt.title(condition_name)
            if xlim:
                plt.xlim(xlim[0], xlim[1])
            if ylim:
                plt.ylim(ylim[0], ylim[1])
            plt.gca().legend()
            plt.savefig(args.save_dir + "/" + "embeddings_" +
                        condition_name + ".png")

    # # *************************************************
    # # Plot embeddings colored given their feature function labels
    fig = plt.figure(figsize=(10,10))
    
    # colors = ['#ff3322', '#3322ff', '#33ff22', '#00ffff', '#ff00ff', '#DE3163', '#CCCCFF', '#9FE2BF', '#DFFF00']# ["#%06x" % random.randint(0x333333, 0xFFFFFF) for _ in np.unique(labels)]

    cmap = cm.get_cmap('tab20', 20)
    colors = [cmap(i) for i in range(20)]
    
    feature_func_labels = patch_info["feature_func_labels"]
    i = 0
    for lab in np.unique(feature_func_labels):
        ind = np.where(feature_func_labels == lab)
        if len(ind) > 0:
            plt.gca().scatter(embeddings[ind,0], embeddings[ind,1], alpha=0.5, color=colors[i],label=lab, picker=True, pickradius=.1)
        i += 1

    plt.title("Feature Function Labels")
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.gca().legend()
    plt.savefig(args.save_dir + "/" + "embeddings_feature_func_label.png")


    # # *************************************************
    # # Plot embeddings colored given their mean clearance
    fig = plt.figure(figsize=(10,10))
    
    # colors = ['#ff3322', '#3322ff', '#33ff22']# ["#%06x" % random.randint(0x333333, 0xFFFFFF) for _ in np.unique(labels)]

    mean_clearance = patch_info["mean_clearance"]
    plt.scatter(embeddings[:,0], embeddings[:,1], alpha=0.5, c=mean_clearance[:], cmap='viridis', picker=True, pickradius=.1)

    plt.title("Mean Obstacle Clearance")
    plt.colorbar()    
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.savefig(args.save_dir + "/" + "embeddings_mean_clearance.png")



def extract_embeddings(dataloader, model, embedding_dim=128):
    with torch.no_grad():
        mini_batch_num_full = len(dataloader)
        mini_batch_num = min(mini_batch_num_full, floor(MAX_DATA_SIZE_TO_CLUSTER / batch_size))
        print("Generating embeddings for {} data points out of the available {} in the dataset.".format(mini_batch_num * batch_size, mini_batch_num_full * batch_size))

        embeddings = np.zeros((3 * mini_batch_num , batch_size, embedding_dim))
        patches = np.zeros((3 * mini_batch_num, batch_size, 3, input_img_height, input_img_width))
        patches_highres = np.zeros((3 * mini_batch_num, batch_size, 3, input_img_height_highres, input_img_width_highres))
        labels = np.zeros((3 * mini_batch_num, batch_size))
        
        mean_clearance = np.zeros((3 * mini_batch_num, batch_size))
        slip_condition = np.zeros((3 * mini_batch_num, batch_size))
        fog_condition = np.zeros((3 * mini_batch_num, batch_size))
        # TODO(srabiee): make this more efficient
        feature_func_labels = np.empty(shape=(3 * mini_batch_num, batch_size), dtype=object)

        if AIRSIM_Visualization_Mode:
            all_conditions = np.zeros(
                (len(CONDITIONS_TO_VISUALIZE), 3 * mini_batch_num, batch_size))

        k = 0
        if model:
            model.eval()
        for data, _, data_info, data_highres in itertools.islice(dataloader,  mini_batch_num):
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
            
            labels[k] = 1
            labels[k+1] = 1
            labels[k+2] = 0
            curr_data_counter = 0
            for d in data:
              if model:
                emb = model.get_embedding(d).data.cpu().numpy()
                embeddings[k] = emb
              patches[k] = d.cpu().numpy()
              patches_highres[k] = data_highres[curr_data_counter]
              mean_clearance[k] = data_info[curr_data_counter]["mean_clearance"]
              slip_condition[k] = data_info[curr_data_counter]["slip_condition"]
              fog_condition[k] = data_info[curr_data_counter]["fog_condition"]
              feature_func_labels[k] = data_info[curr_data_counter]["label"]
              
              if AIRSIM_Visualization_Mode:
                  for i in range(len(CONDITIONS_TO_VISUALIZE)):
                      condition = CONDITIONS_TO_VISUALIZE[i]
                    #   print(data_info[curr_data_counter])
                      all_conditions[i, :,
                                     :][k] = data_info[curr_data_counter]["conditions"][condition]

              curr_data_counter += 1
              k+= 1

        embeddings = embeddings[:k, : , :]
        patches = patches[:k, :, :, :]
        patches_highres = patches_highres[:k, :, :, :]
        mean_clearance = mean_clearance[:k, :]
        slip_condition = slip_condition[:k, :]
        labels = labels[:k, :]

        all_conditions_dict = {}
        if AIRSIM_Visualization_Mode:
            for i in range(len(CONDITIONS_TO_VISUALIZE)):
                condition = CONDITIONS_TO_VISUALIZE[i]
                all_conditions[i, :, :] = all_conditions[i, :, :][:k]
                all_conditions_dict[condition] = all_conditions[i, :, :].reshape(k * batch_size)


        patch_info = {"mean_clearance": mean_clearance.reshape(k * batch_size),
            "slip_condition": slip_condition.reshape(k * batch_size),
            "fog_condition": fog_condition.reshape(k * batch_size),
            "feature_func_labels": feature_func_labels.reshape(k * batch_size),
            "conditions":all_conditions_dict }

    return embeddings.reshape(k * batch_size, embedding_dim), labels.reshape(k * batch_size), patches.reshape(k * batch_size, 3, input_img_height, input_img_width), patches_highres.reshape(k * batch_size, 3, input_img_height_highres, input_img_width_highres), patch_info

model=None
if args.model_dir:
    embedding_net = EmbeddingNet(
        args.embedding_dim, high_res=use_high_res_network)
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()

    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'trained_epoch_{}.pth'.format(args.epoch))))

if args.data_format == "minigrid" or args.data_format == "airsim":
    dir_name = os.path.dirname(os.path.normpath(args.root_dirs[0]))
    session_names = []
    for dir in args.root_dirs:
        base_name = os.path.basename(os.path.normpath(dir))
        curr_dir_name = os.path.dirname(os.path.normpath(dir))
        if dir_name != curr_dir_name:
            print("ERROR: All dataset folders should be under the same directory!")
            exit()
        session_names.append(base_name)

    if args.data_format == "minigrid":
        triplet_test_dataset = MiniGridTripletDataset(dir_name, 
                            session_names, 
                            transform_input=data_transform_input,
                            transform_input_secondary=data_transform_input_highres,
                            return_img_info=True,
                            random_state=RANDOM_SEED)
    elif args.data_format == "airsim":
        triplet_test_dataset = AirSimTripletDataset(dir_name, 
                            session_names, 
                            transform_input=data_transform_input,
                            transform_input_secondary=data_transform_input_highres,
                            return_img_info=True,
                            random_state=RANDOM_SEED,
                            only_return_successful_traj = ONLY_CLUSTER_SUCCESSFUL_TRAJ,
                            only_sample_from_dist_tails = ONLY_SAMPLE_FROM_DIST_TAILS,
                            feature_function_set_name=args.feature_functions_name,
                            feature_function_thresholds_path=args.feature_function_thresholds_path)


else:
    datasets = []
    for dir in args.root_dirs:
        datasets.append(PathPreferenceTripletDataset(dir))

    triplet_test_dataset = torch.utils.data.ConcatDataset(datasets)
kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

val_embeddings_tl, val_labels_tl, val_patches_tl, val_patches_tl_highres, patch_info= extract_embeddings(triplet_test_loader, model, args.embedding_dim)

os.makedirs(args.save_dir, exist_ok=True)


clustering_model = None
if args.model_dir:
    if (args.embedding_dim > 2):
        print("Clustering the embedding vectors into {} clusters...".format(CLUSTER_NUM))
        clustering_model = KMeans(CLUSTER_NUM, random_state=RANDOM_SEED).fit(val_embeddings_tl)
        clustering = clustering_model.labels_
        embedded = TSNE(random_state=RANDOM_SEED, perplexity=40).fit_transform(val_embeddings_tl)
    else:
        embedded = val_embeddings_tl
        # clustering = DBSCAN(eps=0.05, min_samples=10).fit(val_embeddings_tl).labels_
        clustering = val_labels_tl
    if VISUALIZE_HIGHRES:
        plot_embeddings(embedded, clustering, val_patches_tl_highres, patch_info)
    else:
        plot_embeddings(embedded, clustering, val_patches_tl, patch_info)

else:
    plt.show()

model_path = os.path.join(args.save_dir, "kmeans.pkl")
print("Saving clustering model to {}".format(model_path))
try:
    with open(model_path, "wb") as f:
        pickle.dump(clustering_model, f)
        print("Successfully saved the clustering model " +
            str(model_path))
except IOError:
    print("Error: can\'t write the file: "
        + model_path)
    exit()


