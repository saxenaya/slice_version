#!/bin/python

# ========================================================================
# Copyright 2021 srabiee@cs.utexas.edu
# Department of Computer Science,
# University of Texas at Austin


# This software is free: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License Version 3,
# as published by the Free Software Foundation.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# Version 3 in the file COPYING that came with this distribution.
# If not, see <http://www.gnu.org/licenses/>.
# ========================================================================

"""
This script loads state-action representations of a machine learned agent
and predicts the associated cluster in the previously learned embedding space
"""

import argparse
import os
import shutil
import json
import numpy as np
import math

import torch
import pickle
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
from torchvision import transforms
import itertools
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from torchvision.utils import make_grid

from network import EmbeddingNet, TripletNet
from trainer import fit
import numpy as np
cuda = torch.cuda.is_available()

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import random
from datasets import MiniGridTripletDataset

from collections import Counter

slice_mode = True
if slice_mode:
    from datasets_slice import AirSimTripletDataset
else:
    from datasets import AirSimTripletDataset

use_high_res_network = True
crop_center_imgs = False
normalize_imgs = True
input_img_width = 224
input_img_height = 224
input_img_width_highres = 512
input_img_height_highres = 512
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
VISUALIZE_HIGHRES = True
RANDOM_SEED = None
# batch_size=20
batch_size=1


# conditions = ["rain", "snow", "fog", "leaves", "road_wetness", "road_snow", "traffic"]
conditions = ["rain", "fog", "road_snow", "traffic", "leaves", "blocked"]
strategies = ["0", "1", "2", "3", "4", "5"]


def create_strategies_dict():
    from itertools import chain, combinations
    keys = list(chain.from_iterable(combinations(conditions, r) for r in range(len(conditions)+1)))
    strategies_dict = {}
    for key in keys:
        strategies_dict[key] = []
    return strategies_dict


def extract_embeddings(dataloader, model, embedding_dim=128):
    with torch.no_grad():
        slice_size = len(dataloader)
        embeddings = np.zeros((slice_size, batch_size, embedding_dim))
        patches = np.zeros((slice_size, batch_size, 3, input_img_height, input_img_width))
        patches_highres = np.zeros((slice_size, batch_size, 3, input_img_height_highres, input_img_width_highres))
        
        mean_clearance = np.zeros((slice_size, batch_size))
        slip_condition = np.zeros((slice_size, batch_size))
        fog_condition = np.zeros((slice_size, batch_size))
        episode_id = np.zeros((slice_size, batch_size), dtype=int)
        # TODO(srabiee): make this more efficient
        feature_func_labels = np.empty(shape=(slice_size, batch_size), dtype=object)

        k = 0
        if model:
            model.eval()
        for data, _, data_info, data_highres in itertools.islice(dataloader, slice_size):
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
            
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
              episode_id[k] = data_info[curr_data_counter]["episode_id"]
              feature_func_labels[k] = data_info[curr_data_counter]["label"]
              curr_data_counter += 1
              k+= 1

        embeddings = embeddings[:k, : , :]
        patches = patches[:k, :, :, :]
        patches_highres = patches_highres[:k, :, :, :]
        mean_clearance = mean_clearance[:k, :]
        slip_condition = slip_condition[:k, :]

        patch_info = {"mean_clearance": mean_clearance.reshape(k * batch_size),
            "slip_condition": slip_condition.reshape(k * batch_size),
            "fog_condition": fog_condition.reshape(k * batch_size),
            "episode_id": episode_id.reshape(k * batch_size),
            "feature_func_labels": feature_func_labels.reshape(k * batch_size)}

    return embeddings.reshape(k * batch_size, embedding_dim), patches.reshape(k * batch_size, 3, input_img_height, input_img_width), patches_highres.reshape(k * batch_size, 3, input_img_height_highres, input_img_width_highres), patch_info

def write_data(meta_data, episode, dir_name, session):
    # Save the updated meta data to file
    if slice_mode:
        output_meta_data_file_path = os.path.join(
            dir_name, session, str(episode), "processed_data", "episode_data_with_strategy.json")
    else:
        output_meta_data_file_path = os.path.join(
            dir_name, session, "{:05d}".format(episode), "processed_data", "episode_data_with_strategy.json")

    with open(output_meta_data_file_path, "w+") as file:
        json.dump(meta_data, file, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description='This script loads state-action representations of a machine'
                    ' learned agent and predicts the associated cluster in the' 
                    ' previously learned embedding space')

    parser.add_argument('--root_dirs', type=str, nargs='+', required=True)
    parser.add_argument('--embedding_model_dir', type=str, required=True)
    parser.add_argument('--epoch', type=str, required=True)
    parser.add_argument('--clustering_model_dir', type=str, required=True)
    parser.add_argument('--save_dir', default='embeddings', type=str, required=False)
    parser.add_argument('--embedding_dim', type=int, default=48, required=True)
    print("Note: task_strategy_dict.json must be in the same directory as the embedding model.")


    args = parser.parse_args()

    task_strategy_dict_path = os.path.join(args.embedding_model_dir, "task_strategy_dict.json")
    with open(task_strategy_dict_path) as f:
        task_strategy_dict = json.load(f)

    # os.makedirs(args.save_dir, exist_ok=True)

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
        transforms_highres_list += [normalize]

    data_transform_input = transforms.Compose(transforms_list)
    data_transform_input_highres = transforms.Compose(transforms_highres_list)


    # Load clustering model
    clustering_model_path = args.clustering_model_dir
    clustering_model = None
    try:
        with open(clustering_model_path, "rb") as f:
            clustering_model = pickle.load(f)
            print("Successfully loaded the clustering model " +
                str(clustering_model_path))
    except IOError:
        print("Error: can\'t read the file: "
            + clustering_model_path)
        exit()

    # Load the embedding model
    embedding_net = EmbeddingNet(
        args.embedding_dim, high_res=use_high_res_network)
    model = TripletNet(embedding_net)
    if cuda:
        model.cuda()

    model.load_state_dict(torch.load(os.path.join(args.embedding_model_dir, 'trained_epoch_{}.pth'.format(args.epoch))))


    # Load the dataset
    dir_name = os.path.dirname(os.path.normpath(args.root_dirs[0]))
    session_names = []
    for dir in args.root_dirs:
        base_name = os.path.basename(os.path.normpath(dir))
        curr_dir_name = os.path.dirname(os.path.normpath(dir))
        if dir_name != curr_dir_name:
            print("ERROR: All dataset folders should be under the same directory!")
            exit()
        session_names.append(base_name)

    strategies_dict = create_strategies_dict()

    for session in session_names:
        triplet_test_dataset = AirSimTripletDataset(dir_name, 
                              [session], 
                              transform_input=data_transform_input,
                              transform_input_secondary=data_transform_input_highres,
                              return_img_info=True,
                              random_state=RANDOM_SEED,
                              inference_mode=True,
                              only_return_successful_traj=False,
                              only_sample_from_dist_tails=False)

        # TODO(srabiee): Currently drop_last=True because of the way retrived 
        # data is expected to be in the same batch sizes. Fix this so that task
        # strategy can be generated for all the episodes even if batch_size > 1 
        kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
        triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

        print("Extracting embeddings for {}".format(session))
        val_embeddings_tl, val_patches_tl, val_patches_tl_highres, patch_info= extract_embeddings(triplet_test_loader, model, args.embedding_dim)

        print("Predicting task strategies for {}".format(session))
        print(val_embeddings_tl.shape)
        task_strategy_id = clustering_model.predict(val_embeddings_tl)
        print("Predictions:", task_strategy_id)
        print("Task strategy shape:", task_strategy_id.shape)

        episode_ids = patch_info["episode_id"]
        print("Episode ID shape:", episode_ids.shape)

        print("Writing task strategies to file...")
        # Update the meta data with the predicted task strategies
        
        # [1234, 1234, 1234, , ....]
        # [3, 1, 2, .... ] # slices

        # determine the starting points for each episode in episode_ids
        episode_start_points = {}
        for idx, val in enumerate(episode_ids):
            if val not in episode_start_points:
                episode_start_points[val] = idx

        prev_episode = -1
        for j in range(len(episode_ids)):
            # Load the episode meta data to update it with the estimated task strategy
            episode = episode_ids[j]
            new_episode = prev_episode != episode
            if new_episode:
                if slice_mode:
                    meta_data_file_path = os.path.join(
                        dir_name, session, str(episode), "processed_data", "episode_data.json")
                else:
                    meta_data_file_path = os.path.join(
                        dir_name, session, "{:05d}".format(episode), "processed_data", "episode_data.json")
                try:
                    file = open(meta_data_file_path, "r")
                    meta_data = json.load(file)
                    file.close()
                    # print("Successfully loaded the meta data file " +
                    #       str(meta_data_file_path))
                except IOError:
                    print("Error: can\'t find file or read data: "
                        + meta_data_file_path)
                    exit()
                prev_episode = episode

            task_strategy = int(task_strategy_id[j])
            meta_data['slices'][j - episode_start_points[episode]]["task_strategy"] = task_strategy
            meta_data['slices'][j - episode_start_points[episode]]["ml_strategy"] = task_strategy_dict[str(task_strategy)]
            
            # meta_data["task_strategy"] = task_strategy

            key = []
            for condition in conditions:
                if meta_data.get(condition):
                    key.append(condition)
            strategies_dict.get(tuple(key)).append(task_strategy)
            
            if j+1 == len(episode_ids) or episode_ids[j + 1] != episode or not slice_mode:
                write_data(meta_data, episode, dir_name, session)

    # write out strategy counts per conditions combination -- update in ALPACA git repo
    strategy_counts = {}
    for key, strategy_list in strategies_dict.items():
        strategy_counts[str(key)] = Counter(strategy_list)
    output_strategy_counts_file_path = os.path.join(dir_name, "strategy_counts.json")
    print("strategy count: {}".format(output_strategy_counts_file_path))
    try:
        with open(output_strategy_counts_file_path, "w+") as f:
            f.write(json.dumps(strategy_counts, indent=2))
        # print(f"Saved task strategies count to " + str(output_strategy_counts_file_path))
    except IOError:
        print(f"Error: can't find file or write strategy counts : {output_strategy_counts_file_path}")
        exit()

 

if __name__ == '__main__':
    main()
