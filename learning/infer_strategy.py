#!/bin/python

# ========================================================================
# Copyright 2021 srabiee@cs.utexas.edu
# Department of Computer Sciences,
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


normalize_imgs = False
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
input_img_width = 40 # 40, 100
input_img_height = 40 # 40, 100
input_img_width_highres = 100
input_img_height_highres = 100
VISUALIZE_HIGHRES = True
RANDOM_SEED = None
batch_size=20



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

def main():
  parser = argparse.ArgumentParser(
      description='This script loads state-action representations of a machine'
                  ' learned agent and predicts the associated cluster in the' 
                  ' previously learned embedding space')

  parser.add_argument('--root_dirs', type=str, nargs='+', required=True)
  parser.add_argument('--embedding_model_dir', type=str, required=True)
  parser.add_argument('--epoch', type=str, required=True)
  parser.add_argument('--clustering_model_dir', type=str, required=True)
  parser.add_argument('--save_dir', default='embeddings', type=str, required=True)
  parser.add_argument('--embedding_dim', type=int, default=48, required=True)

  args = parser.parse_args()

  os.makedirs(args.save_dir, exist_ok=True)


  if normalize_imgs:
      data_transform_input = transforms.Compose([
          transforms.Resize((input_img_height, input_img_width)),
          transforms.ToTensor(),
          normalize
      ])
      data_transform_input_highres = transforms.Compose([
          transforms.Resize((input_img_height_highres, input_img_width_highres)),
          transforms.ToTensor(),
          normalize
      ])
  else:
      data_transform_input = transforms.Compose([
          transforms.Resize((input_img_height, input_img_width)),
          transforms.ToTensor()
      ])
      data_transform_input_highres = transforms.Compose([
          transforms.Resize((input_img_height_highres, input_img_width_highres)),
          transforms.ToTensor()
      ])

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
  embedding_net = EmbeddingNet(args.embedding_dim)
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


  for session in session_names:

      meta_data_file_path = os.path.join(dir_name, session, "info.json")
      try:
          file = open(meta_data_file_path, "r")
          meta_data = json.load(file)
          file.close()
          print("Successfully loaded the meta data file " +
                str(meta_data_file_path))
      except IOError:
          print("Error: can\'t find file or read data: "
                + meta_data_file_path)
          exit()

      episode_id_to_idx = {}
      i = 0
      for item in meta_data:
          episode_id_to_idx[item["episode_id"]] = i
          i += 1


      triplet_test_dataset = MiniGridTripletDataset(dir_name, 
                        [session], 
                        transform_input=data_transform_input,
                        transform_input_secondary=data_transform_input_highres,
                        return_img_info=True,
                        random_state=RANDOM_SEED,
                        inference_mode=True,
                        only_return_successful_traj=False,
                        only_sample_from_dist_tails=False)


      kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
      triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

      print("Extracting embeddings for {}".format(session))
      val_embeddings_tl, val_patches_tl, val_patches_tl_highres, patch_info= extract_embeddings(triplet_test_loader, model, args.embedding_dim)

      print("Predicting task strategies for {}".format(session))
      task_strategy_id = clustering_model.predict(val_embeddings_tl)

      episode_ids = patch_info["episode_id"]

      # Update the meta data with the predicted task strategies
      for j in range(len(episode_ids)):
          idx = episode_id_to_idx[episode_ids[j]]
          meta_data[idx]["task_strategy"] = int(task_strategy_id[j])

      # Save the updated meta data to file
      output_meta_data_file_path = os.path.join(dir_name, session, "info_with_strategy.json")

      try:
        file = open(output_meta_data_file_path, "w")
        json.dump(meta_data, file, indent=2)
        file.close()
        print("Saved the updated meta data including the task strategies to " +
              str(output_meta_data_file_path))
      except IOError:
        print("Error: can\'t find file or write data: " + output_meta_data_file_path)
        exit()


 

if __name__ == '__main__':
  main()
