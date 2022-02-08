import torch
import numpy as np
# Assumes image is WxHxC
def extract_patches(image, patch_size=40):
  patches = []
  patch_coords = []
  for i in range(0, image.shape[0] - patch_size, patch_size):
    for j in range(0, image.shape[1] - patch_size, patch_size):
      patch = image[i:i+patch_size, j:j+patch_size, :]
      patches.append(patch.transpose(2, 0, 1))
      patch_coords.append((i, j))
  return patches, patch_coords

# Takes a full network and constructs the costmap for hte whole image
def construct_costmap(network, image, patch_size=40, device='cuda'):
  patches, patch_coords = extract_patches(image, patch_size)
  patch_tensor = torch.tensor(patches).to(device)

  with torch.no_grad():
    patch_costs = network(patch_tensor).cpu().numpy()

  cost_img = np.zeros((image.shape[0], image.shape[1]))

  for idx, loc in enumerate(patch_coords):
    cost_img[loc[0]:loc[0] + patch_size, loc[1]:loc[1] + patch_size] = patch_costs[idx]
  
  return cost_img

def construct_gt_costmap(patch_infos, image, patch_size=40):
    cost_img = np.zeros((image.shape[0], image.shape[1]))
    for patch in patch_infos.values():
      loc = (int(patch['coord_y']), int(patch['coord_x']))
      cost_img[loc[0]:loc[0] + patch_size, loc[1]:loc[1] + patch_size] = 0.5 if patch['on_path'] else 1
    
    return cost_img

def construct_comparative_costmap(network, image, patch_size=40, device='cuda'):
  patches, patch_coords = extract_patches(image, patch_size)
  patch_tensor = torch.tensor(patches).to(device)
  sum_img = np.zeros((image.shape[0], image.shape[1]))
  for i in range(1, len(patches)):
    compare_tensor = torch.roll(patch_tensor, i, 0).to(device)

    with torch.no_grad():
      output = network(patch_tensor, compare_tensor)
      pred_comparison = (output > 0.5).cpu().numpy()


    for idx, loc in enumerate(patch_coords):
      sum_img[loc[0]:loc[0] + patch_size, loc[1]:loc[1] + patch_size] += pred_comparison[idx]

  return sum_img