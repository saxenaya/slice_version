import argparse
import torch
import numpy as np
from evaluation import construct_costmap, construct_comparative_costmap, construct_gt_costmap
from PIL import Image, ImageDraw
import cv2
import os
import json

from network import EmbeddingNet, CostNet, FullCostNet, PreferenceNet, FullPreferenceNet, DirectCostNet

parser = argparse.ArgumentParser()

parser.add_argument('--dir', type=str)
parser.add_argument('--embedding_dim', type=int, default=48)
parser.add_argument('--cost_model_pretrained', type=str, required=True)
parser.add_argument('--pref_model_pretrained', type=str)
parser.add_argument('--direct_model_pretrained', type=str)

args = parser.parse_args()

emb = EmbeddingNet(args.embedding_dim)
cost_net = CostNet(args.embedding_dim)
cost_model = FullCostNet(emb, cost_net)
pref_net = PreferenceNet(args.embedding_dim)
pref_emb = EmbeddingNet(args.embedding_dim)
pref_model = FullPreferenceNet(pref_emb, pref_net)
direct_model = DirectCostNet()

if torch.cuda.is_available():
  cost_model = cost_model.cuda()
  pref_model = pref_model.cuda()
  direct_model = direct_model.cuda()

if args.cost_model_pretrained:
  cost_model.load_state_dict(torch.load(args.cost_model_pretrained))
if args.pref_model_pretrained:
  pref_model.load_state_dict(torch.load(args.pref_model_pretrained))
if args.direct_model_pretrained:
  direct_model.load_state_dict(torch.load(args.direct_model_pretrained))

cost_model.eval()
pref_model.eval()
direct_model.eval()

patch_files = os.listdir(os.path.join(args.dir, 'patches'))
image_dir = os.path.join(args.dir, 'images')

patch_files = sorted(patch_files, key=lambda f: int(f[:f.find('.json')]))[::10]
patch_iterator = iter(patch_files)

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def getFrames():
  patch_file = next(patch_iterator)
  f = open(os.path.join(args.dir, 'patches', patch_file), 'r')
  patch_infos = json.load(f)
  if not patch_infos:
    return None, None
  
  image_file = os.path.join(image_dir, str(list(patch_infos.values())[0]["image_id"]) + '.png')
  
  img = None
  with open(image_file, 'rb') as img_f:
    img = Image.open(img_f).convert('RGB')
    img_arr = np.array(img).astype(np.float32)

    costmap = construct_costmap(cost_model, img_arr)
    cost_img = Image.fromarray(costmap * 255 / costmap.max())

    if args.pref_model_pretrained:
      pref_costmap = construct_comparative_costmap(pref_model, img_arr)
      pref_cost_img = Image.fromarray(pref_costmap * 255 / pref_costmap.max())

    if args.direct_model_pretrained:
      direct_costmap = construct_costmap(direct_model, img_arr)
      direct_cost_img = Image.fromarray(direct_costmap * 255 / direct_costmap.max())

    gt_costs = construct_gt_costmap(patch_infos, img_arr)
    gt_cost_img = Image.fromarray(gt_costs * 255)

    open_cv_image = get_concat_h(get_concat_h(img, gt_cost_img), cost_img)

    if args.pref_model_pretrained:
      open_cv_image = get_concat_h(open_cv_image, pref_cost_img)

    if args.direct_model_pretrained:
      open_cv_image = get_concat_h(open_cv_image, direct_cost_img)

    open_cv_image = np.array(open_cv_image)
    open_cv_image = cv2.resize(open_cv_image, (2700, 768))
    # Convert RGB to BGR 
    return open_cv_image, np.array(cost_img)

idx = 0
while idx < len(patch_files):
    # Get a numpy array to display from the simulation
    open_cv_image, cost_img = getFrames()
    if open_cv_image  is not None:
      cv2.imwrite('test.png', open_cv_image )
      cv2.imwrite('cost_images/cost_{}.png'.format(idx), open_cv_image)
      # exit(0)
      cv2.waitKey(2)
    idx+= 1