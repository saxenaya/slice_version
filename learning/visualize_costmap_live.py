import argparse
import torch
import rospy

import numpy as np
from evaluation import construct_costmap, construct_comparative_costmap, construct_gt_costmap
from PIL import Image, ImageDraw
import cv2
import os
import json
from sensor_msgs.msg import CompressedImage
from warped_image_converter import ImageConverter

from network import EmbeddingNet, CostNet, FullCostNet, PreferenceNet, FullPreferenceNet, DirectCostNet

parser = argparse.ArgumentParser()

parser.add_argument('--dir', type=str)
parser.add_argument('--embedding_dim', type=int, default=24)
parser.add_argument('--cost_model_pretrained', type=str, required=True)
parser.add_argument('--pref_model_pretrained', type=str)
parser.add_argument('--direct_model_pretrained', type=str)
parser.add_argument('--intrinsic_calib', default='../data_preprocessing/calibration/intrinsics.yaml')
parser.add_argument('--extrinsic_calib', default='../data_preprocessing/calibration/extrinsics.yaml')

args = parser.parse_args()

emb = EmbeddingNet(args.embedding_dim)
cost_net = CostNet(args.embedding_dim)
cost_model = FullCostNet(emb, cost_net)
pref_net = PreferenceNet(args.embedding_dim)
pref_emb = EmbeddingNet(args.embedding_dim)
pref_model = FullPreferenceNet(pref_emb, pref_net)
direct_model = DirectCostNet()


if torch.cuda.is_available():
  device = torch.device('cuda')
  cost_model = cost_model.cuda()
  pref_model = pref_model.cuda()
  direct_model = direct_model.cuda()
else:
  device = torch.device('cpu')

if args.cost_model_pretrained:
  cost_model.load_state_dict(torch.load(args.cost_model_pretrained, map_location=device))
if args.pref_model_pretrained:
  pref_model.load_state_dict(torch.load(args.pref_model_pretrained, map_location=device))
if args.direct_model_pretrained:
  direct_model.load_state_dict(torch.load(args.direct_model_pretrained, map_location=device))

cost_model.eval()
pref_model.eval()
direct_model.eval()

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def img_callback(converter):
  def callback(img_msg):
    img_arr = converter.convert_image(img_msg).astype(np.float32)
    costmap = construct_costmap(cost_model, img_arr, device='cpu')
    cost_img = Image.fromarray(costmap * 255 / costmap.max())

    open_cv_image = get_concat_h(Image.fromarray(img_arr.astype(np.uint8)), cost_img)

    if args.pref_model_pretrained:
      pref_costmap = construct_comparative_costmap(pref_model, img_arr)
      pref_cost_img = Image.fromarray(pref_costmap * 255 / pref_costmap.max())

    if args.direct_model_pretrained:
      direct_costmap = construct_costmap(direct_model, img_arr, device='cpu')
      direct_cost_img = Image.fromarray(direct_costmap * 255 / direct_costmap.max())

    if args.pref_model_pretrained:
      open_cv_image = get_concat_h(open_cv_image, pref_cost_img)

    if args.direct_model_pretrained:
      open_cv_image = get_concat_h(open_cv_image, direct_cost_img)

    open_cv_image = np.array(open_cv_image)
    open_cv_image = cv2.resize(open_cv_image, (2700, 768))
    # Convert RGB to BGR 
    cv2.imwrite('cost.png', open_cv_image)
    return open_cv_image, np.array(cost_img)
  return callback



if __name__ == '__main__':
  rospy.init_node("visualize_costmap")


  converter = ImageConverter(args.intrinsic_calib, args.extrinsic_calib)
  img_sub = rospy.Subscriber('/left/image_raw/compressed', CompressedImage, img_callback(converter))

  while not rospy.is_shutdown():
    rospy.spin()