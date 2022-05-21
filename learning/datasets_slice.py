import numpy as np
from PIL import Image
import os
import json
import random
from torch._C import dtype

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
import matplotlib.pyplot as plt
import shutil
import argparse
import math
import copy
from tqdm import tqdm

class AirSimTripletDataset(Dataset):
    def __init__(self,
                 root_dir,
                 session_list,
                 episode_prefix_length=10, 
                 transform_input=None,
                 transform_input_secondary=None,
                 return_img_info=False,
                 random_state=None,
                 inference_mode=False,
                 only_return_successful_traj=False,
                 only_sample_from_dist_tails=True,
                 feature_function_set_name="Clearance",
                 feature_function_thresholds_path=None):
        self.root_dir = root_dir
        self.img_dir = ""
        self.info_file_path = ""
        self.feature_function_set_name = feature_function_set_name
        self.computed_feature_function_thresholds = {}
        
        print("Active feature-function set name: {}".format(self.feature_function_set_name))

        if feature_function_set_name == "Clearance":
            self.class_labels = ['careless',
                                'cautious']
            self.active_feature_functions = [self.feature_func_clearance]

        elif feature_function_set_name == "Velocity":
            self.class_labels = ['slow',
                                'fast']
            self.active_feature_functions = [self.feature_func_velocity]

        elif feature_function_set_name == "TurnRate":
            self.class_labels = ['low_turn',
                                'high_turn']
            self.active_feature_functions = [self.feature_func_turn_rate]
        elif feature_function_set_name == "LinAcc":
            self.class_labels = ['lowLinAcc',
                                'highLinAcc']
            self.active_feature_functions = [self.feature_func_lin_acc]

        elif feature_function_set_name == "Clearance_Velocity":
            self.class_labels = ['careless_slow',
                                'cautious_slow',
                                'careless_fast',
                                'cautious_fast']
            self.active_feature_functions = [self.feature_func_clearance,   
                                            self.feature_func_velocity]

        elif feature_function_set_name == "Clearance_TurnRate":
            self.class_labels = ['careless_low_turn',
                                 'cautious_low_turn',
                                 'careless_high_turn',
                                 'cautious_high_turn']
            self.active_feature_functions = [self.feature_func_clearance,   
                                             self.feature_func_turn_rate]

        elif feature_function_set_name == "Velocity_TurnRate":
            self.class_labels = ['slow_low_turn',
                                 'fast_low_turn',
                                 'slow_high_turn',
                                 'fast_high_turn']
            self.active_feature_functions = [self.feature_func_velocity,   
                                             self.feature_func_turn_rate]

        elif feature_function_set_name == "Velocity_TurnRate_LinAcc":
            self.class_labels = ['slow_low_turn_lowLinAcc',
                                 'fast_low_turn_lowLinAcc',
                                 'slow_high_turn_lowLinAcc',
                                 'fast_high_turn_lowLinAcc',
                                 'slow_low_turn_highLinAcc',
                                 'fast_low_turn_highLinAcc',
                                 'slow_high_turn_highLinAcc',
                                 'fast_high_turn_highLinAcc']
            self.active_feature_functions = [self.feature_func_velocity,   
                                             self.feature_func_turn_rate,
                                             self.feature_func_lin_acc]

        elif feature_function_set_name == "Velocity_TurnRate_AngAcc":
            self.class_labels = ['slow_low_turn_lowAngAcc',
                                 'fast_low_turn_lowAngAcc',
                                 'slow_high_turn_lowAngAcc',
                                 'fast_high_turn_lowAngAcc',
                                 'slow_low_turn_highAngAcc',
                                 'fast_low_turn_highAngAcc',
                                 'slow_high_turn_highAngAcc',
                                 'fast_high_turn_highAngAcc']
            self.active_feature_functions = [self.feature_func_velocity,   
                                             self.feature_func_turn_rate,
                                             self.feature_func_turn_rate_change]

        elif feature_function_set_name == "Velocity_TurnRate_AngAcc_LinAcc":
            self.class_labels = ['slow_low_turn_lowAngAcc_lowLinAcc',
                                 'fast_low_turn_lowAngAcc_lowLinAcc',
                                 'slow_high_turn_lowAngAcc_lowLinAcc',
                                 'fast_high_turn_lowAngAcc_lowLinAcc',
                                 'slow_low_turn_highAngAcc_lowLinAcc',
                                 'fast_low_turn_highAngAcc_lowLinAcc',
                                 'slow_high_turn_highAngAcc_lowLinAcc',
                                 'fast_high_turn_highAngAcc_lowLinAcc',
                                 'slow_low_turn_lowAngAcc_highLinAcc',
                                 'fast_low_turn_lowAngAcc_highLinAcc',
                                 'slow_high_turn_lowAngAcc_highLinAcc',
                                 'fast_high_turn_lowAngAcc_highLinAcc',
                                 'slow_low_turn_highAngAcc_highLinAcc',
                                 'fast_low_turn_highAngAcc_highLinAcc',
                                 'slow_high_turn_highAngAcc_highLinAcc',
                                 'fast_high_turn_highAngAcc_highLinAcc']
            self.active_feature_functions = [self.feature_func_velocity,   
                                             self.feature_func_turn_rate,
                                             self.feature_func_turn_rate_change,
                                             self.feature_func_lin_acc]

        elif feature_function_set_name == "Clearance_Velocity_TurnRate":
            self.class_labels = ['careless_slow_low_turn',
                                'cautious_slow_low_turn',
                                'careless_fast_low_turn',
                                'cautious_fast_low_turn',
                                'careless_slow_high_turn',
                                'cautious_slow_high_turn',
                                'careless_fast_high_turn',
                                'cautious_fast_high_turn']
            self.active_feature_functions = [self.feature_func_clearance,   
                                            self.feature_func_velocity,
                                            self.feature_func_turn_rate]
        
        elif feature_function_set_name == "Clearance_Velocity_LinAcc":
            self.class_labels = ['careless_slow_lowLinAcc',
                                'cautious_slow_lowLinAcc',
                                'careless_fast_lowLinAcc',
                                'cautious_fast_lowLinAcc',
                                'careless_slow_highLinAcc',
                                'cautious_slow_highLinAcc',
                                'careless_fast_highLinAcc',
                                'cautious_fast_highLinAcc']
            self.active_feature_functions = [self.feature_func_clearance,   
                                            self.feature_func_velocity,
                                            self.feature_func_lin_acc]
        else:
            print("ERROR: Unsupported feature function set: {}".format(feature_function_set_name))
            print("Currently supported feature function set names include (case-sensitive): ",
                  "Clearance, Velocity, TurnRate, Clearance_Velocity, Clearance_TurnRate, Clearance_Velocity_TurnRate")
            exit()

        # If set to True, only trajectories where the agent has
        # successfully reached the goal will be used
        self.only_use_successful_traj = only_return_successful_traj
        # If set to True, only episodes with either very small or
        # very large clearance will be sampled
        self.only_sample_from_dist_tails = only_sample_from_dist_tails
        # Whether to use mean clearance of the agent trajectory normalized 
        # by the mean clearance of the global plan. If set to false the raw 
        # mean clearance values will be used to find triplets
        self.use_normalized_mean_clearance = False
        

        self.feature_function_thresholds={
        "mean_clearance_high_thresh": 8.93,
        "mean_clearance_low_thresh": 6.85,
        "mean_clearance_mid_thresh": 7.74,
        "mean_velocity_high_thresh": 4.76,
        "mean_velocity_low_thresh": 4.6,
        "mean_velocity_mid_thresh": 4.66,
        "turn_rate_high_thresh":0.08,
        "turn_rate_low_thresh":0.054,
        "turn_rate_mid_thresh":0.067,
        "action_count_high_thresh": 60,
        "action_count_low_thresh": 40,
        "action_count_mid_thresh": 50,
        "mean_norm_clearance_high_thresh": 0.11,
        "mean_norm_clearance_low_thresh": 0.07,
        "mean_norm_clearance_mid_thresh": 0.09,
        "mean_acc_lin_high_thresh":0.0,
        "mean_acc_lin_low_thresh":0.0,
        "mean_acc_lin_mid_thresh":0.0,
        "turn_rate_change_high_thresh":0.0,
        "turn_rate_change_low_thresh":0.0,
        "turn_rate_change_mid_thresh":0.0,
        }


        if feature_function_thresholds_path is not None:
            try:
                file = open(feature_function_thresholds_path, "r")
                feature_func_thresholds = json.load(file)
                self.feature_function_thresholds.update(feature_func_thresholds)
                file.close()
                print("Successfully loaded the feature function thresholds from " +
                      str(feature_function_thresholds_path))
            except IOError:
                print("Error: can\'t find file or read data: "
                      + feature_function_thresholds_path)
                exit()

        print("Using the following feature function threshold values: ")
        print(self.feature_function_thresholds)

        self.transform_input = transform_input
        self.transform_input_secondary = transform_input_secondary
        self.return_img_info = return_img_info
        self.inference_mode = inference_mode
        self.data_label = []
        self.data_img_path = []
        self.label_to_indices = {}
        self.labels_set = set(self.class_labels)
        self.data_info = []
        if random_state is not None:
            np.random.seed(random_state)

        if self.use_normalized_mean_clearance:
            self.clearance_labeling_high_thresh = self.feature_function_thresholds["mean_norm_clearance_high_thresh"]
            self.clearance_labeling_low_thresh = self.feature_function_thresholds["mean_norm_clearance_low_thresh"]
            self.clearance_labeling_mid_thresh = self.feature_function_thresholds["mean_norm_clearance_mid_thresh"]
        else:
            self.clearance_labeling_high_thresh = self.feature_function_thresholds["mean_clearance_high_thresh"]
            self.clearance_labeling_low_thresh = self.feature_function_thresholds["mean_clearance_low_thresh"]
            self.clearance_labeling_mid_thresh = self.feature_function_thresholds["mean_clearance_mid_thresh"]

        for label in self.class_labels:
            self.label_to_indices[label] = []
        

        data_counter = 0
        all_data_counter = 0
        self.step_data_bad_counter = 0
        for session in session_list:
            session_dir = os.path.join(root_dir, session)
            # root_dir = /robodata/user_data/srabiee/CAML
            # session = processed_data_05_06_22
            print("Session directory: ", session_dir)
            episode_names = self.get_episode_names(session_dir)
            print("First 10 episode names: ", episode_names[:10])
            # episode_names = self.get_episode_names(os.path.join(session_dir, "processed_data"))
            print("ex:", os.path.join(root_dir, session, episode_names[0], "processed_data"))
            
            # tqdm will show progress bar in terminal
            for i in tqdm(range(len(episode_names))):
                curr_episode = episode_names[i]
                self.img_dir = os.path.join(root_dir, 
                                            session,
                                            curr_episode,
                                            "processed_data")
                self.step_file_path = os.path.join(root_dir,
                            session,
                            curr_episode,
                            "processed_data",           
                            'step_data.json')
                self.episode_file_path = os.path.join(root_dir,
                            session,
                            curr_episode,
                            "processed_data",           
                            'episode_data.json')

                try:
                    file = open(self.step_file_path, "r")
                    step_data = json.load(file)
                    file.close()
                except IOError:
                    print("WARNING: can\'t find file or read data: " + self.step_file_path)
                    print("Skipping this episode.")
                    all_data_counter += 1
                    continue

                try:
                    file = open(self.episode_file_path, "r")
                    episode_data = json.load(file)
                    file.close()
                except IOError:
                    print("Error: can\'t find file or read data: " + self.episode_file_path)
                    exit()

            
                if self.only_use_successful_traj and episode_data["reached_goal"] == False:
                    #TODO: remove debugging
                    # print("Did not reach goal at episode {}".format(curr_episode))
                    # print("Reached-Goal:{}".format(episode_data["reached_goal"]))
                    # print("Did not reach goal at episode", curr_episode)
                    all_data_counter += 1
                    continue
                # print("Curr episode", curr_episode)    
                slices_indexes = []
                for idx, step in enumerate(step_data):
                    # when entering a new slice, the step starts again at 1
                    # this means that we can look at where that happens to find the slices
                    # ie. slice 2 can be found with step_data[slices_indexes[2] : slices_indexes[3]]
                    if step['step'] == 1:
                        slices_indexes.append(idx)
                slices_indexes.append(len(step_data)) # for the last slice in step_data
                
                # there are len(slices_indexes) - 1 slices
                for i in range(len(slices_indexes) - 1):
                    p1 = slices_indexes[i]
                    p2 = slices_indexes[i + 1]
                    curr_slice = step_data[p1 : p2]

                    # no way to calculate acceleration if there's only one step
                    if len(curr_slice) == 1:
                        continue
                    episode_data_processed = self.extract_episode_info(episode_data, curr_slice, curr_episode)
                    
                    # TODO: change when step_data is fixed
                    # overriding the step_data based features with episode_data based features
                    episode_data_processed['mean_velocity'] = episode_data['slices'][i]['mean_speed']
                    episode_data_processed['mean_clearance'] = episode_data['slices'][i]['mean_clearance']
                    episode_data_processed['normalized_mean_clearance'] = episode_data_processed['mean_clearance']
                    episode_data_processed['mean_acc_lin'] = episode_data['slices'][i]['mean_acceleration']
                    
                    data_label, skip = self.get_label(episode_data, episode_data_processed)

                    if skip:
                        all_data_counter += 1
                        continue

                    curr_data_info = {
                        "label": data_label,
                        "mean_velocity": episode_data_processed["mean_velocity"],
                        "fog_condition": episode_data_processed["fog_condition"],
                        "mean_clearance": episode_data_processed["mean_clearance"],
                        "normalized_mean_clearance": episode_data_processed["normalized_mean_clearance"],
                        "slip_condition": episode_data_processed["slip_condition"],
                        "turn_rate": episode_data_processed["turn_rate"],
                        "mean_acc_lin": episode_data_processed["mean_acc_lin"],
                        "turn_rate_change": episode_data_processed["turn_rate_change"],
                        "action_count": episode_data_processed["action_count"],
                        "conditions": episode_data_processed["conditions"],
                        "episode_id": curr_episode,
                        "slice_id": step_data[p1]['slice']}
                    


                    self.data_info.append(curr_data_info)

                    self.data_label.append(data_label)
                    self.label_to_indices[data_label].append(data_counter)

                    # find all the .png files in the directory
                    files = [f for f in os.listdir(self.img_dir) if os.path.splitext(f)[-1].lower() == ".png"]
                    img_name = "nothing"
                    for f in files:
                        # the split and indexing trickery get the slice number from a string formatted like:
                        # "thumbnail_000_138.png"
                        if f != "trajectory_thumbnail.png" and int(f.split("_")[-1][:-4]) == step_data[p1]['slice']:
                            img_name = f
                            # print(f, "matched with img_name")
                            break
                    if img_name == "nothing":
                        print("thumbnail not found for slice", step_data[p1]['slice'], "in episode", curr_episode)
                        continue
                    self.data_img_path.append(os.path.join(self.img_dir, img_name))
                        
                    data_counter += 1
                    all_data_counter +=1

        print("Dataset Statistics:")
        print("Saving data_info")
        np.save("data_info.npy", np.array(self.data_info), allow_pickle=True)

        for item in self.label_to_indices.items():
            print("Label {}: {}".format(item[0], len(item[1])))
        print("{} out of {} data points were used.".format(data_counter, all_data_counter))

        self.compute_feature_func_thresholds()
        print("number of step_data discrepancies:", self.step_data_bad_counter)

    def get_episode_names(self, dir):
        """ 
        Returns a list of name of all episodes in the input directory
        """
        if os.path.isdir(dir):
            episode_names = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]
        else:
            print("ERROR: Directory does not exist: ", dir)
            exit()
        return episode_names

    def __getitem__(self, index):
        if self.return_img_info:
            if self.inference_mode:
                return self.get_item_with_info_inference(index)
            else:
                return self.get_item_with_info(index)
        else:
            if self.inference_mode:
                return self.get_item_default_inference(index)
            else:
                return self.get_item_default(index)

    def get_item_default_inference(self, index):
        anchor_img = Image.open(self.data_img_path[index])
        anchor_img_arr = np.asarray(anchor_img, 'float32')
        anchor_img_arr = anchor_img_arr.transpose(2, 0, 1)

        if self.transform_input:
            anchor_img = self.transform_input(anchor_img)
            return (anchor_img), []
            
        return (anchor_img_arr), []


    def get_item_with_info_inference(self, index):
        info_list = []
        anchor_img = Image.open(self.data_img_path[index])
        anchor_img_arr = np.asarray(anchor_img, 'float32')
        anchor_img_arr = anchor_img_arr.transpose(2, 0, 1)
        info_list.append(self.data_info[index])

        if self.transform_input:
            high_res_img_set = ()
            if self.transform_input_secondary:
                high_res_img_set = (self.transform_input_secondary(anchor_img))
            anchor_img = self.transform_input(anchor_img)
            return (anchor_img), [], info_list, high_res_img_set
            
        return (anchor_img_arr), [], info_list

    def get_item_default(self, index):
        anchor_img = Image.open(self.data_img_path[index])
        anchor_img_arr = np.asarray(anchor_img, 'float32')
        anchor_img_arr = anchor_img_arr.transpose(2, 0, 1)
        anchor_label = self.data_label[index]

        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(
                self.label_to_indices[anchor_label])

        sample_found_for_neg_label = False
        while not sample_found_for_neg_label:
            negative_label = np.random.choice(
                list(self.labels_set - set([anchor_label])))
            if len(self.label_to_indices[negative_label]) > 0:
                negative_index = np.random.choice(
                    self.label_to_indices[negative_label])
                sample_found_for_neg_label = True

        positive_img = Image.open(self.data_img_path[positive_index])
        negative_img = Image.open(self.data_img_path[negative_index])
        positive_img_arr = np.asarray(positive_img, 'float32')
        positive_img_arr = positive_img_arr.transpose(2, 0, 1)
        negative_img_arr = np.asarray(negative_img, 'float32')
        negative_img_arr = negative_img_arr.transpose(2, 0, 1)

        if self.transform_input:
            anchor_img = self.transform_input(anchor_img)
            positive_img = self.transform_input(positive_img)
            negative_img = self.transform_input(negative_img)
            return (anchor_img, positive_img, negative_img), []
            
        return (anchor_img_arr, positive_img_arr, negative_img_arr), []

    def get_item_with_info(self, index):
        info_list = []
        anchor_img = Image.open(self.data_img_path[index])
        anchor_img_arr = np.asarray(anchor_img, 'float32')
        anchor_img_arr = anchor_img_arr.transpose(2, 0, 1)
        anchor_label = self.data_label[index]
        info_list.append(self.data_info[index])

        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(
                self.label_to_indices[anchor_label])

        sample_found_for_neg_label = False
        while not sample_found_for_neg_label:
            negative_label = np.random.choice(
                list(self.labels_set - set([anchor_label])))
            if len(self.label_to_indices[negative_label]) > 0:
                negative_index = np.random.choice(
                    self.label_to_indices[negative_label])
                sample_found_for_neg_label = True

        info_list.append(self.data_info[positive_index])
        info_list.append(self.data_info[negative_index])

        positive_img = Image.open(self.data_img_path[positive_index])
        negative_img = Image.open(self.data_img_path[negative_index])
        positive_img_arr = np.asarray(positive_img, 'float32')
        positive_img_arr = positive_img_arr.transpose(2, 0, 1)
        negative_img_arr = np.asarray(negative_img, 'float32')
        negative_img_arr = negative_img_arr.transpose(2, 0, 1)

        if self.transform_input:
            high_res_img_set = ()
            if self.transform_input_secondary:
                high_res_img_set = (self.transform_input_secondary(anchor_img),
                                self.transform_input_secondary(positive_img),
                                self.transform_input_secondary(negative_img))
            anchor_img = self.transform_input(anchor_img)
            positive_img = self.transform_input(positive_img)
            negative_img = self.transform_input(negative_img)
            return (anchor_img, positive_img, negative_img), [], info_list, high_res_img_set
            
        return (anchor_img_arr, positive_img_arr, negative_img_arr), [], info_list

    def extract_episode_info(self, episode_data, step_data, episode_id):
        # TODO: compare output of this function to slice -> does it match?
        # TODO: compare mean velocity from episode_data to this
        episode_info = {
            "fog_condition": False,
            "slip_condition": False,
            
            "mean_velocity": 0,
            "mean_clearance": 0,
            "normalized_mean_clearance": 0,
            "turn_rate": 0,
            "action_count": 0,
            "episode_id": 0,

            "conditions":{
                "rain": False, 
                "snow":False, 
                "road_snow":False, 
                "road_wetness":False, 
                "fog":False, 
                "leaves":False,
                "traffic":False,
                "blocked":False
            }
        }

        N = len(step_data)
        
        # Calculate the mean velocity
        sum_vel = 0.0
        for frame in step_data:
            sum_vel += frame["speed"]
        episode_info["mean_velocity"] = sum_vel / N

        # Calculate the mean turn rate
        sum_steering = 0.0
        for frame in step_data:
            sum_steering += math.fabs(frame["controls"]["steering"])
        episode_info["turn_rate"] = sum_steering / N

        # Calculate the mean clearance
        sum_clearance = 0.0
        for frame in step_data:
            sum_clearance += frame["distance_to_nearest_obstacle"]
        episode_info["mean_clearance"] = sum_clearance / N
        episode_info["normalized_mean_clearance"] = episode_info["mean_clearance"]

        # Calculate the mean linear acceleration
        sum_acc_lin = 0.0
        prev_speed = 0.0
        i = 0
        for frame in step_data:
            if i > 0:
                sum_acc_lin += math.fabs(frame["speed"] - prev_speed)
            
            prev_speed = frame["speed"]
            i += 1
        episode_info["mean_acc_lin"] = sum_acc_lin / (N - 1)


        # Calculate the mean rate of change of steering
        sum_acc_rot = 0.0
        prev_steering = 0.0
        i = 0
        for frame in step_data:
            if i > 0:
                sum_acc_rot += math.fabs(frame["controls"]["steering"] - prev_steering)
            
            prev_steering = frame["controls"]["steering"]
            i += 1
        episode_info["turn_rate_change"] = sum_acc_rot / (N - 1)


        episode_info["action_count"] = N
        episode_info["episode_id"] = int(episode_id)

        episode_info["conditions"]["rain"] = episode_data["rain"]
        episode_info["conditions"]["snow"] = episode_data["snow"]
        episode_info["conditions"]["road_snow"] = episode_data["road_snow"]
        episode_info["conditions"]["road_wetness"] = episode_data["road_wetness"]
        episode_info["conditions"]["fog"] = episode_data["fog"]
        episode_info["conditions"]["leaves"] = episode_data["leaves"]
        episode_info["conditions"]["traffic"] = episode_data["traffic"]
        episode_info["conditions"]["blocked"] = episode_data["blocked"]

        # TODO(srabiee): I have kept these two condition names for 
        # compatibility with the minigrid data. To be removed in the future.
        episode_info["fog_condition"] = episode_data["fog"]
        episode_info["slip_condition"] = episode_data["snow"]

        # print("Episode ID {}".format(episode_id))
        # print("Mean vel: {}".format( episode_info["mean_velocity"]))
        # print("Mean clearance: {}".format( episode_info["mean_clearance"]))


        return episode_info

    def get_label(self, episode_data, episode_data_processed):
        N = len(self.active_feature_functions)
        feature_bits = np.full(N, False)
        skip = False

        i = 0
        for feature_func in self.active_feature_functions:
            feature_bits[i], skip = feature_func(episode_data, episode_data_processed)
            i += 1

        label_idx = 0
        for j in range(feature_bits.size):
            label_idx += feature_bits[j] * ( 2 ** j)

        return self.class_labels[label_idx], skip

    def feature_func_clearance(self, episode_data, episode_data_processed):
        value = False
        skip = False

        # The quantity based on which labeling and finding triplets 
        # will be done
        labeling_value = 0.0
        if self.use_normalized_mean_clearance:
            labeling_value = episode_data_processed["mean_clearance"]
        else:
            labeling_value = episode_data_processed["normalized_mean_clearance"]
        
        if self.only_sample_from_dist_tails and labeling_value < self.clearance_labeling_high_thresh and labeling_value > self.clearance_labeling_low_thresh:
            skip = True

        if labeling_value > self.clearance_labeling_mid_thresh:
            value = True
        else:
            value = False

        return value, skip
        

    def feature_func_velocity(self, episode_data, episode_data_processed):
        value = False
        skip = False
        
        labeling_value = episode_data_processed["mean_velocity"]

        if self.only_sample_from_dist_tails and labeling_value < self.feature_function_thresholds["mean_velocity_high_thresh"] and labeling_value > self.feature_function_thresholds["mean_velocity_low_thresh"]:
            skip = True

        if labeling_value > self.feature_function_thresholds["mean_velocity_mid_thresh"]:
            value = True
        else:
            value = False

        return value, skip

    def feature_func_turn_rate(self, episode_data, episode_data_processed):
        value = False
        skip = False

        labeling_value = episode_data_processed["turn_rate"]

        if self.only_sample_from_dist_tails and labeling_value < self.feature_function_thresholds["turn_rate_high_thresh"] and labeling_value > self.feature_function_thresholds["turn_rate_low_thresh"]:
            skip = True

        if labeling_value > self.feature_function_thresholds["turn_rate_mid_thresh"]:
            value = True
        else:
            value = False

        return value, skip


    def feature_func_lin_acc(self, episode_data, episode_data_processed):
        value = False
        skip = False

        labeling_value = episode_data_processed["mean_acc_lin"]

        if self.only_sample_from_dist_tails and labeling_value < self.feature_function_thresholds["mean_acc_lin_high_thresh"] and labeling_value > self.feature_function_thresholds["mean_acc_lin_low_thresh"]:
            skip = True

        if labeling_value > self.feature_function_thresholds["mean_acc_lin_mid_thresh"]:
            value = True
        else:
            value = False

        return value, skip


    def feature_func_turn_rate_change(self, 
                                     episode_data, 
                                    episode_data_processed):
        value = False
        skip = False

        labeling_value = episode_data_processed["turn_rate_change"]

        if self.only_sample_from_dist_tails and labeling_value < self.feature_function_thresholds["turn_rate_change_high_thresh"] and labeling_value > self.feature_function_thresholds["turn_rate_change_low_thresh"]:
            skip = True

        if labeling_value > self.feature_function_thresholds["turn_rate_change_mid_thresh"]:
            value = True
        else:
            value = False

        return value, skip


    def feature_func_action_count(self, episode_data, episode_data_processed):
        value = False
        skip = False

        labeling_value =  episode_data_processed["action_count"]

        if self.only_sample_from_dist_tails and labeling_value < self.feature_function_thresholds["action_count_high_thresh"] and labeling_value > self.feature_function_thresholds["action_count_low_thresh"]:
            skip = True

        if labeling_value > self.feature_function_thresholds["action_count_mid_thresh"]:
            value = True
        else:
            value = False

        return value, skip


    def compute_feature_func_thresholds(self):
        """
        TODO: Processes all the loaded data and computes threshold values to be
        used by individual feature functions given the distribution of the data.
        """

        ALL_CONDITIONS = ["rain", "snow", "road_snow", "road_wetness", "fog", "leaves", "traffic", "blocked"]

        # In addition to computing the statistics for all data, compute that   for these conditions as well
        SELECTED_CONDITION_COMBINATIONS = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0], dtype=bool, ndmin=2)
        SELECTED_CONDITION_MASK = np.array(
            [0, 0, 1, 0, 0, 0, 0, 0], dtype=bool, ndmin=2)

        # A list for printing out the condition combinations that sets
        # any wild-card bit to "NAN" 
        SELECTED_CONDITION_VISUALIZATION = np.ones(
            (SELECTED_CONDITION_MASK.size), dtype=float)
        for i in range(SELECTED_CONDITION_MASK.size):
            mask_entry = SELECTED_CONDITION_MASK[0, i]
            if not mask_entry:
                SELECTED_CONDITION_VISUALIZATION[i] = np.NAN
            else:
                SELECTED_CONDITION_VISUALIZATION[i] = SELECTED_CONDITION_COMBINATIONS[0, i]


        N = len(self.data_info)
        if (N > 0):
            mean_vel = 0.0
            mean_clearance = 0.0
            turn_rate = 0.0
            mean_action_count = 0.0
            mean_acc_lin = 0.0
            turn_rate_change = 0.0

            mean_vel_list = np.zeros((N, 1), dtype=float)
            mean_clearance_list = np.zeros((N, 1), dtype=float)
            turn_rate_list = np.zeros((N, 1), dtype=float)
            mean_acc_lin_list = np.zeros((N, 1), dtype=float)
            turn_rate_change_list = np.zeros((N, 1), dtype=float)
            action_count_list = np.zeros((N, 1), dtype=float)
            condition_code_list = np.zeros((N, len(ALL_CONDITIONS)), dtype=bool)

            for i in range(N):
                data_point = self.data_info[i]
                mean_vel_list[i] = data_point["mean_velocity"]
                mean_clearance_list[i] = data_point["mean_clearance"]
                turn_rate_list[i] = data_point["turn_rate"]
                mean_acc_lin_list[i] = data_point["mean_acc_lin"]
                turn_rate_change_list[i] = data_point["turn_rate_change"]
                action_count_list[i] = data_point["action_count"]

                for j in range(len(ALL_CONDITIONS)):
                    condition_code_list[i, j] = data_point["conditions"][ALL_CONDITIONS[j]] 

            selected_cnd = np.repeat(SELECTED_CONDITION_COMBINATIONS, N, 0)
            mask_cnd = np.repeat(SELECTED_CONDITION_MASK, N, 0)

            masked_conditions = np.logical_and(mask_cnd, condition_code_list) 
            comp_result = np.all(masked_conditions == selected_cnd, axis=1)
            met_cnd_idx = np.argwhere(comp_result)
            

            print("Statistics over all the data:")
            print("Mean velocity = {} +- {}.  q1:{}, q3:{}".format(
                    np.mean(mean_vel_list), np.std(mean_vel_list), np.quantile(mean_vel_list, 0.25), np.quantile(mean_vel_list, 0.75)))
            print("Mean clearance = {} +- {}.  q1:{}, q3:{}".format(
                    np.mean(mean_clearance_list), np.std(mean_clearance_list), np.quantile(mean_clearance_list, 0.25), np.quantile(mean_clearance_list, 0.75)))
            print("Mean turn_rate = {} +- {}.  q1:{}, q3:{}".format(
                    np.mean(turn_rate_list), np.std(turn_rate_list), np.quantile(turn_rate_list, 0.25), np.quantile(turn_rate_list, 0.75)))
            print("Mean linear acc = {} +- {}.  q1:{}, q3:{}".format(
                    np.mean(mean_acc_lin_list), np.std(mean_acc_lin_list), np.quantile(mean_acc_lin_list, 0.25), np.quantile(mean_acc_lin_list, 0.75)))
            print("Mean turn_rate_change = {} +- {}.  q1:{}, q3:{}".format(
                    np.mean(turn_rate_change_list), np.std(turn_rate_change_list), np.quantile(turn_rate_change_list, 0.25), np.quantile(turn_rate_change_list, 0.75)))    
            print("Mean action count = {} +- {}.  q1:{}, q3:{}".format(
                    np.mean(action_count_list), np.std(action_count_list), np.quantile(action_count_list, 0.25), np.quantile(action_count_list, 0.75)))

            
            print("")
            print("Statistics for condition:")
            print(list(zip(ALL_CONDITIONS, SELECTED_CONDITION_VISUALIZATION.tolist() )))
            print("Mean velocity = {} +- {}".format(
                np.mean(mean_vel_list[met_cnd_idx, :]), np.std(mean_vel_list[met_cnd_idx, :])))
            print("Mean clearance = {} +- {}".format(
                np.mean(mean_clearance_list[met_cnd_idx, :]), np.std(mean_clearance_list[met_cnd_idx, :])))
            print("Mean turn_rate = {} +- {}".format(
                np.mean(turn_rate_list[met_cnd_idx, :]), np.std(turn_rate_list[met_cnd_idx, :])))
            print("Mean linear acc = {} +- {}".format(
                np.mean(mean_acc_lin_list[met_cnd_idx, :]), np.std(mean_acc_lin_list[met_cnd_idx, :])))
            print("Mean turn_rate_change = {} +- {}".format(
                np.mean(turn_rate_change_list[met_cnd_idx, :]), np.std(turn_rate_change_list[met_cnd_idx, :])))
            print("Mean action count = {} +- {}".format(
                np.mean(action_count_list[met_cnd_idx, :]), np.std(action_count_list[met_cnd_idx, :])))

            # Computed feature function thresholds for the currently loaded 
            # dataset. Note: this can be different from the active feature
            # thresholds that might be loaded from file
            self.computed_feature_function_thresholds={
            "mean_clearance_high_thresh":np.quantile(mean_clearance_list, 0.75),
            "mean_clearance_low_thresh":np.quantile(mean_clearance_list, 0.25),
            "mean_clearance_mid_thresh":np.mean(mean_clearance_list),
            "mean_velocity_high_thresh":np.quantile(mean_vel_list, 0.75),
            "mean_velocity_low_thresh":np.quantile(mean_vel_list, 0.25),
            "mean_velocity_mid_thresh":np.mean(mean_vel_list),
            "turn_rate_high_thresh":np.quantile(turn_rate_list, 0.75),
            "turn_rate_low_thresh":np.quantile(turn_rate_list, 0.25),
            "turn_rate_mid_thresh":np.mean(turn_rate_list),
            "mean_acc_lin_high_thresh":np.quantile(mean_acc_lin_list, 0.75),
            "mean_acc_lin_low_thresh":np.quantile(mean_acc_lin_list, 0.25),
            "mean_acc_lin_mid_thresh":np.mean(mean_acc_lin_list),
            "turn_rate_change_high_thresh":np.quantile(turn_rate_change_list, 0.75),
            "turn_rate_change_low_thresh":np.quantile(turn_rate_change_list, 0.25),
            "turn_rate_change_mid_thresh":np.mean(turn_rate_change_list)
            }

    def get_computed_feature_function_thresholds(self):
        return self.computed_feature_function_thresholds

    def __len__(self):
        return len(self.data_label)


class MiniGridTripletDataset(Dataset):
    def __init__(self,
                 root_dir,
                 session_list,
                 episode_prefix_length=10, 
                 transform_input=None,
                 transform_input_secondary=None,
                 return_img_info=False,
                 random_state=None,
                 inference_mode=False,
                 only_return_successful_traj=True,
                 only_sample_from_dist_tails=True):
        self.root_dir = root_dir
        # self.img_dir = os.path.join(root_dir, 'images', 'full_episodes')
        # self.info_file_path = os.path.join(root_dir, 'info.json')
        self.img_dir = ""
        self.info_file_path = ""
        self.episode_name_format = "{0:010d}"
        self.image_folder_name = "full_episodes" # "full_episodes", "full_episodes_no_color"
        
        # self.class_labels = ['careless',
        #                     'cautious']
        # self.active_feature_functions = [self.feature_func_clearance]

        self.class_labels = ['slow',
                             'fast']
        self.active_feature_functions = [self.feature_func_velocity]

        # self.class_labels = ['low_turn',
        #                      'high_turn']
        # self.active_feature_functions = [self.feature_func_turn_rate]

        # self.class_labels = ['careless_slow',
        #                      'cautious_slow',
        #                      'careless_fast',
        #                      'cautious_fast']
        # self.active_feature_functions = [self.feature_func_clearance,   
        #                                  self.feature_func_velocity]

        # If set to True, only trajectories where the agent has
        # successfully reached the goal will be used
        self.only_use_successful_traj = only_return_successful_traj
        # If set to True, only episodes with either very small or
        # very large clearance will be sampled
        self.only_sample_from_dist_tails = only_sample_from_dist_tails
        # Whether to use mean clearance of the agent trajectory normalized 
        # by the mean clearance of the global plan. If set to false the raw 
        # mean clearance values will be used to find triplets
        self.use_normalized_mean_clearance = False
        
        self.mean_clearance_high_thresh = 5.0 # 4.5 , 4.4
        self.mean_clearance_low_thresh = 3.2 # 2.5, 3.6
        self.mean_clearance_mid_thresh = 4.18 # 3, 4.18
        self.mean_norm_clearance_high_thresh = 1.2 # 1.4, 1.5, 1.6
        self.mean_norm_clearance_low_thresh = 0.8 # 0.7, 0.6, 0.5
        self.mean_norm_clearance_mid_thresh = 1.0 # 1.0, 1.0, 1.0
        self.mean_velocity_high_thresh = 2.1
        self.mean_velocity_low_thresh = 1.6
        self.mean_velocity_mid_thresh = 1.8
        self.turn_rate_high_thresh = 0.45
        self.turn_rate_low_thresh = 0.37
        self.turn_rate_mid_thresh = 0.40
        self.action_count_high_thresh = 300
        self.action_count_low_thresh = 100
        self.action_count_mid_thresh = 200

        self.transform_input = transform_input
        self.transform_input_secondary = transform_input_secondary
        self.return_img_info = return_img_info
        self.inference_mode = inference_mode
        self.data_label = []
        self.data_img_path = []
        self.label_to_indices = {}
        self.labels_set = set(self.class_labels)
        self.data_info = []
        if random_state is not None:
            np.random.seed(random_state)

        if self.use_normalized_mean_clearance:
            self.clearance_labeling_high_thresh = self.mean_norm_clearance_high_thresh
            self.clearance_labeling_low_thresh = self.mean_norm_clearance_low_thresh
            self.clearance_labeling_mid_thresh = self.mean_norm_clearance_mid_thresh
        else:
            self.clearance_labeling_high_thresh = self.mean_clearance_high_thresh
            self.clearance_labeling_low_thresh = self.mean_clearance_low_thresh
            self.clearance_labeling_mid_thresh = self.mean_clearance_mid_thresh

        for label in self.class_labels:
            self.label_to_indices[label] = []
        

        # The length of the session name prefix
        if (episode_prefix_length == 10):
            self.episode_name_format = "{0:010d}"
        else:
            print("ERROR: Episode prefix length of %d is not supported.",
                        episode_prefix_length)
            exit()

        # TODO(srabiee): Verify that the length of self.class_labels matches the
        # number of feature_functions that are queued
        
        data_counter = 0
        all_data_counter = 0
        for session in session_list:
            self.img_dir = os.path.join(root_dir, session, 'images', self.image_folder_name)
            self.info_file_path = os.path.join(root_dir, session, 'info.json')

            try:
                file = open(self.info_file_path, "r")
                meta_data = json.load(file)
                file.close()
            except IOError:
                print("Error: can\'t find file or read data: " + self.info_file_path)
                exit()

            
            for i in range(len(meta_data)):
                entry = meta_data[i]
                if self.only_use_successful_traj and entry["reached_goal"] == 0:
                    all_data_counter += 1
                    continue
                
                data_label, skip = self.get_label(entry)

                if skip:
                    all_data_counter += 1
                    continue 

                self.data_label.append(data_label)
                self.label_to_indices[data_label].append(data_counter)
                img_name = self.episode_name_format.format(i) + ".png";
                self.data_img_path.append(os.path.join(self.img_dir, img_name))

                curr_data_info = {
                    "label": data_label,
                    "mean_velocity": entry["mean_velocity"],
                    "fog_condition": entry["fog_condition"],
                    "mean_clearance": entry["mean_clearance"],
                    "normalized_mean_clearance": entry["mean_clearance"] / entry["global_plan_mean_clearance"],
                    "slip_condition": entry["slips"]["slippage_condition"],
                    "turn_rate": entry["counts"]["turn_count"]/ entry["counts"]["action_count"],
                    "action_count": entry["counts"]["action_count"]}
                if "episode_id" in entry:
                    curr_data_info["episode_id"] = entry["episode_id"]

                self.data_info.append(curr_data_info)
                        
                data_counter += 1
                all_data_counter +=1

        print("Dataset Statistics:")
        
        for item in self.label_to_indices.items():
            print("Label {}: {}".format(item[0], len(item[1])))
        print("{} out of {} data points were used.".format(data_counter, all_data_counter))

        if (len(self.data_info) > 0):
            mean_vel = 0.0
            mean_clearance = 0.0
            turn_rate = 0.0
            mean_action_count = 0.0
            for data_point in self.data_info:
                mean_vel += data_point["mean_velocity"]
                mean_clearance += data_point["mean_clearance"]
                turn_rate += data_point["turn_rate"]
                mean_action_count += data_point["action_count"]
            mean_vel /= len(self.data_info)
            mean_clearance /= len(self.data_info)
            turn_rate /= len(self.data_info)
            mean_action_count /= len(self.data_info)
            print("Mean velocity = {}".format(mean_vel))
            print("Mean clearance = {}".format(mean_clearance))
            print("Mean turn_rate = {}".format(turn_rate))
            print("Mean action count = {}".format(mean_action_count))



        def __getitem__(self, index):
            if self.return_img_info:
                if self.inference_mode:
                    return self.get_item_with_info_inference(index)
                else:
                    return self.get_item_with_info(index)
            else:
                if self.inference_mode:
                    return self.get_item_default_inference(index)
                else:
                    return self.get_item_default(index)

        def get_item_default_inference(self, index):
            anchor_img = Image.open(self.data_img_path[index])
            anchor_img_arr = np.asarray(anchor_img, 'float32')
            anchor_img_arr = anchor_img_arr.transpose(2, 0, 1)

            if self.transform_input:
                anchor_img = self.transform_input(anchor_img)
                return (anchor_img), []
                
            return (anchor_img_arr), []


        def get_item_with_info_inference(self, index):
            info_list = []
            anchor_img = Image.open(self.data_img_path[index])
            anchor_img_arr = np.asarray(anchor_img, 'float32')
            anchor_img_arr = anchor_img_arr.transpose(2, 0, 1)
            info_list.append(self.data_info[index])

            if self.transform_input:
                high_res_img_set = ()
                if self.transform_input_secondary:
                    high_res_img_set = (self.transform_input_secondary(anchor_img))
                anchor_img = self.transform_input(anchor_img)
                return (anchor_img), [], info_list, high_res_img_set
                
            return (anchor_img_arr), [], info_list

        def get_item_default(self, index):
            anchor_img = Image.open(self.data_img_path[index])
            anchor_img_arr = np.asarray(anchor_img, 'float32')
            anchor_img_arr = anchor_img_arr.transpose(2, 0, 1)
            anchor_label = self.data_label[index]

            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(
                    self.label_to_indices[anchor_label])

            negative_label = np.random.choice(
                list(self.labels_set - set([anchor_label])))
            negative_index = np.random.choice(
                self.label_to_indices[negative_label])

            positive_img = Image.open(self.data_img_path[positive_index])
            negative_img = Image.open(self.data_img_path[negative_index])
            positive_img_arr = np.asarray(positive_img, 'float32')
            positive_img_arr = positive_img_arr.transpose(2, 0, 1)
            negative_img_arr = np.asarray(negative_img, 'float32')
            negative_img_arr = negative_img_arr.transpose(2, 0, 1)

            if self.transform_input:
                anchor_img = self.transform_input(anchor_img)
                positive_img = self.transform_input(positive_img)
                negative_img = self.transform_input(negative_img)
                return (anchor_img, positive_img, negative_img), []
                
            return (anchor_img_arr, positive_img_arr, negative_img_arr), []

        def get_item_with_info(self, index):
            info_list = []
            anchor_img = Image.open(self.data_img_path[index])
            anchor_img_arr = np.asarray(anchor_img, 'float32')
            anchor_img_arr = anchor_img_arr.transpose(2, 0, 1)
            anchor_label = self.data_label[index]
            info_list.append(self.data_info[index])

            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(
                    self.label_to_indices[anchor_label])

            negative_label = np.random.choice(
                list(self.labels_set - set([anchor_label])))
            negative_index = np.random.choice(
                self.label_to_indices[negative_label])

            info_list.append(self.data_info[positive_index])
            info_list.append(self.data_info[negative_index])

            positive_img = Image.open(self.data_img_path[positive_index])
            negative_img = Image.open(self.data_img_path[negative_index])
            positive_img_arr = np.asarray(positive_img, 'float32')
            positive_img_arr = positive_img_arr.transpose(2, 0, 1)
            negative_img_arr = np.asarray(negative_img, 'float32')
            negative_img_arr = negative_img_arr.transpose(2, 0, 1)

            if self.transform_input:
                high_res_img_set = ()
                if self.transform_input_secondary:
                    high_res_img_set = (self.transform_input_secondary(anchor_img),
                                    self.transform_input_secondary(positive_img),
                                    self.transform_input_secondary(negative_img))
                anchor_img = self.transform_input(anchor_img)
                positive_img = self.transform_input(positive_img)
                negative_img = self.transform_input(negative_img)
                return (anchor_img, positive_img, negative_img), [], info_list, high_res_img_set
                
            return (anchor_img_arr, positive_img_arr, negative_img_arr), [], info_list

        def get_label(self, data):
            N = len(self.active_feature_functions)
            feature_bits = np.full(N, False)
            skip = False

            i = 0
            for feature_func in self.active_feature_functions:
                feature_bits[i], skip = feature_func(data)
                i += 1

            label_idx = 0
            for j in range(feature_bits.size):
                label_idx += feature_bits[j] * ( 2 ** j)

            return self.class_labels[label_idx], skip

        def feature_func_clearance(self, data):
            value = False
            skip = False

            # The quantity based on which labeling and finding triplets 
            # will be done
            labeling_value = 0.0
            if self.use_normalized_mean_clearance:
                labeling_value = data["mean_clearance"] / data["global_plan_mean_clearance"]
            else:
                labeling_value = data["mean_clearance"]
            
            if self.only_sample_from_dist_tails and labeling_value < self.clearance_labeling_high_thresh and labeling_value > self.clearance_labeling_low_thresh:
                skip = True

            if labeling_value > self.clearance_labeling_mid_thresh:
                value = True
            else:
                value = False

            return value, skip
        
    # Yash
    def feature_func_mean_acc_lin(self, data):
        value = False
        skip = False

        labeling_value = data["mean_acc_lin"]
        if self.only_sample_from_dist_tails and labeling_value < self.mean_acc_lin_high_thresh and labeling_value > self.mean_acc_lin_low_thresh:
            skip = True

        if labeling_value > self.mean_acc_lin_mid_thresh:
            value = True
        else:
            value = False

        return value, skip
       

    def feature_func_velocity(self, data):
        value = False
        skip = False

        labeling_value = data["mean_velocity"]

        if self.only_sample_from_dist_tails and labeling_value < self.mean_velocity_high_thresh and labeling_value > self.mean_velocity_low_thresh:
            skip = True

        if labeling_value > self.mean_velocity_mid_thresh:
            value = True
        else:
            value = False

        return value, skip

    def feature_func_turn_rate(self, data):
        value = False
        skip = False

        labeling_value = data["counts"]["turn_count"] / data["counts"]["action_count"]

        if self.only_sample_from_dist_tails and labeling_value < self.turn_rate_high_thresh and labeling_value > self.turn_rate_low_thresh:
            skip = True

        if labeling_value > self.turn_rate_mid_thresh:
            value = True
        else:
            value = False

        return value, skip


    def feature_func_action_count(self, data):
        value = False
        skip = False

        labeling_value = data["counts"]["action_count"] 

        if self.only_sample_from_dist_tails and labeling_value < self.action_count_high_thresh and labeling_value > self.action_count_low_thresh:
            skip = True

        if labeling_value > self.action_count_mid_thresh:
            value = True
        else:
            value = False

        return value, skip


    def compute_feature_func_thresholds(self):
        """
        TODO: Processes all the loaded data and computes threshold values to be
        used by individual feature functions given the distribution of the data.
        """
        pass


    def __len__(self):
        return len(self.data_label)


class PathPreferenceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'images')
        self.patch_dir = os.path.join(root_dir, 'patches')

    def get_image_path(self, image_id):
        return os.path.join(self.img_dir, str(image_id) + '.png')

    def get_patch_details(self, image_id, patch_id):
        img_patch_path = os.path.join(self.patch_dir, str(image_id) + '.json')
        f = open(img_patch_path, 'r')
        img_patch_details = json.load(f)
        f.close()
        return img_patch_details[str(patch_id)]

    def extract_patch_from_info(self, patch_info):
        img = Image.open(self.get_image_path(patch_info["image_id"]))
        img_arr = np.asarray(img, 'float32')
        patch_details = self.get_patch_details(patch_info["image_id"], patch_info["patch_id"])
        center_x, center_y = patch_details["coord_x"], patch_details["coord_y"]
        patch_radius = int(patch_details["patch_size"] / 2.0)
        img_arr = img_arr[int(center_y - patch_radius):int(center_y+patch_radius), int(center_x - patch_radius): int(center_x + patch_radius), :]
        if (img_arr.shape[0] != patch_radius * 2 or img_arr.shape[1] != patch_radius * 2):
            raise Exception("bad patch size for patch centered at {}, {}".format(center_y, center_x))
        return img_arr.transpose(2, 0, 1)
class PathPreferenceTripletDataset(PathPreferenceDataset):
    def __init__(self, root_dir):
        super(PathPreferenceTripletDataset, self).__init__(root_dir)
        loc_patch_file = open(os.path.join(root_dir, 'patches_by_loc.json'), 'r')
        self.loc_patch_info = json.load(loc_patch_file)
        loc_patch_file.close()
        neg_patch_file = open(os.path.join(root_dir, 'neg_patches.json'), 'r')
        self.neg_patch_info = json.load(neg_patch_file)
        neg_patch_file.close()

    def get_image_path(self, image_id):
        return os.path.join(self.img_dir, str(image_id) + '.png')

    def get_patch_details(self, image_id, patch_id):
        img_patch_path = os.path.join(self.patch_dir, str(image_id) + '.json')
        f = open(img_patch_path, 'r')
        img_patch_details = json.load(f)
        f.close()
        return img_patch_details[str(patch_id)]

    def extract_patch_from_info(self, patch_info):
        img = Image.open(self.get_image_path(patch_info["image_id"]))
        img_arr = np.asarray(img, 'float32')
        patch_details = self.get_patch_details(patch_info["image_id"], patch_info["patch_id"])
        center_x, center_y = patch_details["coord_x"], patch_details["coord_y"]
        patch_radius = int(patch_details["patch_size"] / 2.0)
        img_arr = img_arr[int(center_y - patch_radius):int(center_y+patch_radius), int(center_x - patch_radius): int(center_x + patch_radius), :]
        if (img_arr.shape[0] != patch_radius * 2 or img_arr.shape[1] != patch_radius * 2):
            raise Exception("bad patch size for patch centered at {}, {}".format(center_y, center_x))
        return img_arr.transpose(2, 0, 1)

    def __getitem__(self, index):
        modulo_idx = index % len(self.loc_patch_info.keys())
        anchor_infos = self.loc_patch_info[str(modulo_idx)]
        anchor_idx, similar_idx = np.random.randint(len(anchor_infos["patches"]), size=2)
        anchor_patch_info = anchor_infos["patches"][anchor_idx]
        similar_patch_info = anchor_infos["patches"][similar_idx]
        neg_idx = np.random.randint(len(self.neg_patch_info["patches"]))
        neg_patch_info = self.neg_patch_info["patches"][neg_idx]

        anchor_patch = self.extract_patch_from_info(anchor_patch_info)
        similar_patch = self.extract_patch_from_info(similar_patch_info)
        neg_patch = self.extract_patch_from_info(neg_patch_info)

        return (anchor_patch, similar_patch, neg_patch), []

    def __len__(self):
        return len(self.loc_patch_info.keys()) * 2

class PathPreferenceCostDataset(PathPreferenceDataset):
    def __init__(self, root_dir):
        super(PathPreferenceCostDataset, self).__init__(root_dir)
        self.patch_infos = []
        for patch_filename in os.listdir(self.patch_dir):
            f = open(os.path.join(self.patch_dir, patch_filename), 'r')
            patch_info = json.load(f)
            if patch_info is None:
                print("Missing patch info for {}".format(patch_filename))
                continue
            for pi in patch_info.values():
                self.patch_infos.append(pi)
            f.close()

    def __getitem__(self, index):
        patch_info = self.patch_infos[index]
        patch = self.extract_patch_from_info(patch_info)

        cost = 0.0 if patch_info["on_path"] else 100.0

        return (patch, np.float32(cost))

    
    def __len__(self):
        return len(self.patch_infos)

class PathPreferenceDecisionDataset(PathPreferenceDataset):
    def __init__(self, root_dir):
        super(PathPreferenceDecisionDataset, self).__init__(root_dir)
        self.patch_infos = []
        for patch_filename in os.listdir(self.patch_dir):
            f = open(os.path.join(self.patch_dir, patch_filename), 'r')
            patch_info = json.load(f)
            if patch_info is None:
                print("Missing patch info for {}".format(patch_filename))
                continue
            for pi in patch_info.values():
                if pi["on_path"]:
                    self.patch_infos.append(pi)
            f.close()
        neg_patch_file = open(os.path.join(root_dir, 'neg_patches.json'), 'r')
        neg_patch_info = json.load(neg_patch_file)
        neg_patch_file.close()
        self.neg_patch_infos = neg_patch_info["patches"]

    def __getitem__(self, index):
        if index < len(self.patch_infos):
            patch_info = self.patch_infos[index]
            patch = self.extract_patch_from_info(patch_info)
            neg_idx = np.random.randint(len(self.neg_patch_infos))
            neg_patch = self.extract_patch_from_info(self.neg_patch_infos[neg_idx])
            return (patch, neg_patch), 0.0
        else:
            neg_patch_info = self.neg_patch_infos[index - len(self.patch_infos)]
            neg_patch = self.extract_patch_from_info(neg_patch_info)
            pos_idx = np.random.randint(len(self.patch_infos))
            pos_patch = self.extract_patch_from_info(self.patch_infos[pos_idx])
            return (neg_patch, pos_patch), 1.0
    
    def __len__(self):
        return len(self.patch_infos) + len(self.neg_patch_infos)

class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

def inspect_triplets(root_dir, dataset_folders):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    transform_input = transforms.Compose([
        # transforms.CenterCrop(1100),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    root_dir = root_dir
    trajectories = [ f for f in dataset_folders]

    triplet_dataset = AirSimTripletDataset(root_dir, trajectories,
                                             transform_input=transform_input,
                                             inference_mode=False,
                                             return_img_info=True,
                                            only_return_successful_traj=True,
                                            only_sample_from_dist_tails=True,
                                            feature_function_set_name="Velocity")


    computed_feature_function_thresholds = triplet_dataset.get_computed_feature_function_thresholds()  

    print()
    print("Computed Feature Function Thresholds: ")
    print(computed_feature_function_thresholds)


    # Sample a number of data points and store the corresponding images
    # to file for testing the data quality and the data loader pipeline
    sample_num = 50
    output_dir = "tmp_output_2"

    
    anchor_output_path = os.path.join(output_dir, "anchor")
    positive_output_path = os.path.join(output_dir, "positive")
    negative_output_path = os.path.join(output_dir, "negative")

    shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    os.mkdir(anchor_output_path)
    os.mkdir(positive_output_path)
    os.mkdir(negative_output_path)

    for i in range(sample_num):
        sample = triplet_dataset[i]
        images = sample[0]
        infos = sample[2]
        
        
        anchor_img = images[0]
        positive_img = images[1]
        negative_img = images[2]
        
        anchor_img = transforms.ToPILImage()(anchor_img)
        positive_img = transforms.ToPILImage()(positive_img)
        negative_img = transforms.ToPILImage()(negative_img)
        
        img_anchor_path = os.path.join(anchor_output_path, "{}_ep{}_sl{}.png".format(i, infos[0]["episode_id"], infos[0]["slice_id"]))
        img_positive_path = os.path.join(positive_output_path, "{}_ep{}_sl{}.png".format(i, infos[1]["episode_id"], infos[1]["slice_id"]))
        img_negative_path = os.path.join(negative_output_path, "{}_ep{}_sl{}.png".format(i, infos[2]["episode_id"], infos[2]["slice_id"]))
        
        print("{} Labels. Anchor:{}, Positive:{}, Negative:{} ".format(i, infos[0]["label"], infos[1]["label"], infos[2]["label"]))
        
        print("Vel. Anchor:{}, Positive:{}, Negative:{} ".format(infos[0]["mean_velocity"], infos[1]["mean_velocity"], infos[2]["mean_velocity"]))
        
        anchor_img.save(img_anchor_path)
        positive_img.save(img_positive_path)
        negative_img.save(img_negative_path)
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help="The base directory for the dataset.")
    parser.add_argument('--dataset_folders', type=str, help="Space separated list of top level folder names for the dataset. Each folder should include two subfolders with the names processed_data and log_files.", nargs='+')
    args = parser.parse_args()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    # transform_input = transforms.Compose([
    #     transforms.Resize((40, 40)),
    #     transforms.ToTensor(),
    # ])
    transform_input = transforms.Compose([
        # transforms.CenterCrop(1100),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # transform_input = transforms.Compose([
    #     transforms.CenterCrop(1100),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     normalize
    # ])


    # root_dir = "/scratch/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results"
    # trajectories = ["medium_sample_width_10"]

    # root_dir = "/robodata/user_data/srabiee/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results"
    # trajectories = ["cp1_00_width_10_WRoadSnow"]

    root_dir = args.root_dir
    trajectories = [ f for f in args.dataset_folders]

    triplet_dataset = AirSimTripletDataset(root_dir, trajectories,
                                             transform_input=transform_input,
                                             inference_mode=True,
                                             return_img_info=True,
                                            only_return_successful_traj=True,
                                            only_sample_from_dist_tails=False,
                                            feature_function_set_name="Clearance_Velocity_TurnRate")

    # triplet_dataset = AirSimTripletDataset(root_dir, trajectories,
    #                                         transform_input=transform_input,
    #                                         inference_mode=True,
    #                                         return_img_info=True,
    #                                     only_return_successful_traj=True,
    #                                     only_sample_from_dist_tails=False,
    #                                     feature_function_set_name="TurnRate",
    #                                     feature_function_thresholds_path="feature_function_thresholds.json")



    computed_feature_function_thresholds = triplet_dataset.get_computed_feature_function_thresholds()  

    print()
    print("Computed Feature Function Thresholds: ")
    print(computed_feature_function_thresholds)


    # Sample a number of data points and store the corresponding images
    # to file for testing the data quality and the data loader pipeline
    sample_num = 50
    output_dir = "tmp_output"
    slow_output_path = os.path.join(output_dir, "slow")
    fast_output_path = os.path.join(output_dir, "fast")
    low_turn_output_path = os.path.join(output_dir, "low_turn")
    high_turn_output_path = os.path.join(output_dir, "high_turn")
    careless_output_path = os.path.join(output_dir, "careless")
    cautious_output_path = os.path.join(output_dir, "cautious")

    shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    os.mkdir(slow_output_path)
    os.mkdir(fast_output_path)
    os.mkdir(low_turn_output_path)
    os.mkdir(high_turn_output_path)
    os.mkdir(careless_output_path)
    os.mkdir(cautious_output_path)

    feature_function_thresholds_path=os.path.join(output_dir, "feature_function_thresholds.json")
    try:
        file = open(feature_function_thresholds_path, "w")
        json.dump(computed_feature_function_thresholds, file, indent=2)
        file.close()
        print("Saved the computed feature function thresholds to " +
              str(feature_function_thresholds_path))
    except IOError:
        print("Error: can\'t find file or write data: " + feature_function_thresholds_path)
        exit()

    for i in range(sample_num):
        sample = triplet_dataset[i]
        img = sample[0]
        info = sample[2]
        img_conv = transforms.ToPILImage()(img)
        print("LABEL:", info[0]["label"])
        if "slow" in info[0]["label"]:
            print("Slow, {}m/s, episode {}, slice {}".format(info[0]['mean_velocity'], info[0]['episode_id'], info[0]['slice_id']))
            img_path = os.path.join(slow_output_path, "{}.png".format(i))
            print("image path:", img_path)
            img_conv.save(img_path)
        if "fast" in info[0]["label"]:
            print("Fast, {}m/s, episode {}, slice {}".format(info[0]['mean_velocity'], info[0]['episode_id'], info[0]['slice_id']))
            img_path = os.path.join(fast_output_path, "{}.png".format(i))
            print("image path:", img_path)
            img_conv.save(img_path)
        if "low_turn" in info[0]["label"]:
            img_path = os.path.join(low_turn_output_path, "{}.png".format(i))
            img_conv.save(img_path)
        if "high_turn" in info[0]["label"]:
            img_path = os.path.join(high_turn_output_path, "{}.png".format(i))
            img_conv.save(img_path)
        if "careless" in info[0]["label"]:
            img_path = os.path.join(careless_output_path, "{}.png".format(i))
            img_conv.save(img_path)
        if "cautious" in info[0]["label"]:
            img_path = os.path.join(cautious_output_path, "{}.png".format(i))
            img_conv.save(img_path)
        print("")



    # Samples a number of triplets and saves them to file for debugging purposes
    print("****************************************")
    print("**** Saving Sample Triplets to File ****")
    print("****************************************")
    inspect_triplets(args.root_dir, args.dataset_folders)

        
        
