#!/bin/bash


# ===================================================================
# ==================== 6_4_21 ===================

# ROOT_DIR=("/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/UTData104" 
# "/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/UTData204" 
# "/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/UTData304" 
# "/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/UTData404"
# )
# MODEL_DIR="/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/representation/nn_models_5"
# EPOCH="30" 
# SAVE_DIR="/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/representation/embeddings_5_t"

# EMBEDDING_DIM="48"


# ===================================================================
# ==================== 6_10_21 ===================

# ======  ccne21_airsim3 RL model
# *********
# ROOT_DIR=(
# "/robodata/srabiee/CAML/Task_Strategy/6_10_21/ccne21_airsim3/UTData101" 
# "/robodata/srabiee/CAML/Task_Strategy/6_10_21/ccne21_airsim3/UTData201" 
# "/robodata/srabiee/CAML/Task_Strategy/6_10_21/ccne21_airsim3/UTData301" 
# "/robodata/srabiee/CAML/Task_Strategy/6_10_21/ccne21_airsim3/UTData401"
# )

# MODEL_DIR="/robodata/srabiee/CAML/Task_Strategy/6_10_21/ccne21_airsim3/representation/nn_models_10"
# EPOCH="45" 
# SAVE_DIR="/robodata/srabiee/CAML/Task_Strategy/6_10_21/ccne21_airsim3/representation/embeddings_10_t"

# EMBEDDING_DIM="48"



# ======  scne21_airsim RL model
# ROOT_DIR=(
# "/robodata/srabiee/CAML/Task_Strategy/6_10_21/scne21_airsim/UTData101" 
# "/robodata/srabiee/CAML/Task_Strategy/6_10_21/scne21_airsim/UTData201" 
# "/robodata/srabiee/CAML/Task_Strategy/6_10_21/scne21_airsim/UTData301" 
# "/robodata/srabiee/CAML/Task_Strategy/6_10_21/scne21_airsim/UTData401"
# )

# MODEL_DIR="/robodata/srabiee/CAML/Task_Strategy/6_10_21/scne21_airsim/representation/nn_models_1"
# EPOCH="45" 
# SAVE_DIR="/robodata/srabiee/CAML/Task_Strategy/6_10_21/scne21_airsim/representation/embeddings_1_t"

# EMBEDDING_DIM="48"


# ===================================================================
# ==================== 7_21_21 ===================
# ==================== AirSim =======================================


        # self.mean_clearance_high_thresh = 0.11 
        # self.mean_clearance_low_thresh = 0.07
        # self.mean_clearance_mid_thresh = 0.09
        # self.mean_norm_clearance_high_thresh = 0.11 
        # self.mean_norm_clearance_low_thresh = 0.07
        # self.mean_norm_clearance_mid_thresh = 0.09
        # self.mean_velocity_high_thresh = 7.5
        # self.mean_velocity_low_thresh = 5.0
        # self.mean_velocity_mid_thresh = 6.87
        # self.turn_rate_high_thresh = 1.1
        # self.turn_rate_low_thresh = 0.9
        # self.turn_rate_mid_thresh = 1.0
        # self.action_count_high_thresh = 60
        # self.action_count_low_thresh = 40
        # self.action_count_mid_thresh = 50
        # self.class_labels = ['slow',
        #                      'fast']
        # self.active_feature_functions = [self.feature_func_velocity]
        # self.image_folder_name = "full_episodes"
        # batch_size = 1
        # lr = 1e-4
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        # scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
# MODEL_DIR="/scratch/Projects/CAML/Task_Strategy/7_21_21/AirSim_RL/nn_models/02"
# EPOCH="10"

# ROOT_DIR=("/scratch/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results/medium_sample_width_10")
# SAVE_DIR="/scratch/Projects/CAML/Task_Strategy/7_21_21/AirSim_RL/representation/embeddings_02"
# # SAVE_DIR="/scratch/Projects/CAML/Task_Strategy/7_21_21/AirSim_RL/representation/embeddings_02_unnormalized"
# # SAVE_DIR="/scratch/Projects/CAML/Task_Strategy/7_21_21/AirSim_RL/representation/embeddings_02_full"
# # SAVE_DIR="/scratch/Projects/CAML/Task_Strategy/7_21_21/AirSim_RL/representation/embeddings_02_full_3cluster"

# # Test data 
# # ROOT_DIR=("/scratch/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results/small_sample_width_10_test")
# # SAVE_DIR="/scratch/Projects/CAML/Task_Strategy/7_21_21/AirSim_RL/representation/embeddings_02_small_test"

# DATA_FORMAT="airsim"
# EMBEDDING_DIM="48"


# ===================================================================
# ==================== 8_10_21 ===================
# ==================== AirSim =======================================


        # self.mean_clearance_high_thresh = 8.93 # 7.0
        # self.mean_clearance_low_thresh = 6.85 # 5.0
        # self.mean_clearance_mid_thresh = 7.74 # 6.0
        # self.mean_norm_clearance_high_thresh = 0.11 
        # self.mean_norm_clearance_low_thresh = 0.07
        # self.mean_norm_clearance_mid_thresh = 0.09
        # self.mean_velocity_high_thresh = 4.76 # 4.0
        # self.mean_velocity_low_thresh = 4.6 # 4.0
        # self.mean_velocity_mid_thresh = 4.66 # 4.0
        # self.turn_rate_high_thresh = 0.08 # 0.5
        # self.turn_rate_low_thresh = 0.054 # 0.3
        # self.turn_rate_mid_thresh = 0.067 # 0.4
        # self.action_count_high_thresh = 60
        # self.action_count_low_thresh = 40
        # self.action_count_mid_thresh = 50
        # self.class_labels = ['careless_slow_low_turn',
        #                      'cautious_slow_low_turn',
        #                      'careless_fast_low_turn',
        #                      'cautious_fast_low_turn',
        #                      'careless_slow_high_turn',
        #                      'cautious_slow_high_turn',
        #                      'careless_fast_high_turn',
        #                      'cautious_fast_high_turn']
        # self.active_feature_functions = [self.feature_func_clearance,   
        #                                  self.feature_func_velocity,
        #                                  self.feature_func_turn_rate]
        # self.image_folder_name = "full_episodes"
        # batch_size = 1
        # lr = 1e-4
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        # scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
        # only_return_successl_traj=False
MODEL_DIR="./checkpoints"
EPOCH="9"

ROOT_DIR=("/robodata/user_data/saxenaya/CAML/ut_data_5_27_22")
SAVE_DIR="./save_dir"

CLUSTER_NUM="5"
FEATURE_FUNCTION_NAME="Clearance_Velocity_TurnRate"
FEATURE_FUNCTION_THRESH_PATH="./tmp_output/feature_function_thresholds.json"
DATA_FORMAT="airsim"
EMBEDDING_DIM="48"


ALL_ROOT_DIRS_STR=""
for path in "${ROOT_DIR[@]}"; do 
  echo $path
  ALL_ROOT_DIRS_STR+=" $path"
done


CUDA_VISIBLE_DEVICES=0,1  python3 collect_embeddings_minigrid.py \
--root_dirs $ALL_ROOT_DIRS_STR \
--model_dir $MODEL_DIR \
--save_dir $SAVE_DIR \
--epoch $EPOCH \
--embedding_dim $EMBEDDING_DIM \
--data_format $DATA_FORMAT \
--cluster_num $CLUSTER_NUM \
--feature_functions_name $FEATURE_FUNCTION_NAME \
--feature_function_thresholds_path $FEATURE_FUNCTION_THRESH_PATH
