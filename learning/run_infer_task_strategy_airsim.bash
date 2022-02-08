#!/bin/bash


# ===================================================================
# ==================== 7_21_21 ===================
# ==================== AirSim =======================================

#         # self.mean_clearance_high_thresh = 0.11 
#         # self.mean_clearance_low_thresh = 0.07
#         # self.mean_clearance_mid_thresh = 0.09
#         # self.mean_norm_clearance_high_thresh = 0.11 
#         # self.mean_norm_clearance_low_thresh = 0.07
#         # self.mean_norm_clearance_mid_thresh = 0.09
#         # self.mean_velocity_high_thresh = 7.5
#         # self.mean_velocity_low_thresh = 5.0
#         # self.mean_velocity_mid_thresh = 6.87
#         # self.turn_rate_high_thresh = 1.1
#         # self.turn_rate_low_thresh = 0.9
#         # self.turn_rate_mid_thresh = 1.0
#         # self.action_count_high_thresh = 60
#         # self.action_count_low_thresh = 40
#         # self.action_count_mid_thresh = 50
#         # self.class_labels = ['slow',
#         #                      'fast']
#         # self.active_feature_functions = [self.feature_func_velocity]
#         # self.image_folder_name = "full_episodes"
#         # batch_size = 1
#         # lr = 1e-4
#         # optimizer = optim.Adam(model.parameters(), lr=lr)
#         # scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
# ROOT_DIR=("/scratch/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results/small_sample_width_10_test")
# EMBEDDING_MODEL_DIR="/scratch/Projects/CAML/Task_Strategy/7_21_21/AirSim_RL/nn_models/02"
# EPOCH="10" 
# CLUSTERING_MODEL_DIR="/scratch/Projects/CAML/Task_Strategy/7_21_21/AirSim_RL/representation/embeddings_02/kmeans.pkl"
# SAVE_DIR="/scratch/Projects/CAML/Task_Strategy/7_21_21/AirSim_RL/representation/embeddings_02"
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
        # only_return_successful_traj=False
# ROOT_DIR=("/robodata/user_data/srabiee/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results/cp1_00_width_10_WRoadSnow")
ROOT_DIR=("/robodata/user_data/srabiee/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results/cp4_00_width_10_WRoadSnow")

EMBEDDING_MODEL_DIR="/robodata/user_data/srabiee/Projects/CAML/Task_Strategy/8_10_21/AirSim_RL/nn_models/05"
EPOCH="30" 
CLUSTERING_MODEL_DIR="/robodata/user_data/srabiee/Projects/CAML/Task_Strategy/8_10_21/AirSim_RL/representation/embeddings_05_full_6cluster/kmeans.pkl"
SAVE_DIR="/robodata/user_data/srabiee/Projects/CAML/Task_Strategy/8_10_21/AirSim_RL/representation/embeddings_05_full_6cluster"

EMBEDDING_DIM="48"


ALL_ROOT_DIRS_STR=""
for path in "${ROOT_DIR[@]}"; do 
  echo $path
  ALL_ROOT_DIRS_STR+=" $path"
done

CUDA_VISIBLE_DEVICES=1  python infer_strategy_airsim.py \
--root_dirs $ALL_ROOT_DIRS_STR \
--embedding_model_dir $EMBEDDING_MODEL_DIR \
--save_dir $SAVE_DIR \
--epoch $EPOCH \
--embedding_dim $EMBEDDING_DIM \
--clustering_model_dir $CLUSTERING_MODEL_DIR \