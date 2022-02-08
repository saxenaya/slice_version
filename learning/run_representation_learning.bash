#!/bin/bash


# ===================================================================
# ==================== 6_4_21 ===================

#         # self.mean_clearance_high_thresh = 5.0 
#         # self.mean_clearance_low_thresh = 3.2 
#         # self.mean_clearance_mid_thresh = 4.18 
#         # self.mean_norm_clearance_high_thresh = 1.2 
#         # self.mean_norm_clearance_low_thresh = 0.8 
#         # self.mean_norm_clearance_mid_thresh = 1.0 
#         # self.mean_velocity_high_thresh = 2.2 
#         # self.mean_velocity_low_thresh = 1.8 
#         # self.mean_velocity_mid_thresh = 2
#         # self.class_labels = ['careless',
#         #                     'cautious']
#         # self.active_feature_functions = [self.feature_func_clearance]
#         # self.image_folder_name = "full_episodes_no_color"
#         # batch_size = 10
# ROOT_DIR="/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim"
# SESSION_NAMES="UTData110 UTData111 UTData112 UTData113 UTData210 UTData211 UTData212 UTData213 UTData310 UTData311 UTData312 UTData313 UTData410 UTData411 UTData412 UTData413"
# # CKPT_DIR="/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/representation/nn_models_5_t"

# DATA_FORMAT="minigrid"
# EMBEDDING_DIM="48"


# ===================================================================
# ==================== 6_10_21 ===================

# ======  ccne21_airsim3 RL model
# *******

#         # self.mean_clearance_high_thresh = 5.0 
#         # self.mean_clearance_low_thresh = 3.2 
#         # self.mean_clearance_mid_thresh = 4.18 
#         # self.mean_norm_clearance_high_thresh = 1.2 
#         # self.mean_norm_clearance_low_thresh = 0.8 
#         # self.mean_norm_clearance_mid_thresh = 1.0 
#         # self.mean_velocity_high_thresh = 2.0 
#         # self.mean_velocity_low_thresh = 1.8 
#         # self.class_labels = ['careless',
#         #                    'cautious']
#         # self.active_feature_functions = [self.feature_func_clearance]
#         # self.image_folder_name = "full_episodes"
#         # batch_size = 10
#         # lr = 1e-4
#         # optimizer = optim.Adam(model.parameters(), lr=lr)
#         # scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)

# ROOT_DIR="/robodata/srabiee/CAML/Task_Strategy0/6_10_21/ccne21_airsim3"
# SESSION_NAMES="UTData100 UTData200 UTData300 UTData400"
# CKPT_DIR="/robodata/srabiee/CAML/Task_Strategy/6_10_21/ccne21_airsim3/representation/nn_models_8_t"

# DATA_FORMAT="minigrid"
# EMBEDDING_DIM="48"


# *******

#         # self.mean_clearance_high_thresh = 5.0 
#         # self.mean_clearance_low_thresh = 3.2 
#         # self.mean_clearance_mid_thresh = 4.18 
#         # self.mean_norm_clearance_high_thresh = 1.2 
#         # self.mean_norm_clearance_low_thresh = 0.8 
#         # self.mean_norm_clearance_mid_thresh = 1.0 
#         # self.mean_velocity_high_thresh = 2.1 
#         # self.mean_velocity_low_thresh = 1.6 
#         # self.mean_velocity_mid_thresh = 1.8
#         # self.class_labels = ['slow',
#         #                      'fast']
#         # self.active_feature_functions = [self.feature_func_velocity]
#         # self.image_folder_name = "full_episodes"
#         # batch_size = 10
#         # lr = 1e-4
#         # optimizer = optim.Adam(model.parameters(), lr=lr)
#         # scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)

# ROOT_DIR="/robodata/srabiee/CAML/Task_Strategy/6_10_21/ccne21_airsim3"
# SESSION_NAMES="UTData100 UTData200 UTData300 UTData400"
# CKPT_DIR="/robodata/srabiee/CAML/Task_Strategy/6_10_21/ccne21_airsim3/representation/nn_models_10_t"

# DATA_FORMAT="minigrid"
# EMBEDDING_DIM="48"


# *******

#         # self.mean_clearance_high_thresh = 5.0 
#         # self.mean_clearance_low_thresh = 3.2 
#         # self.mean_clearance_mid_thresh = 4.18 
#         # self.mean_norm_clearance_high_thresh = 1.2 
#         # self.mean_norm_clearance_low_thresh = 0.8 
#         # self.mean_norm_clearance_mid_thresh = 1.0 
#         # self.mean_velocity_high_thresh = 2.1 
#         # self.mean_velocity_low_thresh = 1.6 
#         # self.mean_velocity_mid_thresh = 1.8
#         # self.class_labels = ['careless_slow',
#         #                     'cautious_slow',
#         #                     'careless_fast',
#         #                     'cautious_fast']
#         # self.active_feature_functions = [self.feature_func_clearance,   
#         #                                 self.feature_func_velocity]
#         # self.image_folder_name = "full_episodes"
#         # batch_size = 10
#         # lr = 1e-4
#         # optimizer = optim.Adam(model.parameters(), lr=lr)
#         # scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)

# ROOT_DIR="/robodata/srabiee/CAML/Task_Strategy/6_10_21/ccne21_airsim3"
# SESSION_NAMES="UTData100 UTData200 UTData300 UTData400"
# CKPT_DIR="/robodata/srabiee/CAML/Task_Strategy/6_10_21/ccne21_airsim3/representation/nn_models_12_t"

# DATA_FORMAT="minigrid"
# EMBEDDING_DIM="48"


# ======  scne21_airsim RL model

        # self.mean_clearance_high_thresh = 5.0 
        # self.mean_clearance_low_thresh = 4.0 
        # self.mean_clearance_mid_thresh = 4.6
        # self.mean_norm_clearance_high_thresh = 1.2 
        # self.mean_norm_clearance_low_thresh = 0.8 
        # self.mean_norm_clearance_mid_thresh = 1.0 
        # self.mean_velocity_high_thresh = 2.0
        # self.mean_velocity_low_thresh = 1.78
        # self.mean_velocity_mid_thresh = 1.9
        # self.class_labels = ['careless_slow',
        #                      'cautious_slow',
        #                      'careless_fast',
        #                      'cautious_fast']
        # self.active_feature_functions = [self.feature_func_clearance,   
        #                                  self.feature_func_velocity]
        # self.image_folder_name = "full_episodes_no_color"
        # batch_size = 10
# ROOT_DIR="/robodata/srabiee/CAML/Task_Strategy/6_10_21/scne21_airsim"
# SESSION_NAMES="UTData100 UTData200 UTData300 UTData400"
# CKPT_DIR="/robodata/srabiee/CAML/Task_Strategy/6_10_21/scne21_airsim/representation/nn_models_0_t"

# DATA_FORMAT="minigrid"
# EMBEDDING_DIM="48"

# ******

        # self.mean_clearance_high_thresh = 5.0 
        # self.mean_clearance_low_thresh = 4.0 
        # self.mean_clearance_mid_thresh = 4.6 
        # self.mean_norm_clearance_high_thresh = 1.2 
        # self.mean_norm_clearance_low_thresh = 0.8
        # self.mean_norm_clearance_mid_thresh = 1.0 
        # self.mean_velocity_high_thresh = 2.0
        # self.mean_velocity_low_thresh = 1.78
        # self.mean_velocity_mid_thresh = 1.9
        # self.class_labels = ['slow',
        #                      'fast']
        # self.active_feature_functions = [self.feature_func_velocity]
        # self.image_folder_name = "full_episodes"
        # batch_size = 10
        # lr = 1e-4
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        # scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
# ROOT_DIR="/robodata/srabiee/CAML/Task_Strategy/6_10_21/scne21_airsim"
# SESSION_NAMES="UTData100 UTData200 UTData300 UTData400"
# CKPT_DIR="/robodata/srabiee/CAML/Task_Strategy/6_10_21/scne21_airsim/representation/nn_models_1_t"

# DATA_FORMAT="minigrid"
# EMBEDDING_DIM="48"



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
# ROOT_DIR="/scratch/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results"
# SESSION_NAMES="medium_sample_width_10"
# CKPT_DIR="/scratch/Projects/CAML/Task_Strategy/7_21_21/AirSim_RL/nn_models/02"

# DATA_FORMAT="airsim"
# EMBEDDING_DIM="48"


# ===================================================================
# ==================== 8_10_21 ===================
# ==================== AirSim =======================================

#         # self.mean_clearance_high_thresh = 8.93 # 7.0
#         # self.mean_clearance_low_thresh = 6.85 # 5.0
#         # self.mean_clearance_mid_thresh = 7.74 # 6.0
#         # self.mean_norm_clearance_high_thresh = 0.11 
#         # self.mean_norm_clearance_low_thresh = 0.07
#         # self.mean_norm_clearance_mid_thresh = 0.09
#         # self.mean_velocity_high_thresh = 4.76 # 4.0
#         # self.mean_velocity_low_thresh = 4.6 # 4.0
#         # self.mean_velocity_mid_thresh = 4.66 # 4.0
#         # self.turn_rate_high_thresh = 0.08 # 0.5
#         # self.turn_rate_low_thresh = 0.054 # 0.3
#         # self.turn_rate_mid_thresh = 0.067 # 0.4
#         # self.action_count_high_thresh = 60
#         # self.action_count_low_thresh = 40
#         # self.action_count_mid_thresh = 50
#         # self.class_labels = ['careless_slow_low_turn',
#         #                      'cautious_slow_low_turn',
#         #                      'careless_fast_low_turn',
#         #                      'cautious_fast_low_turn',
#         #                      'careless_slow_high_turn',
#         #                      'cautious_slow_high_turn',
#         #                      'careless_fast_high_turn',
#         #                      'cautious_fast_high_turn']
#         # self.active_feature_functions = [self.feature_func_clearance,   
#         #                                  self.feature_func_velocity,
#         #                                  self.feature_func_turn_rate]
#         # self.image_folder_name = "full_episodes"
#         # batch_size = 1
#         # lr = 1e-4
#         # optimizer = optim.Adam(model.parameters(), lr=lr)
#         # scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
#         # only_return_successful_traj=False
# ROOT_DIR="/robodata/user_data/srabiee/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results"
# SESSION_NAMES="cp1_00_width_10_WRoadSnow"
# CKPT_DIR="/robodata/user_data/srabiee/Projects/CAML/Task_Strategy/8_10_21/AirSim_RL/nn_models/05"

# FEATURE_FUNCTION_NAME="Clearance_Velocity_TurnRate"
# FEATURE_FUNCTION_THRESH_PATH="tmp_output/feature_function_thresholds.json"
# DATA_FORMAT="airsim"
# EMBEDDING_DIM="48"



# ===================================================================
# ==================== 8_27_21 ===================
# ==================== AirSim =======================================

        # "mean_clearance_high_thresh": 9.935221526856765,
        # "mean_clearance_low_thresh": 6.994573419744318,
        # "mean_clearance_mid_thresh": 8.745142177946176,
        # "mean_velocity_high_thresh": 4.5964437985219,
        # "mean_velocity_low_thresh": 2.4951523878097555,
        # "mean_velocity_mid_thresh": 4.549956090582633,
        # "turn_rate_high_thresh": 0.05768300060496068,
        # "turn_rate_low_thresh": 0.0,
        # "turn_rate_mid_thresh": 0.03500837963456111
        # self.image_folder_name = "full_episodes"
        # batch_size = 1
        # lr = 1e-4
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        # scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)
        # only_return_successful_traj=False
ROOT_DIR="/robodata/user_data/srabiee/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results"
SESSION_NAMES="cp3_01_width_10_v3"
CKPT_DIR="/robodata/user_data/srabiee/Projects/CAML/Task_Strategy/8_27_21/AirSim_RL/nn_models/00"

FEATURE_FUNCTION_NAME="Clearance_Velocity_TurnRate"
FEATURE_FUNCTION_THRESH_PATH="tmp_output/feature_function_thresholds.json"
DATA_FORMAT="airsim"
EMBEDDING_DIM="48"


python main_representation.py \
--root_dir $ROOT_DIR \
--trajectories $SESSION_NAMES \
--ckpt_dir $CKPT_DIR \
--embedding_dim $EMBEDDING_DIM \
--data_format $DATA_FORMAT \
--feature_functions_name $FEATURE_FUNCTION_NAME \
--feature_function_thresholds_path $FEATURE_FUNCTION_THRESH_PATH