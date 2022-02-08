#!/bin/bash


# ==================== 6_4_21 ===================

        # self.mean_clearance_high_thresh = 5.0
        # self.mean_clearance_low_thresh = 3.2 
        # self.mean_clearance_mid_thresh = 4.18 
        # self.mean_norm_clearance_high_thresh = 1.2 
        # self.mean_norm_clearance_low_thresh = 0.8 
        # self.mean_norm_clearance_mid_thresh = 1.0 
        # self.mean_velocity_high_thresh = 2.2 
        # self.mean_velocity_low_thresh = 1.8 
        # self.mean_velocity_mid_thresh = 2
        # self.class_labels = ['careless',
        #                     'cautious']
        # self.active_feature_functions = [self.feature_func_clearance]
        # self.image_folder_name = "full_episodes_no_color"
        # batch_size = 10
# ROOT_DIR=("/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/UTData104" 
# "/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/UTData204" 
# "/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/UTData304" 
# "/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/UTData404"
# )
# ROOT_DIR=("/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/UTDataTest500"
# )
ROOT_DIR=(
"/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/UMASS_Data_0/UTData120" 
"/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/UMASS_Data_0/UTData220" 
"/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/UMASS_Data_0/UTData320" 
"/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/UMASS_Data_0/UTData420" 
)

EMBEDDING_MODEL_DIR="/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/representation/nn_models_5"
EPOCH="30" 
CLUSTERING_MODEL_DIR="/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/representation/embeddings_5/kmeans.pkl"
SAVE_DIR="/robodata/srabiee/CAML/Task_Strategy/6_4_21/ccne21_airsim/representation/task_strategy_5_t"

EMBEDDING_DIM="48"


ALL_ROOT_DIRS_STR=""
for path in "${ROOT_DIR[@]}"; do 
  echo $path
  ALL_ROOT_DIRS_STR+=" $path"
done

CUDA_VISIBLE_DEVICES=1  python infer_strategy.py \
--root_dirs $ALL_ROOT_DIRS_STR \
--embedding_model_dir $EMBEDDING_MODEL_DIR \
--save_dir $SAVE_DIR \
--epoch $EPOCH \
--embedding_dim $EMBEDDING_DIM \
--clustering_model_dir $CLUSTERING_MODEL_DIR 