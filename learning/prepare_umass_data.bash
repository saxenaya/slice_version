#!/bin/bash

# SOURCE_DATASET_DIR="/scratch/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results/medium_sample_width_10"
# SOURCE_DATASET_DIR="/scratch/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results/small_sample_width_10_test"
# SOURCE_DATASET_DIR="/robodata/user_data/srabiee/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results/cp1_00_width_10_WRoadSnow"
# SOURCE_DATASET_DIR="/robodata/user_data/srabiee/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results/cp4_00_width_10_WRoadSnow"

# SOURCE_DATASET_DIR="/robodata/user_data/srabiee/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results/cp3_01_width_10_v3"
# SOURCE_DATASET_DIR="/robodata/user_data/srabiee/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results/cp1_00_width_10_v3"
# SOURCE_DATASET_DIR="/robodata/user_data/srabiee/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results/cp1_01_width_10_v3"

# QPR Oct 2021
# SOURCE_DATASET_DIR="/robodata/user_data/srabiee/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results_3/cp6_00_6cond_width_10_cra_v5"

# QPR Feb 2022
# SOURCE_DATASET_DIR="/robodata/user_data/srabiee/Projects/CAML/alpaca-core/airsim_data_collection/AirSim_RL_results_4/ut_data_split/test"
SOURCE_DATASET_DIR="/robodata/user_data/saxenaya/CAML/ut_data_2450"

# Copy the following list of files from each episode
FILES_OF_INTEREST=(
  "episode_data_with_strategy.json"
  "step_data.json")

TARGET_PATH="$SOURCE_DATASET_DIR/umass_data"
mkdir -p $TARGET_PATH

EPISODES=$( ls "$SOURCE_DATASET_DIR/" )

for episode in $EPISODES; do
  # echo $episode
  episode_path="$SOURCE_DATASET_DIR/$episode"

  target_path=$TARGET_PATH/$episode
  mkdir -p $target_path

  step_data_file_path="$episode_path/step_data.json"

  skip_episode="0"
  for file in ${FILES_OF_INTEREST[@]}; do
    if [ ! -f "$episode_path/$file" ]; then
      echo "Not all files were available for episode $episode. Skipping this one!"
      skip_episode="1"
      break
    fi
  done

  if [ "$skip_episode" == "1" ]; then
    echo "SKIPPING ..........."
    continue
  fi

  for file in ${FILES_OF_INTEREST[@]}; do
    # echo "cp $episode_path/$file  $target_path/"
    cp $episode_path/$file  $target_path/
  done
done


echo "Compressing the results ..."
pushd $TARGET_PATH/..
session_name="$(basename -- $SOURCE_DATASET_DIR)"
zip -r "umass_data.zip" "umass_data"
popd



