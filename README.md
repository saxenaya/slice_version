# CAML Representation Learning


## Installation

### Create a conda environment
```
conda create -n pytorch-env python=3.7.7
conda activate pytorch-env

cd <path_to>task_strategy_learning
pip install -e .
```

## Training: Learning Task Strategies 

1) Run the script `datasets.py` to compute and save a set of threshold values for the feature functions given the distribution of data in the training dataset.
```
python datasets_slice.py \
--root_dir <root_dataset_directory>  \
--dataset_folders <session_names>
```   
`root_dir` is the base directory for the dataset. `dataset_folders` is a space separated list of top level folder names for the dataset. Each folder should include two subfolders with the names `processed_data` and `log_files`. The computed feature function thresholds will be saved under `tmp_output/feature_function_thresholds.json`. 

2) Run the script `main_representation.py` to learn an embedding function via contrastive learning:
```
python main_representation_slice.py \
--root_dir <root_dataset_directory> \
--trajectories <session_names> \
--ckpt_dir <checkpoint_directory> \
--embedding_dim <embedding_dimensions> \
--data_format <data_format> \
--feature_functions_name <desired_feature_functions_string_id> \
--feature_function_thresholds_path <path_to_the_feature_function_thresholds_file>
```
* `data_format` should be set to either `minigrid` or `airsim` to support the corresponding logged data formats.

* `session_names` is a space separated list of folder names that exist under `root_dataset_directory` and each contain the logged data from running the RL agent data generation script. In the `airsim` data format, each folder is expected to include the sub-folders `log_files` and `processed_data`.

* `embedding_dim` is the length of the embedding vectors with a default value of `48`. 

* `feature_functions_name` is a string ID of the desired set of feature function to be used for forming triplets. Currently supported feature function names include (case-sensitive):
                  `Clearance`, `Velocity`, `TurnRate`, `Clearance_Velocity`, `Clearance_TurnRate`, `Clearance_Velocity_TurnRate`. 

    Please take a look at the script `run_representation_learning.bash` for an examples of running the above script.


3) Run the script `collect_embeddings_minigrid.py` to cluster the traces of the RL agent when projected into the learned embedding space. Each cluster represents a task strategy.
```
python collect_embeddings_minigrid.py \
--root_dirs <root_directories> \
--model_dir <embedding_network_model_base_path> \
--save_dir <output_directory> \
--epoch <embedding_network_check_point_number> \
--embedding_dim <embedding_dimensions> \
--data_format <data_format> \
--cluster_num <number_of_clusters_to_be_extracted> \
--feature_functions_name <desired_feature_functions_string_id> \
--feature_function_thresholds_path <path_to_the_feature_function_thresholds_file>
```

  * This program runs all the data through the embedding function that was learned in the previous step, and then extracts clusters using the `k-means` algorithm. Visualization of the clusters in the embedding space and sampled RL agent trajectories from each cluster are save to file along with the cluster center information. The number of clusters to be extracted from data is user-specified and is set via the `cluster_num` argument. Make sure to use the same `feature_functions_name` that you have used for training the embedding network.
Please take a look at the script `run_collect_embeddings.bash` for an examples of running the above script.


## Test: Estimating Task Strategies for New Data
Once an embedding network and a set of clusters are learned during the training step. These information can be used at inference time to assign task strategy labels to new examples of the RL agent trajectories.

```
python infer_strategy_airsim.py \
--root_dirs <root_directories> \
--embedding_model_dir <embedding_network_model_base_path> \
--save_dir <output_directory> \
--epoch <embedding_network_check_point_number> \
--embedding_dim <embedding_dimensions> \
--clustering_model_dir <clustering_model_path> \
```
The script loads the RL agent trajectories --- expected to be in the same format as that output by the RL agent data generation script --- then applies the embedding function to the data and lastly labels each data point with the closest cluster center ID (task strategy ID). The estimated task strategy is appended to the `episode_data.json` file corresponding to each trajectory and saved as `episode_data_with_strategy.json`.


Finally, you can optionally use the `prepare_umass_data.bash` script to copy over the minimal data required for competency estimation. This includes task strategy IDs as well as the episode data.
