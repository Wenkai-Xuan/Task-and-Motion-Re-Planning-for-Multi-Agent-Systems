import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json

def latest_sequences(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()] #full paths for all subdirectories within folder_path
    # print(subfolders)
    sequences = [s for s in subfolders if s.startswith(f"{folder_path}/sequence_generation")]  # the list of all generated sequences
    plans = [s for s in subfolders if s.startswith(f"{folder_path}/sequence_plan")]
    # print(sequences)
    if not plans: # "plan_for_sequence" failed
        return None
    
    latest_sequences = max(sequences, key=os.path.getmtime)
    with open(f'{latest_sequences}/sequences.json', 'r') as file:
        seq = json.load(file)
    
    # "generate_candidate_sequences" failed may cause "plan_for_sequence" success
    if all(list(tasks.values())[0] is None for tasks in seq['sequences']): 
        return None
    
    latest_plans = max(plans, key=os.path.getmtime)

    return latest_plans

# Directory containing the JSON files
input_directory = "replan_data/replan_ini_shelf_52_20250317_143018"
output_parquet_file = f"{input_directory}/output_file.parquet"
env_name = "shelf"
old_df = pd.read_parquet('../samples_000000000_to_000001376.parquet') # read the old conveyor data
sample_indx = 52
part_num = 25 #the number of the selected drop-steps

df = old_df[sample_indx:sample_indx+1]
dfs = old_df[sample_indx:sample_indx+1]

home_initial_pose = [0, 0.5, -1.2, 0, 1.57, 1.57]

i=1
# iterate over all the replan-conifgs
for replan_config in sorted(os.listdir(input_directory)):
    replan_config_path = os.path.join(input_directory, replan_config)
    
    # only when replan_config is a directory named as drop-step, let's proceed and iterate later
    if os.path.isdir(replan_config_path) and replan_config.isdigit():
        # print(replan_config_path)
        with open(f'{replan_config_path}/{env_name}_obj_output_{sample_indx}_rela.json', 'r') as file:
            obj_file = json.load(file)
        with open(f'{replan_config_path}/{env_name}_robot_output_{sample_indx}_rela.json', 'r') as file:
            robot_file = json.load(file)

        plan_directory = latest_sequences(replan_config_path) # get the latest sequence_plan directory
        if plan_directory is None: # skip the steps that "plan_for_sequence" failed
            continue
        # print(plan_directory)
        # iterate all the plans inside the 'sequence_plan'
        for single_plan in sorted(os.listdir(plan_directory)): #plan_directory is definitely a directory
            
            single_plan_path = os.path.join(plan_directory, single_plan)
            # print(single_plan)

            if os.path.isdir(single_plan_path):
                # print(single_plan_path)
                for file in os.listdir(single_plan_path):
                    file_path = os.path.join(single_plan_path, file)
                    # print(file_path)
                    if os.path.isfile(file_path):
                        if file.startswith('metadata'):
                            with open(file_path, 'r') as f:
                                metadata = json.load(f)
                        elif file.startswith('plan'):
                            with open(file_path, 'r') as f:
                                plan = json.load(f)
                        elif file.startswith('scene'):
                            with open(file_path, 'r') as f:
                                scene = json.load(f)
                        elif file.startswith('sequence'):
                            with open(file_path, 'r') as f:
                                sequence = json.load(f)
                        elif file.startswith('trajectory'):
                            with open(file_path, 'r') as f:
                                trajectory = json.load(f)


                if env_name == 'random':
                    # original data for 'random' scene has 'home_pose'
                    for i in range(len(robot_file['robots'])):
                        robot_file['robots'][i]["home_pose"] = df['robot_file'][sample_indx]['robots'][0]['home_pose'].tolist()
                        # robot_file['robots'][i]["initial_pose"] = home_initial_pose
                    table = pd.DataFrame({'obj_file': [obj_file],
                                   'robot_file': [robot_file],
                                   'metadata': [metadata],
                                   'plan': [plan],
                                   'scene': [scene],
                                   'sequence': [sequence],
                                   'trajectory': [trajectory]}) # only 'random' scene data doesn't have 'scene_file' and 'obstacles_file'
                else:
                    table = pd.DataFrame({'obj_file': [obj_file],
                                   'robot_file': [robot_file],
                                   'metadata': [metadata],
                                   'plan': [plan],
                                   'scene': [scene],
                                   'sequence': [sequence],
                                   'trajectory': [trajectory],
                                   'scene_file': df['scene_file'].tolist(),
                                   'obstacles_file': df['obstacles_file'].tolist()}) #convert each element to a list to ensure the same length
                dfs = pd.concat([dfs, table], ignore_index=True)
            
# original data doesn't have 'parent'
for i in range(len(dfs['obj_file'][0]['objects'])):
    dfs['obj_file'][0]['objects'][i]["parent"] = "table"
# dfs.to_parquet('example.parquet')

if env_name != 'random': # original data for 'conveyor', 'husky' and 'shelf' scenes don't have 'start_pose' we add it manually
    for i in range(len(dfs['robot_file'][0]['robots'])):
        dfs['robot_file'][0]['robots'][i]["start_pose"] = home_initial_pose #this value is stored in 'scene'

dfs.to_parquet(f"{input_directory}/{env_name}_{sample_indx}_rela_{part_num}.parquet")

new_dfs = pd.read_parquet(f"{input_directory}/{env_name}_{sample_indx}_rela_{part_num}.parquet")

train = new_dfs[ : int(len(new_dfs)*0.8)]
test = new_dfs[int(len(new_dfs)*0.8) : ]
train.to_parquet(f"{input_directory}/{env_name}_{sample_indx}_rela_{part_num}_train.parquet")
test.to_parquet(f"{input_directory}/{env_name}_{sample_indx}_rela_{part_num}_test.parquet")