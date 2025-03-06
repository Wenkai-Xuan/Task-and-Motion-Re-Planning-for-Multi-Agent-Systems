import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json

def latest_sequences(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()] #full paths for all subdirectories within folder_path
    # print(subfolders)
    sequences = [s for s in subfolders if s.startswith(f"{folder_path}/sequence_plan")]  # the list of all generated sequences
    # print(sequences)
    if not sequences:
        return None
    
    latest_sequences = max(sequences, key=os.path.getmtime)

    return latest_sequences

# Directory containing the JSON files
input_directory = "replan_data/replan_ini_conveyor_5_20250225_201416"
output_parquet_file = f"{input_directory}/output_file.parquet"
de = pd.read_parquet('../samples_000080000_to_000080868.parquet') # read any parquet to replace with the new data
df = de[0:470].copy()


old_df = pd.read_parquet('../samples_000010000_to_000012113.parquet') # read the old conveyor data
sample_indx = 5
for key in ['metadata', 'plan', 'scene', 'obj_file', 'robot_file', 'sequence', 'trajectory']:
    df[key][0] = old_df[key][sample_indx]
#df[0:1] = old_df[sample_indx:sample_indx+1]
df['obj_file'][0]['objects'][0]["parent"] = "table"
df['obj_file'][0]['objects'][1]["parent"] = "table"
i=1
# iterate over all the replan-conifgs
for replan_config in sorted(os.listdir(input_directory)):
    replan_config_path = os.path.join(input_directory, replan_config)
    
    # only when replan_config is a directory, let's proceed and iterate later
    if os.path.isdir(replan_config_path):
        # print(replan_config_path)
        with open(f'{replan_config_path}/conveyor_obj_output_5_rela.json', 'r') as file:
            obj_file = json.load(file)
        with open(f'{replan_config_path}/conveyor_robot_output_5_rela.json', 'r') as file:
            robot_file = json.load(file)

        plan_directory = latest_sequences(replan_config_path) # get the latest sequence_plan directory
        # print(plan_directory)
        # iterate all the plans inside the 'sequence_plan'
        for single_plan in sorted(os.listdir(plan_directory)): #plan_directory is definitely a directory
            
            single_plan_path = os.path.join(plan_directory, single_plan)
            # print(single_plan)
            df['obj_file'][i] = obj_file
            df['robot_file'][i] = robot_file

            if os.path.isdir(single_plan_path):
                # print(single_plan_path)
                for file in os.listdir(single_plan_path):
                    file_path = os.path.join(single_plan_path, file)
                    # print(file_path)
                    if os.path.isfile(file_path):
                        if file.startswith('metadata'):
                            with open(file_path, 'r') as f:
                                metadata = json.load(f)
                                df['metadata'][i] = metadata
                        elif file.startswith('plan'):
                            with open(file_path, 'r') as f:
                                plan = json.load(f)
                                df['plan'][i] = plan
                        elif file.startswith('scene'):
                            with open(file_path, 'r') as f:
                                scene = json.load(f)
                                df['scene'][i] = scene
                        elif file.startswith('sequence'):
                            with open(file_path, 'r') as f:
                                sequence = json.load(f)
                                df['sequence'][i] = sequence
                        elif file.startswith('trajectory'):
                            with open(file_path, 'r') as f:
                                trajectory = json.load(f)
                                df['trajectory'][i] = trajectory
            # print(i)
            i+=1

# for entry in list(['obj_file', 'robot_file', 'metadata', 'plan', 'scene', 'sequence', 'trajectory']):
#     df[entry][100] = old_df[entry][sample_indx]
print(i)
new_df = df[0:i].copy()
# new_df_train = df[0:int(0.8*i)].copy()
# new_df_test = df[int(0.8*i):i].copy()
# if isinstance(new_df_train, pd.DataFrame):
#     print('Dataframe loaded')
# nnew_df=new_df.to_dict(orient="records")
# nnew_df_train=new_df_train.to_dict(orient="records")
# nnew_df_test=new_df_test.to_dict(orient="records")
df.to_parquet(f"{input_directory}/conveyor_5_rela.parquet")
# nnew_df_train.to_parquet(f"{input_directory}/conveyor_5_rela_train.parquet")
# nnew_df_test.to_parquet(f"{input_directory}/conveyor_5_rela_test.parquet")