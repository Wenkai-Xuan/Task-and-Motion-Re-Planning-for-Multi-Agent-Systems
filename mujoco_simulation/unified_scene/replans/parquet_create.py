import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json

# Directory containing the JSON files
input_directory = "sequence_plan_20241217_185149_conveyor_rela"
output_parquet_file = "output_file.parquet"
de = pd.read_parquet('../samples_000080000_to_000080868.parquet')
df = de[0:100].copy()



with open('conveyor_obj_output_5_rela_n.json', 'r') as file:
    obj_file = json.load(file)

with open('conveyor_robot_output_5_rela_n.json', 'r') as file:
    robot_file = json.load(file)


i=0

for subfolder_name in sorted(os.listdir(input_directory)):
    subfolder_path = os.path.join(input_directory, subfolder_name)

    if os.path.isdir(subfolder_path):
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            
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
    
    i+=1

df.to_parquet('conveyor_5_rela.parquet')

