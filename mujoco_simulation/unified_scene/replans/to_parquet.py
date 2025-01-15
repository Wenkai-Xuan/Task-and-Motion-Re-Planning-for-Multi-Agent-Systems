import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json

# Directory containing the JSON files
input_directory = "sequence_plan_20241217_185149_conveyor_rela"
output_parquet_file = "output_file.parquet"

# List to hold DataFrames
dataframes = []

conveyor_5_rela = pa.schema([
('obj_file', pa.string()),
('robot_file', pa.string()),
('metadata', pa.string()),
('plan', pa.string()),
('scene', pa.string()),
('sequence', pa.string()),
('trajectory', pa.string())
])

with open('conveyor_obj_output_5_rela_n.json', 'r') as file:
    obj_file = json.load(file)

with open('conveyor_robot_output_5_rela_n.json', 'r') as file:
    robot_file = json.load(file)

# Iterate over all JSON files in the directory
# for file in os.listdir(input_directory):
#     match file.startswith():
#         case 'metadata':
#             with open(file, 'r') as f:
#                 metadata = json.load(f)
#         case 'plan':
#             with open(file, 'r') as f:
#                 plan = json.load(f)
#         case 'scene':
#             with open(file, 'r') as f:
#                 scene = json.load(f)
#         case 'sequence':
#             with open(file, 'r') as f:
#                 sequence = json.load(f)
#         case 'trajectory':
#             with open(file, 'r') as f:
#                 trajectory = json.load(f)

# for (root,dirs,files) in os.walk(input_directory, topdown=True):
#     for file in files:
#         if file.startswith('metadata'):
#             filepath = os.path.join(root, file)
#             with open(filepath, 'r') as f:
#                 metadata = json.load(f)
#         elif file.startswith('plan'):
#             filepath = os.path.join(root, file)
#             with open(filepath, 'r') as f:
#                 plan = json.load(f)
#         elif file.startswith('scene'):
#             filepath = os.path.join(root, file)
#             with open(filepath, 'r') as f:
#                 scene = json.load(f)
#         elif file.startswith('sequence'):
#             filepath = os.path.join(root, file)
#             with open(filepath, 'r') as f:
#                 sequence = json.load(f)
#         elif file.startswith('trajectory'):
#             filepath = os.path.join(root, file)
#             with open(filepath, 'r') as f:
#                 trajectory = json.load(f)
    
#     batch = pa.RecordBatch.from_arrays(
#         [obj_file, robot_file, metadata, plan, scene, sequence, trajectory],
#         names=conveyor_5_rela.names
#     )
    
#     table = pa.Table.from_batches([batch])  # Create a Table from the RecordBatch
#     dataframes.append(table)

for subfolder_name in sorted(os.listdir(input_directory)):
    subfolder_path = os.path.join(input_directory, subfolder_name)

    if os.path.isdir(subfolder_path):
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            
            if os.path.isfile(file_path):
                if file.startswith('metadata'):
                    with open(file_path, 'r') as f:
                        metadata = json.load(f)
                elif file.startswith('plan'):
                    with open(file_path, 'r') as f:
                        plan = json.load(f)
                    for i in range(len(plan)):
                        plan[i]['tasks'] = pa.array(plan[i]['tasks'])
                elif file.startswith('scene'):
                    with open(file_path, 'r') as f:
                        scene = json.load(f)
                    scene = pa.array(scene)
                elif file.startswith('sequence'):
                    with open(file_path, 'r') as f:
                        sequence = json.load(f)
                    sequence = pa.array(sequence)
                elif file.startswith('trajectory'):
                    with open(file_path, 'r') as f:
                        trajectory = json.load(f)
                    trajectory = pa.array(trajectory)
        
    batch = pa.RecordBatch.from_arrays(
        [obj_file, robot_file, metadata, plan, scene, sequence, trajectory],
        names=conveyor_5_rela.names
    )
    table = pa.Table.from_batches([batch])
    dataframes.append(table)

pq.write_table(dataframes, 'conveyor_5_rela.parquet')

