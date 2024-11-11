import pandas as pd
import json
import numpy as np
import os

# print(os.getcwd())
df = pd.read_parquet('samples_000080000_to_000080868.parquet') # the train data of random scene

sample_indx = 5  # which sample in the dataset file is used (1377 in total)
# obj_num = 0  # which robot/object is selected  (4 objects, 2 robots)
# robot_num = 0

# save the start and goal poses of objects and robots
initial_states = []
initial_states.append({'objects': df['obj_file'][sample_indx]['objects'].tolist(), 'robots': df['robot_file'][sample_indx]['robots'].tolist()})
with open(f'initial_{sample_indx}.json', 'w') as file:
    json.dump(initial_states, file, default=str, indent=4)


# save the traj of the 3 objects to two json files
for obj_num in range(3):
    obj_traj = []

    obj_step = df['trajectory'][sample_indx]['objs'][obj_num]['steps']
    obj_name = df['trajectory'][sample_indx]['objs'][obj_num]['name']


    for i in range(obj_step.size):
        obj_traj.append({'pos': obj_step[i]['pos'].tolist(), 'quat': obj_step[i]['quat'].tolist()}) # convert np.ndarray object to list

    with open(f'{obj_name}_traj_{sample_indx}.json', 'w') as file:
        json.dump(obj_traj, file, indent=4)


# save the traj of the 3 robots to four json files
for robot_num in range(3):
    joint_traj = []

    joint_state = df['trajectory'][sample_indx]['robots'][robot_num]['steps']
    rob_name = df['trajectory'][sample_indx]['robots'][robot_num]['name']

    for j in range(joint_state.size):
        joint_traj.append({'joint_state': joint_state[j]['joint_state'].tolist()}) # convert np.ndarray object to list

    with open(f'{rob_name}traj_{sample_indx}.json', 'w') as file:
        json.dump(joint_traj, file, indent=4)

    # save the end-effector traj of the 3 robot to a json file
    ee_traj = []

    ee_step = df['trajectory'][sample_indx]['robots'][robot_num]['steps']

    for i in range(ee_step.size):
        ee_traj.append({'pos': ee_step[i]['ee_pos'].tolist(), 'quat': ee_step[i]['ee_quat'].tolist()}) # convert np.ndarray object to list

    with open(f'{rob_name}eetraj_{sample_indx}.json', 'w') as file:
        json.dump(ee_traj, file, indent=4)
