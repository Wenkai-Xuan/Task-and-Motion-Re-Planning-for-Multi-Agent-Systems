import os
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import mediapy as media
from mujoco import viewer
import time

# load the joint states of robots from json file
with open('a0_traj.json', 'r') as file:
    states = json.load(file)
joint_states = []
for i in range(489):   # 474 joint states in total
    joint_states.append(states[i]['joint_state']) # join_states:(474,7)

with open('a1_traj.json', 'r') as file:
    states1 = json.load(file)
joint_states1 = []
for i in range(489):
    joint_states1.append(states1[i]['joint_state'])

with open('a2_traj.json', 'r') as file:
    states2 = json.load(file)
joint_states2 = []
for i in range(489):
    joint_states2.append(states2[i]['joint_state'])


# load the object states from json file
with open('obj1_traj.json', 'r') as file:
    obj = json.load(file)
obj_pos = []
obj_quat = []
for i in range(489):   # 474 object states in total
    obj_pos.append([a - b for a, b in zip(obj[i]['pos'], [0, 0, 0.6])]) # join_states:(474,7), obj[i]['pos'] - (0, 0, 0.6) as the table's height 
    obj_quat.append(obj[i]['quat'])

with open('obj2_traj.json', 'r') as file:
    obj1 = json.load(file)
obj_pos1 = []
obj_quat1 = []
for i in range(489):
    obj_pos1.append([a - b for a, b in zip(obj1[i]['pos'], [0, 0, 0.6])])
    obj_quat1.append(obj1[i]['quat'])

with open('obj3_traj.json', 'r') as file:
    obj2 = json.load(file)
obj_pos2 = []
obj_quat2 = []
for i in range(489):   
    obj_pos2.append([a - b for a, b in zip(obj2[i]['pos'], [0, 0, 0.6])])
    obj_quat2.append(obj2[i]['quat'])



# Load the MuJoCo model from an XML file
model = mj.MjModel.from_xml_path('scene.xml')
data = mj.MjData(model)

# Create a simulation object
# sim = mujoco.Mujoco(model)

# Create a viewer window to display the simulation
# viewer = mj.viewer.launch(model,data)
with viewer.launch_passive(model, data) as viewer:
# Simulate the system for a set number of timesteps and update the viewer
    for current_step in range(489):
        data.qpos[0:6] = joint_states[current_step][0:6]
        data.qpos[6:12] = joint_states1[current_step][0:6]
        data.qpos[12:18] = joint_states2[current_step][0:6]

        data.qpos[18:21] = obj_pos[current_step][0:3]
        data.qpos[21:25] = obj_quat[current_step][0:4]
        data.qpos[25:28] = obj_pos1[current_step][0:3]
        data.qpos[28:32] = obj_quat1[current_step][0:4]
        data.qpos[32:35] = obj_pos2[current_step][0:3]
        data.qpos[35:39] = obj_quat2[current_step][0:4]

        mj.mj_step(model,data)  # Advance the simulation by one timestep
        viewer.sync()  # Render the current state in the viewer

# Close the viewer when done
viewer.close()
