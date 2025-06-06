import os
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import pandas as pd
import xml.etree.ElementTree as ET
import random
import mediapy as media


# xml_path = 'scene.xml' #xml file (assumes this is in the same folder as this file)
simend = 2 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

sample_indx = 288  #decide which sample to read and show
file_save = 0  #set to 1 to generate data files


# load the joint states from dataset
df = pd.read_parquet('samples_000000000_to_000001376.parquet')



# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def add_physics(obj_num, skip_count):
    nums = list(range(obj_num))
    # random.shuffle(nums)
    skip_nums = random.sample(nums, skip_count)
    
    return skip_nums

def mjformat(array):
    str_array = ''
    for i in range(array.size - 1):
        str_array += str(array[i]) + ' '
    
    str_array += str(array[array.size-1])
    
    return str_array

def quat2r(quat_mujoco):
    #mujocoy quat is constant,x,y,z,
    #scipy quaut is x,y,z,constant
    quat_scipy = np.array([quat_mujoco[3],quat_mujoco[0],quat_mujoco[1],quat_mujoco[2]])

    r = R.from_quat(quat_scipy)

    return r

def r2quat(r):
    # Create a Rotation object from the rotation matrix
    rotation = R.from_matrix(r)
    
    # Get the quaternion (x, y, z, w)
    quat = rotation.as_quat()  # Returns as [x, y, z, w]
    
    return quat

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    global current_step
    global max_steps
    global frame
    global physics_flag

    # Check for keyboard input
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

    if act == glfw.PRESS and key == glfw.KEY_LEFT:
        current_step -= 1
        if current_step < 0:
            current_step = 0

    if act == glfw.PRESS and key == glfw.KEY_LEFT_CONTROL:
        current_step -= 10
        if current_step < 0:
            current_step = 0  

    if act == glfw.PRESS and key == glfw.KEY_RIGHT:
        current_step += 1
        if current_step >= max_steps:
            current_step = max_steps - 1

    if act == glfw.PRESS and key == glfw.KEY_RIGHT_CONTROL:
        current_step += 10
        if current_step >= max_steps:
            current_step = max_steps - 1        
        
    if act == glfw.PRESS and key == glfw.KEY_ENTER:
        frame = not frame
    
    if act == glfw.PRESS and key == glfw.KEY_P:
        physics_flag = True

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)


# set the current scene
# check the number of objects and robots
obj_num = df['obj_file'][sample_indx]['objects'].size
robot_num = df['robot_file'][sample_indx]['robots'].size
objs_file = f'objs_{obj_num}.xml'

# check which the current scene is (shelf, husky, conveyor, random)
if ('divider' in df['scene'][sample_indx]['Obstacles'].keys()):
    scene_flag = "shelf"
    xml_path = 'scene_shelf.xml'
    rob_file = 'ur5e_shelf.xml'
    table_gap = [0, 0.05, 0.55]
elif ('base_link' in df['scene'][sample_indx]['Obstacles'].keys()):
    scene_flag = "husky"
    xml_path = 'scene_husky.xml'
    rob_file = 'ur5e_husky.xml'
    table_gap = [0, 0, 0]
elif ('table_left' in df['scene'][sample_indx]['Obstacles'].keys()):
    scene_flag = "conveyor"
    xml_path = 'scene_conveyor.xml'
    rob_file = 'ur5e_conveyor.xml'
    table_gap = [0, 0.05, 0.55]
else:
    scene_flag = "random"
    xml_path = 'scene_random.xml'
    rob_file = f'ur5e_{robot_num}.xml'
    table_gap = [0, 0, 0.6]

# set the object file in scene file according to the number of objects
tree = ET.parse(xml_path)
root = tree.getroot()
i=0
for include in root.iter('include'):
    if (i==0):
        include.set('file', rob_file)
    else:
        include.set('file', objs_file)
    i += 1
tree.write(xml_path)

#load the initail data of robots and objects
ini_obj = df['obj_file'][sample_indx]['objects'].tolist()
ini_rob = df['robot_file'][sample_indx]['robots'].tolist()
ini_scene = df['scene'][sample_indx]

tree1 = ET.parse(objs_file)
root1 = tree1.getroot()

i = 0
for geo in root1.iter('geom'):
    # print(geo.attrib['size'])
    mj_size = ini_obj[i]['shape']/2

    geo.attrib['size'] = mjformat(mj_size)
    tree1.write(objs_file)
    i += 1

i = 0
for obj in root1.iter('body'):
    # print(obj.attrib['pos'])

    mj_pos = ini_obj[i]['start_pos']
    mj_quat = ini_obj[i]['start_quat']

    obj.attrib['pos'] = mjformat(mj_pos)
    obj.attrib['quat'] = mjformat(mj_quat)
    tree1.write(objs_file)
    i += 1    

# husky scene is a little different, we use the absolute position values and we don't change the robot xml file
if (scene_flag != "husky"):
    tree2 = ET.parse(rob_file)
    root2 = tree2.getroot()

    j = 0
    for rob in root2.iter('body'):
        if rob.attrib['name'].startswith('base'):
            # print(obj.attrib['name'])
            mj_pos = ini_rob[j]['base_pos']
            mj_quat = ini_rob[j]['base_quat']
            rob.attrib['pos'] = mjformat(mj_pos)
            rob.attrib['quat'] = mjformat(mj_quat)
        
            tree2.write(rob_file)
            j += 1


# read the traj data of robots and objects
# read the traj of several objects
obj_pos = [[] for k in range(obj_num)]
obj_quat = [[] for k in range(obj_num)]
for obj_idx in range(obj_num):
    obj_step = df['trajectory'][sample_indx]['objs'][obj_idx]['steps']
    obj_name = df['trajectory'][sample_indx]['objs'][obj_idx]['name']

    for i in range(obj_step.size):
        obj_pos[int(obj_name[3])-1].append([a - b for a, b in zip(obj_step[i]['pos'].tolist(), table_gap)])  #the first obj is obj1 instead of 0
        obj_quat[int(obj_name[3])-1].append(obj_step[i]['quat'].tolist())

    if file_save:
        obj_traj = []
        for i in range(obj_step.size):
            obj_traj.append({'pos': obj_step[i]['pos'].tolist(), 'quat': obj_step[i]['quat'].tolist()}) # convert np.ndarray object to list
        with open(f'{obj_name}_traj_{sample_indx}.json', 'w') as file:
            json.dump(obj_traj, file, indent=4)

# read the traj of several robots
joint_traj = [[] for k in range(robot_num)]
ee_pos = [[] for k in range(robot_num)]
ee_quat = [[] for k in range(robot_num)]
for robot_idx in range(robot_num):
    joint_state = df['trajectory'][sample_indx]['robots'][robot_idx]['steps']
    rob_name = df['trajectory'][sample_indx]['robots'][robot_idx]['name']

    for j in range(joint_state.size):
        joint_traj[int(rob_name[1])].append(joint_state[j]['joint_state'].tolist())  #the first traj might not be the traj of the first robot

    if file_save:
        robot_traj = []
        for j in range(joint_state.size):
            robot_traj.append({'joint_state': joint_state[j]['joint_state'].tolist()}) # convert np.ndarray object to list

        with open(f'{rob_name}traj_{sample_indx}.json', 'w') as file:
            json.dump(robot_traj, file, indent=4)

    # read the end-effector traj
    ee0_step = df['trajectory'][sample_indx]['robots'][robot_idx]['steps']
    
    for i in range(ee0_step.size):
        ee_pos[int(rob_name[1])].append(ee0_step[i]['ee_pos'].tolist())
        ee_quat[int(rob_name[1])].append(ee0_step[i]['ee_quat'].tolist())

    if file_save:
        ee0_traj = []
        for i in range(ee0_step.size):
            ee0_traj.append({'pos': ee0_step[i]['ee_pos'].tolist(), 'quat': ee0_step[i]['ee_quat'].tolist()}) # convert np.ndarray object to list

        with open(f'{rob_name}eetraj_{sample_indx}.json', 'w') as file:
            json.dump(ee0_traj, file, indent=4)

length = len(joint_traj[0])  #length of the trajectory



#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname, xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options


# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])
cam.azimuth = 51.59999999999994 ; cam.elevation = -27.599999999999962 ; cam.distance =  4.321844071190101
cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

# data.qpos[0] = np.pi / 2
# data.qpos[1] = np.pi / 3
# data.qpos[2] = np.pi / 4

#initialize the controller
init_controller(model,data)

#set the controller
mj.set_mjcb_control(controller)

# enable joint visualization option:
scene_option = mj.MjvOption()
scene_option.flags[mj.mjtVisFlag.mjVIS_JOINT] = True

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

current_step = 0
max_steps = length
step_size = 1.0 / max_steps
frame = False
frames = []
physics_flag = False
skip_flag = False
skip_nums = []
skip_idx = None
with mj.Renderer(model) as renderer:
    while data.time < duration:
        # # set the qpos values of current step
        # for k in range(robot_num):
        #     data.qpos[6*k:6*k+6] = joint_traj[k][current_step][0:6]

        # for l in range(obj_num):
        #     if l in skip_nums or l == skip_idx:
        #         # print(data.qpos[6*robot_num+7*l:6*robot_num+7*l+7])
        #         continue
        #     else:
        #         data.qpos[6*robot_num+7*l:6*robot_num+7*l+7] = obj_pos[l][current_step][0:3] + obj_quat[l][current_step][0:4]
        
        mj.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)
        
        
        

        # if frame:
        #     current_step += 1

        # # Update simulation state based on current step
        # # print(data.qpos[26])
        # # data_tem = mj.MjData(model)
        # # print(data_tem.qpos[26])
        # mj.mj_step(model, data)
        # viewport_width, viewport_height = glfw.get_framebuffer_size(
        #         window)
        # viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        # # collisoin detection
        # # for contact in data.contact:
        # #     if contact.geom1 < model.ngeom and contact.geom2 < model.ngeom:
        # #         # print(f"Collision detected between {contact.geom1} and {contact.geom2}")
        # #         print(f"Collision detected with distance {contact.dist} at contact point {contact.pos}")

        # # Render the scene
        # mj.mjv_updateScene(model, data, opt, None, cam,
        #                     mj.mjtCatBit.mjCAT_ALL.value, scene)
        # mj.mjr_render(viewport, scene, context)
        # glfw.swap_buffers(window)
        # glfw.poll_events()

media.show_video(frames, fps=framerate)

glfw.terminate()
