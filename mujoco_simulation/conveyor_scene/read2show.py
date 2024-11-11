import os
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import pandas as pd
import xml.etree.ElementTree as ET


xml_path = 'scene.xml' #xml file (assumes this is in the same folder as this file)
simend = 2 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

sample_indx = 9  #decide which sample to read and show
file_save = 0  #set to 1 to generate data files


# load the joint states from dataset
df = pd.read_parquet('samples_000010000_to_000012113.parquet')


# read the traj of the 2 objects
obj_pos = [[],[]]
obj_quat = [[],[]]
for obj_num in range(2):
    obj_step = df['trajectory'][sample_indx]['objs'][obj_num]['steps']
    obj_name = df['trajectory'][sample_indx]['objs'][obj_num]['name']

    for i in range(obj_step.size):
        obj_pos[int(obj_name[3])-1].append([a - b for a, b in zip(obj_step[i]['pos'].tolist(), [0, 0, 0.45])])  #the first obj is obj1 instead of 0
        obj_quat[int(obj_name[3])-1].append(obj_step[i]['quat'].tolist())

    if file_save:
        obj_traj = []
        for i in range(obj_step.size):
            obj_traj.append({'pos': obj_step[i]['pos'].tolist(), 'quat': obj_step[i]['quat'].tolist()}) # convert np.ndarray object to list
        with open(f'{obj_name}_traj_{sample_indx}.json', 'w') as file:
            json.dump(obj_traj, file, indent=4)

# read the traj of the 4 robots
joint_traj = [[],[],[],[]]
ee_pos = [[],[],[],[]]
ee_quat = [[],[],[],[]]
for robot_num in range(4):
    joint_state = df['trajectory'][sample_indx]['robots'][robot_num]['steps']
    rob_name = df['trajectory'][sample_indx]['robots'][robot_num]['name']

    for j in range(joint_state.size):
        joint_traj[int(rob_name[1])].append(joint_state[j]['joint_state'].tolist())  #the first traj might not be the traj of the first robot

    if file_save:
        robot_traj = []
        for j in range(joint_state.size):
            robot_traj.append({'joint_state': joint_state[j]['joint_state'].tolist()}) # convert np.ndarray object to list

        with open(f'{rob_name}traj_{sample_indx}.json', 'w') as file:
            json.dump(robot_traj, file, indent=4)

    # read the end-effector traj
    ee0_step = df['trajectory'][sample_indx]['robots'][robot_num]['steps']
    
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


# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

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
    global frames

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
        frames = not frames

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

#load the initail data of robots and objects
ini_obj = df['obj_file'][sample_indx]['objects'].tolist()
ini_rob = df['robot_file'][sample_indx]['robots'].tolist()

tree1 = ET.parse('objs_two.xml')
root1 = tree1.getroot()

i = 0
for geo in root1.iter('geom'):
    # print(geo.attrib['size'])
    mj_size = ini_obj[i]['shape']/2

    geo.attrib['size'] = mjformat(mj_size)
    tree1.write('objs_two.xml')
    i += 1

i = 0
for obj in root1.iter('body'):
    # print(obj.attrib['pos'])

    mj_pos = ini_obj[i]['start_pos']
    mj_quat = ini_obj[i]['start_quat']

    obj.attrib['pos'] = mjformat(mj_pos)
    obj.attrib['quat'] = mjformat(mj_quat)
    tree1.write('objs_two.xml')
    i += 1    

tree2 = ET.parse('ur5e_four.xml')
root2 = tree2.getroot()

j = 0
for rob in root2.iter('body'):
    if rob.attrib['name'].startswith('base'):
        # print(obj.attrib['name'])
        mj_pos = ini_rob[j]['base_pos']
        mj_quat = ini_rob[j]['base_quat']
        rob.attrib['pos'] = mjformat(mj_pos)
        rob.attrib['quat'] = mjformat(mj_quat)
    
        tree2.write('ur5e_four.xml')
        j += 1

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
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


current_step = 0
max_steps = length
step_size = 1.0 / max_steps
frames = False
while not glfw.window_should_close(window):
    time_prev = data.time
    while (data.time - time_prev < step_size):
        mj.mj_step(model, data)
    
    if current_step >= max_steps:
        break

    # set the qpos values of current step
    data.qpos[0:6] = joint_traj[0][current_step][0:6]
    data.qpos[6:12] = joint_traj[1][current_step][0:6]
    data.qpos[12:18] = joint_traj[2][current_step][0:6]
    data.qpos[18:24] = joint_traj[3][current_step][0:6]

    data.qpos[24:27] = obj_pos[0][current_step][0:3]
    data.qpos[27:31] = obj_quat[0][current_step][0:4]
    data.qpos[31:34] = obj_pos[1][current_step][0:3]
    data.qpos[34:38] = obj_quat[1][current_step][0:4]

    if frames:
        current_step += 1

    # Update simulation state based on current step
    mj.mj_step(model, data)
    viewport_width, viewport_height = glfw.get_framebuffer_size(
            window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # collisoin detection
    for contact in data.contact:
        if contact.geom1 < model.ngeom and contact.geom2 < model.ngeom:
            # print(f"Collision detected between {contact.geom1} and {contact.geom2}")
            print(f"Collision detected with distance {contact.dist} at contact point {contact.pos}")

    # Render the scene
    mj.mjv_updateScene(model, data, opt, None, cam,
                        mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()


glfw.terminate()
