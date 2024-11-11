import os
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import json
from scipy.spatial.transform import Rotation as R


xml_path = 'scene.xml' #xml file (assumes this is in the same folder as this file)
simend = 2 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

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

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0


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

# while not glfw.window_should_close(window):
#     time_prev = data.time

#     while (data.time - time_prev < 1.0/60.0):
#         mj.mj_step(model, data)

#     if (data.time>=simend):
#         break

#     # get framebuffer viewport
#     viewport_width, viewport_height = glfw.get_framebuffer_size(
#         window)
#     viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

#     #print camera configuration (help to initialize the view)
#     if (print_camera_config==1):
#         print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
#         print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

#     # Update scene and render
#     mj.mjv_updateScene(model, data, opt, None, cam,
#                        mj.mjtCatBit.mjCAT_ALL.value, scene)
#     mj.mjr_render(viewport, scene, context)

#     data.qpos[0] = np.pi / 2
#     data.qpos[1] = -np.pi / 3
#     data.qpos[2] = np.pi / 4
#     # print(data.qpos)

#     # swap OpenGL buffers (blocking call due to v-sync)
#     glfw.swap_buffers(window)

#     # process pending GUI events, call GLFW callbacks
#     glfw.poll_events()

# while not glfw.window_should_close(window):
#     time_prev = data.time

#     while (data.time - time_prev < 1.0/60.0):
#         mj.mj_step(model, data)

#     if (data.time>=simend):
#         break

#     # assign the states to data.qpos, which is composed as 
#     # [robot1.joint, robot2.joint, robot3.joint, obj1.pos, obj1.quat, obj2.pos, obj2.quat, obj3.pos, obj3.quat]
#     for i in range(389):   # 474 joint states in total
#         # data.qpos[0] = -joint_states[i][1]
#         # data.qpos[1] = -joint_states[i][4]
#         # data.qpos[2] = -joint_states[i][2] #
#         # data.qpos[3] = -joint_states[i][5] #
#         # data.qpos[4] = -joint_states[i][4] #
#         # data.qpos[5] = -joint_states[i][3] #

#         #set states for robots
#         # for j in range(6):
#         #     data.qpos[j] = joint_states[i][j]
#         # for j in range(6):
#         #     data.qpos[j+6] = joint_states1[i][j]
#         # for j in range(6):
#         #     data.qpos[j+12] = joint_states2[i][j]
        
#         data.qpos[0:6] = joint_states[i][0:6]
#         data.qpos[6:12] = joint_states1[i][0:6]
#         data.qpos[12:18] = joint_states2[i][0:6]

#         #set states for objects
#         # for j in range(3):
#         #     data.qpos[j+18] = obj_pos[i][j] 
#         # for j in range(4):
#         #     data.qpos[j+21] = obj_quat[i][j]
        
#         # for j in range(3):
#         #     data.qpos[j+25] = obj_pos1[i][j]
#         # for j in range(4):
#         #     data.qpos[j+28] = obj_quat1[i][j]
        
#         # for j in range(3):
#         #     data.qpos[j+32] = obj_pos2[i][j]
#         # for j in range(4):
#         #     data.qpos[j+35] = obj_quat2[i][j]

#         data.qpos[18:21] = obj_pos[i][0:3]
#         data.qpos[21:25] = obj_quat[i][0:4]

#         data.qpos[25:28] = obj_pos1[i][0:3]
#         data.qpos[28:32] = obj_quat1[i][0:4]

#         data.qpos[32:35] = obj_pos2[i][0:3]
#         data.qpos[35:39] = obj_quat2[i][0:4]

#         # #convert quaternion
#         # #mujocoy quat and trajectory quat are both constant,x,y,z,
#         # #scipy quaut is x,y,z,constant
#         # quat = np.array([obj_quat[i][0], obj_quat[i][1], obj_quat[i][2], obj_quat[i][3]])
#         # r = quat2r(quat)
#         # C = np.array([[0,0,1],[1,0,0],[0,1,0]]) #transformation from opneGL to MuJoCo
#         # r_mu = np.matmul(r.as_matrix(),C)
#         # quat_mu = r2quat(r_mu)  #scipy quaut is x,y,z,constant
#         # data.qpos[9] = quat_mu[3] 
#         # data.qpos[10] = quat_mu[0]
#         # data.qpos[11] = quat_mu[1]
#         # data.qpos[12] = quat_mu[2]


#         mj.mj_step(model, data)
        
#         # get framebuffer viewport
#         viewport_width, viewport_height = glfw.get_framebuffer_size(
#             window)
#         viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

#         #print camera configuration (help to initialize the view)
#         if (print_camera_config==1):
#             print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
#             print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

#         # Update scene and render
#         mj.mjv_updateScene(model, data, opt, None, cam,
#                         mj.mjtCatBit.mjCAT_ALL.value, scene)
#         mj.mjr_render(viewport, scene, context)

#         # print(data.qpos)

#         # swap OpenGL buffers (blocking call due to v-sync)
#         glfw.swap_buffers(window)

#         # process pending GUI events, call GLFW callbacks
#         glfw.poll_events()

current_step = 0
max_steps = 489
step_size = 1.0 / max_steps
frames = False
while not glfw.window_should_close(window):
    time_prev = data.time
    while (data.time - time_prev < step_size):
        mj.mj_step(model, data)
    
    if current_step >= max_steps:
        break

    # set the qpos values of current step
    data.qpos[0:6] = joint_states[current_step][0:6]
    data.qpos[6:12] = joint_states1[current_step][0:6]
    data.qpos[12:18] = joint_states2[current_step][0:6]

    data.qpos[18:21] = obj_pos[current_step][0:3]
    data.qpos[21:25] = obj_quat[current_step][0:4]
    data.qpos[25:28] = obj_pos1[current_step][0:3]
    data.qpos[28:32] = obj_quat1[current_step][0:4]
    data.qpos[32:35] = obj_pos2[current_step][0:3]
    data.qpos[35:39] = obj_quat2[current_step][0:4]

    if frames:
        current_step += 1

    # Update simulation state based on current step
    mj.mj_step(model, data)
    viewport_width, viewport_height = glfw.get_framebuffer_size(
            window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)


    # Render the scene
    mj.mjv_updateScene(model, data, opt, None, cam,
                        mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()


glfw.terminate()
