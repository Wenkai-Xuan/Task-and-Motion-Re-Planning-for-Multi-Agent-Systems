import os
import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import mediapy as media
from mujoco import viewer
import time

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
# button_left = False
# button_middle = False
# button_right = False
# lastx = 0
# lasty = 0


# def quat2r(quat_mujoco):
#     #mujocoy quat is constant,x,y,z,
#     #scipy quaut is x,y,z,constant
#     quat_scipy = np.array([quat_mujoco[3],quat_mujoco[0],quat_mujoco[1],quat_mujoco[2]])

#     r = R.from_quat(quat_scipy)

#     return r

# def r2quat(r):
#     # Create a Rotation object from the rotation matrix
#     rotation = R.from_matrix(r)
    
#     # Get the quaternion (x, y, z, w)
#     quat = rotation.as_quat()  # Returns as [x, y, z, w]
    
#     return quat

# def init_controller(model,data):
#     #initialize the controller here. This function is called once, in the beginning
#     pass

# def controller(model, data):
#     #put the controller here. This function is called inside the simulation.
#     pass

# def keyboard(window, key, scancode, act, mods):
#     global current_step
#     global max_steps
#     global frames

#     # Check for keyboard input
#     if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
#         mj.mj_resetData(model, data)
#         mj.mj_forward(model, data)

#     if act == glfw.PRESS and key == glfw.KEY_LEFT:
#         current_step -= 1
#         if current_step < 0:
#             current_step = 0

#     if act == glfw.PRESS and key == glfw.KEY_LEFT_CONTROL:
#         current_step -= 10
#         if current_step < 0:
#             current_step = 0  

#     if act == glfw.PRESS and key == glfw.KEY_RIGHT:
#         current_step += 1
#         if current_step >= max_steps:
#             current_step = max_steps - 1

#     if act == glfw.PRESS and key == glfw.KEY_RIGHT_CONTROL:
#         current_step += 10
#         if current_step >= max_steps:
#             current_step = max_steps - 1        
                
#     if act == glfw.PRESS and key == glfw.KEY_ENTER:
#         frames = not frames

# def mouse_button(window, button, act, mods):
#     # update button state
#     global button_left
#     global button_middle
#     global button_right

#     button_left = (glfw.get_mouse_button(
#         window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
#     button_middle = (glfw.get_mouse_button(
#         window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
#     button_right = (glfw.get_mouse_button(
#         window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

#     # update mouse position
#     glfw.get_cursor_pos(window)

# def mouse_move(window, xpos, ypos):
#     # compute mouse displacement, save
#     global lastx
#     global lasty
#     global button_left
#     global button_middle
#     global button_right

#     dx = xpos - lastx
#     dy = ypos - lasty
#     lastx = xpos
#     lasty = ypos

#     # no buttons down: nothing to do
#     if (not button_left) and (not button_middle) and (not button_right):
#         return

#     # get current window size
#     width, height = glfw.get_window_size(window)

#     # get shift key state
#     PRESS_LEFT_SHIFT = glfw.get_key(
#         window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
#     PRESS_RIGHT_SHIFT = glfw.get_key(
#         window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
#     mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

#     # determine action based on mouse button
#     if button_right:
#         if mod_shift:
#             action = mj.mjtMouse.mjMOUSE_MOVE_H
#         else:
#             action = mj.mjtMouse.mjMOUSE_MOVE_V
#     elif button_left:
#         if mod_shift:
#             action = mj.mjtMouse.mjMOUSE_ROTATE_H
#         else:
#             action = mj.mjtMouse.mjMOUSE_ROTATE_V
#     else:
#         action = mj.mjtMouse.mjMOUSE_ZOOM

#     mj.mjv_moveCamera(model, action, dx/height,
#                       dy/height, scene, cam)

# def scroll(window, xoffset, yoffset):
#     action = mj.mjtMouse.mjMOUSE_ZOOM
#     mj.mjv_moveCamera(model, action, 0.0, -0.05 *
#                       yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# # Init GLFW, create window, make OpenGL context current, request v-sync
# glfw.init()
# window = glfw.create_window(1200, 900, "Demo", None, None)
# glfw.make_context_current(window)
# glfw.swap_interval(1)

# # initialize visualization data structures
# mj.mjv_defaultCamera(cam)
# mj.mjv_defaultOption(opt)
# scene = mj.MjvScene(model, maxgeom=10000)
# context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# # install GLFW mouse and keyboard callbacks
# glfw.set_key_callback(window, keyboard)
# glfw.set_cursor_pos_callback(window, mouse_move)
# glfw.set_mouse_button_callback(window, mouse_button)
# glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])
# cam.azimuth = 51.59999999999994 ; cam.elevation = -27.599999999999962 ; cam.distance =  4.321844071190101
# cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

# # data.qpos[0] = np.pi / 2
# # data.qpos[1] = np.pi / 3
# # data.qpos[2] = np.pi / 4

# #initialize the controller
# init_controller(model,data)

# #set the controller
# mj.set_mjcb_control(controller)



# current_step = 0
# max_steps = 489
# step_size = 1.0 / max_steps
# frames = False
# while not glfw.window_should_close(window):
#     time_prev = data.time
#     while (data.time - time_prev < step_size):
#         mj.mj_step(model, data)
    
#     if current_step >= max_steps:
#         break

#     # set the qpos values of current step
#     data.qpos[0:6] = joint_states[current_step][0:6]
#     data.qpos[6:12] = joint_states1[current_step][0:6]
#     data.qpos[12:18] = joint_states2[current_step][0:6]

#     data.qpos[18:21] = obj_pos[current_step][0:3]
#     data.qpos[21:25] = obj_quat[current_step][0:4]
#     data.qpos[25:28] = obj_pos1[current_step][0:3]
#     data.qpos[28:32] = obj_quat1[current_step][0:4]
#     data.qpos[32:35] = obj_pos2[current_step][0:3]
#     data.qpos[35:39] = obj_quat2[current_step][0:4]

#     if frames:
#         current_step += 1

#     # Update simulation state based on current step
#     mj.mj_step(model, data)
#     viewport_width, viewport_height = glfw.get_framebuffer_size(
#             window)
#     viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)


#     # Render the scene
#     mj.mjv_updateScene(model, data, opt, None, cam,
#                         mj.mjtCatBit.mjCAT_ALL.value, scene)
#     mj.mjr_render(viewport, scene, context)
#     glfw.swap_buffers(window)
#     glfw.poll_events()


# glfw.terminate()


def key_callback(keycode):
    if chr(keycode) == ' ':
        global paused
        paused = not paused

    if chr(keycode) == 'p':
        global paused
        paused = not paused
        global drop
        drop = not drop
    

model.opt.timestep = 0.02
paused = False
drop = False
current_step = 0
with viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()

    data.qpos[0:6] = joint_states[current_step][0:6]
    data.qpos[6:12] = joint_states1[current_step][0:6]
    data.qpos[12:18] = joint_states2[current_step][0:6]

    data.qpos[18:21] = obj_pos[current_step][0:3]
    data.qpos[21:25] = obj_quat[current_step][0:4]
    data.qpos[25:28] = obj_pos1[current_step][0:3]
    data.qpos[28:32] = obj_quat1[current_step][0:4]
    data.qpos[32:35] = obj_pos2[current_step][0:3]
    data.qpos[35:39] = obj_quat2[current_step][0:4]

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    if not paused:
        mj.mj_step(model, data)
        if (current_step < 488):
            current_step += 1

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
