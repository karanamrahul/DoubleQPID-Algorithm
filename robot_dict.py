import numpy as np
# custom files
from Husky.husky import husky_pi
from HectorQ.drone import drone_pi



robot_dict = {
    
    'husky_pi_random' : {'class': husky_pi, 'set_point': np.array([0.31, -0.19]), 'action_centroid': np.array([np.maximum(2.*np.random.rand(),0.31), np.maximum(2.*np.random.rand(),0.31),np.maximum(2.*np.random.rand(),0.31),np.maximum(2.*np.random.rand(),0.31)]), 'initial_state': np.array([[0., 0.]]),'K_step': 0.3, 'comentarios': 'roslaunch husky_gazebo husky_playpen.launch'},

    'husky_pi' : {'class': husky_pi, 'set_point': np.array([0.31,-0.19]), 'action_centroid': np.array([0.5, 0.51, 0.5, 0.51]), 'initial_state': np.array([[0., 0.]]),'K_step': 0.3, 'comentarios': 'roslaunch husky_gazebo husky_playpen.launch'},

    'drone_pi' : {'class': drone_pi, 'set_point': np.array([0.21,0.21]), 'action_centroid': np.array([0.5, 0.51, 0.5, 0.51]), 'initial_state': np.array([[0., 0.]]),'K_step': 0.3, 'comentarios': 'roslaunch hector_quadrotor_gazebo quadrotor_empty_world.launch '},

    'drone_2' : {'class': drone_pi, 'set_point': np.array([0.21,0.21]), 'action_centroid': np.array([0.5, 0.51, 0.5, 0.51]), 'initial_state': np.array([[0., 0.]]),'K_step': 0.3, 'comentarios': 'roslaunch hector_quadrotor_gazebo quadrotor_empty_world.launch '},
    
    }


