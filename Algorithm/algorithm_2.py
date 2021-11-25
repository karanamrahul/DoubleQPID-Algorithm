# algorithm 2

import numpy as np


def algorithm_2 (obj, state_index, next_state, reward, action_index, flag_ab):


    centroid_num, distance = obj.identify_nearest_centroid(next_state)
    # Get s' closer to actual state 
    probably_next_state_index, distance = obj.identify_nearest_centroid(next_state)

    if (np.abs(distance) < obj.max_state): 

        # if it is inside the vicinity s' is the state
        next_state_index = probably_next_state_index
       
    else:
        # if it is not in the vicinity I add a new state
        print('Algorithm 2 adds a new centroid at the level l= , h=') 
        next_state_index = obj.get_new_centroid(next_state)

    #print(centroid_actual)
    if flag_ab=='A':
            Q_B_max_next_value = np.max(obj.Q_B[next_state_index])
            obj.update_Q(state_index,action_index,reward,Q_B_max_next_value,flag_ab)
    else:  
            Q_A_max_next_value = np.max(obj.Q_A[next_state_index])
            obj.update_Q(state_index,action_index,reward,Q_A_max_next_value,flag_ab)
    


    return obj


