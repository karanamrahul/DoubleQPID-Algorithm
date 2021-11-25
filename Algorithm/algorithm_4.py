import numpy as np 
from Algorithm.functions import identify_nearest_centroid_for_multiple_tables
from Algorithm.algorithm_2 import algorithm_2
from Algorithm.double_Q import DQPID
import time


def algorithm_4(Q_arrange,Mt,next_state,reward,maximum_depth,number_of_actions,e_greed_counter,set_point, flag_ab, K_step):
    start = time.time()
    state_index = Mt[0].astype(int)
    action_index = Mt[1].astype(int)
    Q_index = Mt[2].astype(int)
    action = Mt[3:,]
    next_Q_index = Q_index
    next_state_index, min_distance_to_centroid = Q_arrange[Q_index].identify_nearest_centroid(next_state)
    delta = Q_arrange[Q_index].max_state
    l_depth = Q_arrange[Q_index].depth
    

    # Create new table with higher depth, because next state is inside 
    if (np.abs(min_distance_to_centroid) <= np.abs(delta)) and (l_depth<maximum_depth):
        print('new object in algorithm 4') 
        amount_of_objects = len(Q_arrange)

        min_state = Q_arrange[Q_index].centroids[state_index] + Q_arrange[Q_index].min_state
        max_state = Q_arrange[Q_index].centroids[state_index] + Q_arrange[Q_index].max_state
        Q_cheat = (Q_arrange[Q_index].Q_A[state_index][action_index] + Q_arrange[Q_index].Q_B[state_index][action_index])/2.
        current_depth_v = Q_arrange[Q_index].depth
        new_centroid = np.array([Q_arrange[Q_index].centroids[state_index]]) # this is to give it an array format 
        
        Q_arrange.append(DQPID(new_centroid,Q_index, current_depth_v + 1, action, number_of_actions, maximum_depth, action_index, Q_cheat, K_step ))
        temp_Q_index = amount_of_objects
        new_state_index, new_min_distance_to_centroid = Q_arrange[temp_Q_index].identify_nearest_centroid(next_state)
        delta_new = Q_arrange[temp_Q_index].max_state
        if (np.abs(new_min_distance_to_centroid) >= delta_new):
            
            new_state_index = Q_arrange[temp_Q_index].get_new_centroid(next_state)
        e_greed_counter = 0.
        #I increase the descendence of the parent
        Q_arrange[Q_index].descendence = Q_arrange[Q_index].descendence + 1
        # I store the index of the son in the parent
        Q_arrange[Q_index].descendence_index = np.append(Q_arrange[Q_index].descendence_index , amount_of_objects)
        
        # I set the next table 
        next_Q_index = temp_Q_index
        #next_state_index, min_distance_to_centroid = Q_arrange[Q_index].identify_nearest_centroid(next_state)

        print('state_index',state_index, action_index)
        if flag_ab=='A':
            Q_B_max_next_value = np.max(Q_arrange[next_Q_index].Q_B[new_state_index])
            Q_arrange[Q_index].update_Q(state_index,action_index,reward,Q_B_max_next_value,flag_ab)
        else:  
            Q_A_max_next_value = np.max(Q_arrange[next_Q_index].Q_A[new_state_index])
            Q_arrange[Q_index].update_Q(state_index,action_index,reward,Q_A_max_next_value,flag_ab)
        

    # maximum depth
    elif(l_depth==maximum_depth and np.abs(min_distance_to_centroid)<=np.abs(delta) ):
        Q_arrange[Q_index] = algorithm_2(Q_arrange[Q_index], state_index, next_state, reward, action_index, flag_ab)
        next_Q_index = Q_index

    # goes back one discretization level because next state is outside    
    elif(np.abs(min_distance_to_centroid)>= np.abs(delta)):
        
        stop_loop = False
        while stop_loop == False:
            l_depth = l_depth - 1
            
            # I search in lower depths to see if there is a centroid for the state
            if l_depth > 0:    
                temp_min_distance_to_centroid, temp_state_index, temp_Q_index = identify_nearest_centroid_for_multiple_tables(Q_arrange,l_depth,next_state)
            
            else: 
                l_depth = 1 # bug catcher, means deapth cant be smaller than 1 
                temp_min_distance_to_centroid, temp_state_index, temp_Q_index = identify_nearest_centroid_for_multiple_tables(Q_arrange,l_depth,next_state)

            # if the distance the the centroid is less than the minimum I create a new centroid in higher depth
            if( np.abs(temp_min_distance_to_centroid)<Q_arrange[temp_Q_index].max_state and l_depth>1):
                
                l_depth = l_depth +1
                min_distance_to_centroid, state_index, Q_index = identify_nearest_centroid_for_multiple_tables(Q_arrange,l_depth,next_state)
                Q_arrange[Q_index] = algorithm_2(Q_arrange[Q_index], state_index, next_state, reward,action_index, flag_ab)
                next_Q_index = Q_index
                stop_loop = True
                

            if l_depth == 1:
                l_new = 1
                Q_index = 0
                h_new = Q_arrange[Q_index].h
                Q_arrange[Q_index] = algorithm_2(Q_arrange[Q_index], state_index,next_state,reward,action_index, flag_ab)
                next_Q_index = Q_index
                stop_loop = True
            
    
    else:
        print('*********************************')
        print('************ERROR****************')
        print('*********************************')
        next_Q_index = -1


    end = time.time()
    print('the execution time is',end-start)
    return Q_arrange, Q_index, next_Q_index, e_greed_counter