import numpy as np 
from Algorithm.functions import identify_nearest_centroid_for_multiple_tables
from Algorithm.algorithm_2 import algorithm_2

def algorithm_3(Q_arrange,Mt,next_state,reward,flag_ab):

    # get information from memory
    state_index = Mt[0].astype(int)
    action_index = Mt[1].astype(int)
    Q_index = Mt[2].astype(int)
    action = Mt[3:,]

    # we are searching for close centroids
    next_Q_index = Q_index
    next_state_index, min_distance_to_centroid = Q_arrange[next_Q_index].identify_nearest_centroid(next_state)
    delta = Q_arrange[next_Q_index].max_state
    
   # if the current state is inside the centroid
    if (np.abs(min_distance_to_centroid) <= np.abs(delta)): 
        l_depth = Q_arrange[next_Q_index].depth
        h_new = Q_arrange[next_Q_index].h

        # We get the maximum depth achieved so far
        L_depth_vector = np.zeros(len(Q_arrange))
        for _ in range(len(Q_arrange)):
            L_depth_vector[_] = Q_arrange[_].depth
        L_max = np.max(L_depth_vector)
        
        if (l_depth < L_max) and len(Q_arrange)>1:
            
            # if we have several objects and we are not currently on the highest achieved depth, we will go and look for an object
            # for a higher depth
            stop_flag = False

            while stop_flag == False and l_depth<L_max:
                # increase depth and check the distance 
                l_depth = l_depth + 1
                temp_min_distance_to_centroid, temp_state_index, temp_Q_index = identify_nearest_centroid_for_multiple_tables(Q_arrange,l_depth,next_state)
                
                if (np.abs(temp_min_distance_to_centroid) <= np.abs(Q_arrange[temp_Q_index].max_state)):
                    # If the present state is in the vicinity of centroid's higher depth, we are temporarlily saving it.
                    # we are looping in for higher depths until we reach the promising states
                    next_state_index = temp_state_index
                    next_Q_index = temp_Q_index
                    h_new = Q_arrange[temp_Q_index].h
                    stop_flag = False
                else: 
                    # We go back to the previous time step if the current state is not in the vicinity to our previous state.
                    l_depth = l_depth - 1 
                    h_new = h_new
                    next_Q_index = next_Q_index
                    next_state_index = next_state_index
                    stop_flag = True


        else: 
            # Updating the current state
            next_Q_index = next_Q_index
            next_state_index = next_state_index

        # update Q     
        if flag_ab=='A':
            Q_B_max_next_value = np.max(Q_arrange[next_Q_index].Q_B[next_state_index])
            Q_arrange[Q_index].update_Q(state_index,action_index,reward,Q_B_max_next_value,flag_ab)
        else:  
            Q_A_max_next_value = np.max(Q_arrange[next_Q_index].Q_A[next_state_index])
            Q_arrange[Q_index].update_Q(state_index,action_index,reward,Q_A_max_next_value,flag_ab)

        
    else:   
        # if the current state is outside the centroid  
        # I will have to go backwards until I find one
        #If the current state is outside  the neighbouring state space,then we need to search by going to previous state. 
        l_depth = Q_arrange[Q_index].depth
        h_new = Q_arrange[Q_index].h
        stop_flag = False
        while stop_flag == False: 
            l_depth = l_depth -1 
            # Search for centroids in the lower depths
            if l_depth > 0:    
                temp_min_distance_to_centroid, temp_state_index, temp_Q_index = identify_nearest_centroid_for_multiple_tables(Q_arrange,l_depth,next_state)
            
            else: 
                l_depth = 1 # The depth shouldn't or cannot be less than 1.
                temp_min_distance_to_centroid, temp_state_index, temp_Q_index = identify_nearest_centroid_for_multiple_tables(Q_arrange,l_depth,next_state)


            if (np.abs(temp_min_distance_to_centroid) <= Q_arrange[temp_Q_index].max_state) and (l_depth > 1):
                # I consider this state as my next state when it is near to my previous state.
                stop_flag = True
                next_state_index = temp_state_index
                next_Q_index = temp_Q_index
                # update Q value functions based upon the succesor state   
                if flag_ab=='A':
                    Q_B_max_next_value = np.max(Q_arrange[next_Q_index].Q_B[next_state_index])
                    Q_arrange[Q_index].update_Q(state_index,action_index,reward,Q_B_max_next_value,flag_ab)
                else:  
                    Q_A_max_next_value = np.max(Q_arrange[next_Q_index].Q_A[next_state_index])
                    Q_arrange[Q_index].update_Q(state_index,action_index,reward,Q_A_max_next_value,flag_ab)



            elif l_depth == 1: 
                # if the depth is the lowest one, I have no option but to use algorithm 2 to see if there is a centroid for that state and to update it 
                l_new = 1
                Q_index = 0
                next_Q_index = 0
                h_new = Q_arrange[Q_index].h
                Q_arrange[Q_index] = algorithm_2(Q_arrange[Q_index], state_index,next_state,reward,action_index, flag_ab)
                next_Q_index = Q_index
                stop_flag = True


    return Q_arrange, Q_index, next_Q_index