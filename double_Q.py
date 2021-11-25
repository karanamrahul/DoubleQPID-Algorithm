import numpy as np
from Algorithm.generate_k_table_pioneer_6 import generate_k_table_pioneer_6, generate_k_table_pioneer_4, generate_k_table_pioneer_3, generate_k_table_pioneer_2
import time

np.random.seed(1234)

class DQPID():
         
    
    
    # the constructor takes 0.006s. it is quite fast. 
    def __init__ (self, centroids, ascendence, depth, k_centroids, k_size, maximum_depth, action_index = 0., Q_cheat = 0. , K_step = 0.3 ):

        start = time.time()
        # constants
        self.K_STEP_DEFAULT = K_step
        self.DELTA_STATE = 0.1
        self.ALPHA = 0.2
        self.GAMMA = 0.95
        # variables 
        self.descendence = 0 # descendence is added later
        self.descendence_index = []
        self.ascendence = ascendence
        self.states_size = 1.
        self.centroids = centroids
        self.number_of_centroids = self.centroids.shape[0] # it is the first dimesion that i need
        self.depth = depth   
        self.k_centroids = k_centroids  
        self.k_centroids_original = k_centroids
        self.control_variables = len(self.k_centroids) 
        self.maximum_depth = maximum_depth
        self.action_index = action_index

        self.k_max = np.zeros(self.control_variables)  
        self.k_min = np.zeros(self.control_variables)  
        self.k_step = np.zeros(self.control_variables)  
        self.k_size = k_size  
  
        self.number_of_actions = np.power(self.k_size, self.control_variables).astype(int)
        if self.depth == 1: 
            #this is the first object
            self.h = np.array([self.centroids, 0.])
            self.max_state = +self.DELTA_STATE
            self.min_state = -self.DELTA_STATE
            for _ in range(self.control_variables):
                self.k_step[_] = self.K_STEP_DEFAULT
                self.k_max[_] = self.k_centroids[_] + self.K_STEP_DEFAULT*((self.k_size-1.)/2.)
                self.k_min[_] = self.k_centroids[_] - self.K_STEP_DEFAULT*((self.k_size-1.)/2.)

        else:
            # this is not the first object 
            self.h = np.array([self.centroids, self.action_index])
            radio_min = 0.005
            radio_max = 0.1
            b = radio_max
            a = (radio_min- radio_max)/self.maximum_depth
            y_radio = (a*self.depth) + b 
            self.max_state = y_radio
            self.min_state =-y_radio
            # create the spacing of the action spaces 
            for _ in range(self.control_variables):
                # a table of higher depth, the actions have to be calculated for each depth 
                correction_factor = 0.75
                if (self.k_centroids[_] != 0.):
                    self.k_step[_] = self.K_STEP_DEFAULT/(correction_factor*np.power(self.depth,2.))
                    self.k_max[_] = self.k_centroids[_] + self.k_step[_]*((self.k_size-1.)/2.)
                    self.k_min[_] = self.k_centroids[_] - self.k_step[_]*((self.k_size-1.)/2.)
                    # if one of the actions is less than zero, I make the actions smallers
                    while self.k_min[_] < 0.: 
                        self.k_step[_] = 0.9*self.k_step[_]
                        self.k_max[_] = self.k_centroids[_] + self.k_step[_]*((self.k_size-1.)/2.)
                        self.k_min[_] = self.k_centroids[_] - self.k_step[_]*((self.k_size-1.)/2.)
                       
                else:
                      
                        # I calculate the actions normally
                        self.k_step[_] = self.K_STEP_DEFAULT/(correction_factor*np.power(self.depth,2.))
                        self.k_min[_] = 0. # min value is zero of course
                        self.k_max[_] = self.k_centroids[_] + self.k_step[_]*((self.k_size-1.)/2.)
                        # And then I recalculate the step
                        self.k_step[_] = self.k_max[_] - (self.k_min[_]/(self.k_size-1.))
                        self.k_max[_] = 0. + self.k_step[_]*((self.k_size-1.))

        # create k_table
        if self.control_variables == 6:
            self.k_table = generate_k_table_pioneer_6(self.number_of_actions, self.k_step, self.k_min, self.k_max, self.k_size )
        if self.control_variables == 4:
            self.k_table = generate_k_table_pioneer_4(self.number_of_actions, self.k_step, self.k_min, self.k_max, self.k_size )
        if self.control_variables == 3:
            self.k_table = generate_k_table_pioneer_3(self.number_of_actions, self.k_step, self.k_min, self.k_max, self.k_size )
        if self.control_variables == 2:
            self.k_table = generate_k_table_pioneer_2(self.number_of_actions, self.k_step, self.k_min, self.k_max, self.k_size )
        
        # create Q_table
        # optimized method
        self.Q_A = -0.5 + np.multiply(-.5, np.random.rand(self.number_of_centroids, self.number_of_actions)) #before 0.1
        self.Q_B = -0.5 + np.multiply(-.5, np.random.rand(self.number_of_centroids, self.number_of_actions))
        end3 = time.time()  

       
    def identify_nearest_centroid(self, state):


        
        distance_to_centroid = np.zeros(self.number_of_centroids)
        for _ in range(self.number_of_centroids):
            distance_to_centroid[_]=  np.linalg.norm(np.subtract(state, self.centroids[_])  )

        index_of_near_centroid = np.argmin(distance_to_centroid)
        min_distance_to_centroid = distance_to_centroid[index_of_near_centroid]
        

        return index_of_near_centroid, min_distance_to_centroid
        
    
    
    def get_new_centroid(self,new_centroid):
        self.number_of_centroids = self.number_of_centroids + 1 
        print('self.centroids',self.centroids,'new_centroid',new_centroid)
        self.centroids = np.vstack((self.centroids, new_centroid))
        print(self.centroids)
        new_Q_A_row = np.zeros(self.number_of_actions)
        new_Q_B_row = np.zeros(self.number_of_actions)
        for _ in range(self.number_of_actions):
            new_Q_A_row[_] = -0.5*np.random.rand(1) - 0.5  
            new_Q_B_row[_] = -0.5*np.random.rand(1) - 0.5  

        self.Q_A = np.vstack((self.Q_A, new_Q_A_row))   
        self.Q_B = np.vstack((self.Q_B, new_Q_B_row))
        # returns the index of the new centroid
        new_centroid_index = self.number_of_centroids-1
        return new_centroid_index
    
    def update_Q(self, centroid_index, action_index, reward, Q_max_next_value,flag_ab):
        if flag_ab == 'A':
            self.Q_A[centroid_index][action_index] = self.Q_A[centroid_index][action_index] + self.ALPHA*(reward + self.GAMMA*Q_max_next_value -  self.Q_A[centroid_index][action_index]) 
        else:
            self.Q_B[centroid_index][action_index] = self.Q_B[centroid_index][action_index] + self.ALPHA*(reward + self.GAMMA*Q_max_next_value -  self.Q_B[centroid_index][action_index]) 


    def get_action_index(self, action):
        for _ in range(self.number_of_actions):
            value = np.allclose(self.k_table[_], action)
            if value == True:
                action_index = _
        return action_index


