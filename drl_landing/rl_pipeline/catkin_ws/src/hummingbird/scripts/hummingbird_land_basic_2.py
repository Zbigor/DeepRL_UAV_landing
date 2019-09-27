# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
#!/usr/bin/env python

import rospy
import numpy as np
from gym import spaces
from openai_ros.robot_envs import hummingbird_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion
import random
from hummingbird.msg import Rewards
# registering the environment with the openai
#timestep_limit_per_episode = 200 # Can be any Value
#register(
#        id='HummingbirdLandBasic-v0',
#        entry_point='openai_ros:task_envs.hummingbird.hummingbird_land_basic.HummingbirdLandBasic',
#        timestep_limit=timestep_limit_per_episode
#    )


#if __name__ == '__main__':
#    try:
#        timestep_limit_per_episode = 200 # Can be any Value
#        register(
#                id='HummingbirdLandBasic-v0',
#                entry_point='openai_ros:task_envs.hummingbird.hummingbird_land_basic.HummingbirdLandBasic',
#                timestep_limit=timestep_limit_per_episode
#            )
#        print("DONE")
#       
#    except rospy.ROSInterruptException:
#        pass




class HummingbirdLandBasic(hummingbird_env.HummingbirdEnv):
    def __init__(self,env_id):
        """
        Make the drone learn how to land safely on a specified location in an empty world
        """
        # Here we will add any init functions prior to starting the MyRobotEnv
        super(HummingbirdLandBasic, self).__init__(env_id = env_id)
 
        self.episode_num = 0

        # reading parameters from the yaml file

        
        # Get WorkSpace Cube Dimensions
        self.work_space_x_max = rospy.get_param("/drone/work_space/x_max")
        self.work_space_x_min = rospy.get_param("/drone/work_space/x_min")
        self.work_space_y_max = rospy.get_param("/drone/work_space/y_max")
        self.work_space_y_min = rospy.get_param("/drone/work_space/y_min")
        self.work_space_z_max = rospy.get_param("/drone/work_space/z_max")
        self.work_space_z_min = rospy.get_param("/drone/work_space/z_min")
        
        # Maximum roll and pitch values for the drone to be considered flipped
        # currently +/- 90 degrees
        # all angles are given in radians
        self.max_roll = rospy.get_param("/drone/max_roll")
        self.max_pitch = rospy.get_param("/drone/max_pitch")
        
        # maximum yaw rate
        self.max_yaw_rate = rospy.get_param("/drone/max_yaw_rate")
        
        # maximum thrust values
        # neural networks require symmetric actions, the code that sets the actions will correct for this
        self.max_thrust = rospy.get_param("/drone/max_thrust")
        self.min_thrust = -self.max_thrust
        
        
        # maximum allowed roll and pitch for the drone to be considered successfully landed
        # currently at 0.2 rad
        self.max_landing_roll = rospy.get_param("/drone/max_landing_roll")
        self.max_landing_pitch = rospy.get_param("/drone/max_landing_pitch")
        
        # Get Desired Point to land on
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/drone/desired_pose/x")
        self.desired_point.y = rospy.get_param("/drone/desired_pose/y")
        self.desired_point.z = rospy.get_param("/drone/desired_pose/z")
        # the tolerance around the desired point
        self.desired_point_epsilon = rospy.get_param("/drone/desired_point_epsilon")
        
        # actions and their number are defined by the arrays of upper and lower bounds
        # setpoints for roll and pitch allowed to be at most 0.3 of the flipping angle
        self.action_upper_bound = np.array([self.max_roll * 0.15,
                                            self.max_pitch * 0.15,
                                            self.max_yaw_rate,
                                            self.max_thrust])
                                        
        self.action_lower_bound = np.array([-1 * self.max_roll * 0.15,
                                            -1 * self.max_pitch * 0.15,
                                            -1 * self.max_yaw_rate,
                                            self.min_thrust])
    
        # initializing action space in the gym environment as the type Box
        self.action_space = spaces.Box(self.action_lower_bound, self.action_upper_bound,dtype=np.float32)
       
        # initializing actions for the current and the previous time step
        self.current_act = np.ones_like(self.action_lower_bound)
        self.current_scaled_act = np.ones_like(self.action_lower_bound)
        self.prev_act = np.ones_like(self.action_lower_bound)
        self.prev_scaled_act = np.ones_like(self.action_lower_bound)
        
        # Setting the reward range
        #  maybe  clip it to some exact values?
        self.reward_range = (-4000, 5000)
        
        # steps limit for each episode - equivalent to  10 seconds, since
        # the actions are taken at the rate of 10 Hz
        self.step_limit = 100;
        
        # Observations and their number are defined by their upper and lower bound
        # max roll and pitch are 2*pi/2
        self.max_yaw = 3.14
        # adding the simulated binary ground contact sensor C to the observation space 

        observation_upper_bound = np.array([self.work_space_x_max,
                                            self.work_space_y_max,
                                            self.work_space_z_max,
                                            self.max_roll,
                                            self.max_pitch,
                                            self.max_yaw,
                                            1,
                                            self.work_space_x_max,
                                            self.work_space_y_max,
                                            ])
                                        
        observation_lower_bound = np.array([self.work_space_x_min,
                                            self.work_space_y_min,
                                            self.work_space_z_min,
                                            -self.max_roll,
                                            -self.max_pitch,
                                            -1*self.max_yaw,
                                            0,
                                            self.work_space_x_min,
                                            self.work_space_y_min,
                                            ])
        
        # ddpg baseline does not work with non box spaces
        # creating a gym space for observations
        # always need to specify the dtype
        self.observation_space = spaces.Box(observation_lower_bound,
                                            observation_upper_bound, 
                                            dtype = np.float32)        
    
        
        # initializing observations for the current and the previous step 
        self.current_obs = np.zeros(9)
        self.current_scaled_obs = np.zeros(9)
        self.prev_obs =  np.zeros(9)
        self.prev_scaled_obs = np.zeros(9)

        
        # initializing position z log, starts empty for each episode
        # used for logging z at each time step and calculating the average landing
        # velocity right before contact
            
        self.height_log = []        

        
        # initializing the shaping function buffer
        self.prev_shaping_function_val = 0
        # Rewards

        # reward factors
                
        # succesfull landing reward
        self.safe_landing_points = rospy.get_param("/drone/safe_landing_points")
        # reward for not crashing or prematurely ending the episode
        # rewards exploration, could be increased a bit
        self.not_ending_point_reward = rospy.get_param("/drone/not_ending_point_reward")
        
        # reward factor if the drone ends up outside of the defined bounding box
        self.exited_bounding_box = rospy.get_param("/drone/exited_bounding_box")
        
        # reward factor if the engines are cut off at a high altitude
        self.early_engine_cutoff = rospy.get_param("/drone/early_engine_cutoff")
        
        # position shaping factor, rewarding the closeness to the desired position
        self.position_shaping_factor = rospy.get_param("/drone/position_shaping_factor")
        
        # control actions shaping factor, rewarding the "minimal" control signal values
        self.control_actions_shaping_factor = rospy.get_param("/drone/control_actions_shaping_factor")
        
        # fast landing bonus factor, in case of safely landing as early as possible
        # very small anyway, could consider removing it for now
        self.fast_landing_bonus_factor = rospy.get_param("/drone/fast_landing_bonus_factor")
        
        # 'vertical' landing bonus, rewarding small roll and pitch angles
        self.vertical_landing_reward = rospy.get_param("/drone/vertical_landing_reward")
        
        # reward approaching velocity to the desired landing spot
        self.approach_reward_factor = rospy.get_param("/drone/closer_to_point_reward")
        
        
        
        # reward indicators 
        
        # shaping function will not be used until the first step has passed and the previous observation is initialized
        self.use_shaping_function = False
        self.is_inside_workspace_now = True
        self.drone_flipped = False
        self.landed_safely = False
        self.engine_off = False
        self.desired_position_reached = False

        # indicates if ground contact occurred during the episode
        self.ground_contact = False
                
        # indicates high average velocity right before ground contact
        self.crashed = False

        self.engine_off_counter = 0
        self.cumulated_steps = 0.0
        self.cumulated_reward = 0.0
        
        # logging different parts of the reward 
        self.cumulated_shaping_reward = 0.0
        self.cumulated_flipping_reward = 0.0
        self.cumulated_approach_angle_reward = 0.0
        self.exiting_boundaries_reward = 0.0
        self.exploration_reward = 0.0
        self.safe_landing_reward = 0.0
        self.velocity_reward = 0.0
        self.early_stop_reward = 0.0
        self.goal_proximity_bonus = 0.0
        
 
        
        

        
#        rospy.logwarn("ENV INIT DONE")

    def _set_init_pose(self):
        """
        Sets the Robot in its init linear and angular speeds
        Its preparing it to be reseted in the world.
        
        HERE IMPLEMENT HOVERING?
        """

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode. 
        :return:
        """
        #raw_input("TakeOFF PRESS")
        # We TakeOff before sending any movement commands
#        self.takeoff()

        # For Info Purposes
        self.is_inside_workspace_now = True
        self.drone_flipped = False
        self.landed_safely = False
        self.engine_off = False
        self.desired_position_reached = False
        self.ground_contact = False
        self.crashed = False
        self.cumulated_steps = 0.0
        self.cumulated_reward = 0.0
        
        self.prev_shaping_function_val = 0
        self.engine_off_counter = 0
        self.current_obs = np.zeros(9)
        self.current_scaled_obs = np.zeros(9)
        self.current_scaled_act = np.ones(4)
        self.prev_obs =  np.zeros(9)
        self.prev_scaled_obs = np.zeros(9)
        self.current_act = np.ones(4)
        self.prev_act = np.ones(4)
        self.prev_scaled_act = np.ones(4)
        
        self.height_log = []
        
        
        # logging different parts of the reward 
        self.cumulated_shaping_reward = 0.0
        self.cumulated_flipping_reward = 0.0
        self.cumulated_approach_angle_reward = 0.0
        self.exiting_boundaries_reward = 0.0
        self.exploration_reward = 0.0
        self.safe_landing_reward = 0.0
        self.velocity_reward = 0.0
        self.early_stop_reward = 0.0
        self.goal_proximity_bonus = 0.0
    def _set_action(self, action):
        """
        Setting the setpoint parameters for the low level controller
        Input to this functions are action values generated by the neural network
        """
               
        
        # generating actions from the uniform distribution inside the specified allowed interval
        # used for initializing the networks and removing the training dependency on the initial parameters of the policy
        if (self.episode_num <= 100): 
            action = self.action_lower_bound + np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),
            random.uniform(0,1)]) * (self.action_upper_bound-self.action_lower_bound)
            self.action = action
        
   
                    # extracting actions from the neural network output
        cmd_roll = action[0]
        cmd_pitch = action[1]
        cmd_yaw_rate = action[2]
        cmd_thrust = action[3] + self.max_thrust # correcting for the symmetric range
        
        if self.ground_contact:
            self.engine_off = True
            
        # We pass to the rpyt controller the setpoints
        self.move_base(cmd_roll, cmd_pitch, cmd_yaw_rate, 
                       cmd_thrust, epsilon=0.05, update_rate=30, e_off = self.engine_off)
        
        # move base publishes the parameters, as well as the update rate to my rpyt low level controller
        # it waits for 1/30 seconds (adjusted for the realtime factor of the simulation) for the action to execute
        

        # scaling the current action
        # the reward shaping function uses the scaled action values                 
        current_scaled_act = [cmd_roll/self.max_roll,
                              cmd_pitch/self.max_pitch,
                              cmd_yaw_rate/self.max_yaw_rate,
                              cmd_thrust / (2*self.max_thrust)
                              ]

        # updating the action buffer for the shaping function
        self.update_action_buffer(action,current_scaled_act)
        
        

    def _get_obs(self):
        """
        Getting observations from the sensors and preparing them to be used
        as the input for the neural networks
        :return:
        """
        
        # We get the current odometry
        odom = self.get_odometry()
        
        # Converting the orientation quaternion to roll, pitch and yaw for control
        roll, pitch, yaw = self.get_orientation_euler(odom.pose.pose.orientation)
        
        # Reading from the simulated contact sensor
        # Contact is inferred using ground truth pose information in the simulation
        # This is not necessary in the actual real-world implementation
        
        gt_pose = self.get_pose()
        contact = self.touched_ground(gt_pose)
        self.ground_contact = contact
        # position observations are converted into position relative to the goal
        observations = [odom.pose.pose.position.x - self.desired_point.x,
                        odom.pose.pose.position.y - self.desired_point.y,
                        odom.pose.pose.position.z - self.desired_point.z,
                        roll,
                        pitch,
                        yaw,
                        contact,
                        odom.pose.pose.position.x - self.desired_point.x - self.prev_obs[0],
                        odom.pose.pose.position.y - self.desired_point.y - self.prev_obs[1]
                        ]       
        
        
        current_scaled_obs = [
                                (odom.pose.pose.position.x - self.desired_point.x)/self.work_space_x_max,
                                (odom.pose.pose.position.y - self.desired_point.y)/self.work_space_y_max,
                                (odom.pose.pose.position.z - self.desired_point.z)/self.work_space_z_max,
                                roll / self.max_roll,
                                pitch / self.max_pitch,
                                yaw / self.max_yaw,
                                contact,
                                odom.pose.pose.position.x - self.desired_point.x - self.prev_obs[0],
                                odom.pose.pose.position.y - self.desired_point.y - self.prev_obs[1]
                                ]     
        
        
        
        
        # updating values of current and previous observations
        self.update_pos_buffer(observations,current_scaled_obs)
        
        # logging the current height
        self.height_log.append(odom.pose.pose.position.z)
        
        return observations
        

    def _is_done(self, observations):
        """
        The episode can be done/finished due to following reasons:
        1) Drone went outside the workspace
        2) The steps limit reached
        3) It flipped
        4) It crashed    
        5) It has succesfully landed
        """
        
        episode_done = False
        
        current_position = Point()
        current_position.x = observations[0] + self.desired_point.x
        current_position.y = observations[1] + self.desired_point.y
        current_position.z = observations[2] + self.desired_point.z
        
        current_orientation = Point()
        current_orientation.x = observations[3]
        current_orientation.y = observations[4]
        current_orientation.z = observations[5]
        contact = observations[6]
        
        # checking end conditions        
        self.is_inside_workspace_now = self.is_inside_workspace(current_position)
        
        self.drone_flipped = self.drone_has_flipped(current_orientation)
        
        self.desired_position_reached = self.is_in_desired_position(current_position,self.desired_point_epsilon)
        
        # checks if ground contact ensues at high velocity  (more than >1.2 m/s average, >0.8 still penalised)
        self.crashed = (contact == 1) and (self.average_velocity_at_contact(1) >= 0.1)
        
        if (contact == 1): self.engine_off = True
        if self.engine_off:
            self.engine_off_counter += 1
        
        if (self.is_inside_workspace_now and (not self.drone_flipped) and (not(self.crashed))and (not(self.landed_safely))): 
            self.landed_safely = self.has_landed_safely(observations)

        
        
#        if self.drone_flipped:
#            rospy.logerr("drone_flipped="+str(self.drone_flipped))
#  
        if (contact == 1) and (self.average_velocity_at_contact(1) >= 0.1):
            rospy.logerr("drone_crashed")
#  
#        if (self.landed_safely):
#            rospy.logwarn("landed safely")

        
        # Determining if the episode is done based on the criteria
        episode_done = ( not(self.is_inside_workspace_now) or
                        (self.cumulated_steps  == (self.step_limit-1)) or 
                        self.crashed or
                        self.drone_flipped or
                        self.landed_safely or
                        current_position.z <= 0.075 
#                        self.engine_off_counter >=3
#                        current_position.z < 0.075
                        
                        )
        
        # reinitializing, might not be necessary
        if episode_done == True:
            self.engine_off = True
            
        return episode_done

    def _compute_reward(self, observations, done):
        
        reward = 0
        
        rewards_msg = Rewards()
        
        # shaping function only used after initalizing the action and observation buffers
        if not done:
                
            shaping_reward = (self.cumulated_steps>=3) * self.shaping_function_increment(self.current_scaled_obs, 
                                                                                         self.current_scaled_act,
                                                                                         self.prev_scaled_obs,
                                                                                         self.prev_scaled_act)
            self.cumulated_shaping_reward += shaping_reward 
            rewards_msg.shaping_rew = shaping_reward
            rewards_msg.cumulated_shaping_rew = self.cumulated_shaping_reward
            

            
            
            flipping_reward = (not self.drone_flipped) * (self.flip_reward(self.current_obs[3],self.prev_obs[3])*0.5 + 
                                                          self.flip_reward(self.current_obs[4],self.prev_obs[4])*0.5)
            if flipping_reward < - 2500:
                flipping_reward = -2500
                
            self.cumulated_flipping_reward += flipping_reward
            rewards_msg.flipping_rew = flipping_reward
            rewards_msg.cumulated_flipping_rew = self.cumulated_flipping_reward
            
            
            approach_angle_reward = (self.ground_contact) * self.approach_angle_reward(self.current_obs[3],self.current_obs[4]) 
            self.cumulated_approach_angle_reward += approach_angle_reward
            rewards_msg.approach_angle_rew = approach_angle_reward
            rewards_msg.cumulated_approach_angle_rew = self.cumulated_approach_angle_reward
            
            self.exiting_boundaries_reward = (not self.is_inside_workspace_now)*self.exited_bounding_box
            rewards_msg.exiting_boundaries_rew = self.exiting_boundaries_reward
            
            self.exploration_reward +=self.not_ending_point_reward
            rewards_msg.exploration_rew = self.exploration_reward
            
            # the ending rewards are being set to 0
            rewards_msg.safe_landing_rew = 0.0
            rewards_msg.approach_velocity_rew = 0.0
            rewards_msg.early_stop_rew = 0.0
            rewards_msg.goal_prox_bonus_rew = 0.0
            
            reward += 1000*shaping_reward/200 + 300*flipping_reward/2500 + 1000*approach_angle_reward/500 + 3000*self.exiting_boundaries_reward/3000 + 2*self.not_ending_point_reward/2
            if reward<-10000:
#                rospy.logwarn("Exploded")
                reward = -10000 
            
        else:
            
#            rospy.logwarn("EPISODE NUM = " + str(self.episode_num))
            
            self.safe_landing_reward = self.landed_safely * (self.safe_landing_points + 
                                                             self.desired_position_reached * 1000 + 
                                                             100.0 * (10 - np.sqrt(self.current_obs[0]**2 + self.current_obs[1]**2))) 
            rewards_msg.safe_landing_rew = self.safe_landing_reward

            
            self.exiting_boundaries_reward = (not self.is_inside_workspace_now)*self.exited_bounding_box 
            rewards_msg.exiting_boundaries_rew = self.exiting_boundaries_reward

            shaping_reward = self.shaping_function_increment(self.current_scaled_obs, 
                                                             self.current_scaled_act,
                                                             self.prev_scaled_obs,
                                                             self.prev_scaled_act)

            
            self.cumulated_shaping_reward += shaping_reward
            rewards_msg.shaping_rew = shaping_reward
            rewards_msg.cumulated_shaping_rew = self.cumulated_shaping_reward
            
            self.velocity_reward = self.approach_velocity_reward(self.average_velocity_at_contact(1))*(self.ground_contact)
            
            if self.velocity_reward < -2000:
                self.velocity_reward = -2000
            
            rewards_msg.approach_velocity_rew = self.velocity_reward
            

            
            self.early_stop_reward = (self.cumulated_steps<30)*(30-self.cumulated_steps)* (-50.0)
            rewards_msg.early_stop_rew = self.early_stop_reward
            
            self.goal_proximity_bonus = (not (self.landed_safely)) * 200* (-np.sqrt(self.current_obs[0]**2+
                                                                                 self.current_obs[1]**2 + 
                                                                                 self.current_obs[2]**2))
            rewards_msg.goal_prox_bonus_rew = self.average_velocity_at_contact(1)
            
            flipping_reward = 1 * (self.flip_reward(self.current_obs[3],self.prev_obs[3])*0.5 + 
                                   self.flip_reward(self.current_obs[4],self.prev_obs[4])*0.5)
            if flipping_reward < - 2500:
                flipping_reward = -2500
                
            self.cumulated_flipping_reward += flipping_reward 
            rewards_msg.flipping_rew = flipping_reward
            rewards_msg.cumulated_flipping_rew = self.cumulated_flipping_reward
            
          
            
            
            
            # setting the rest of the rewards to zero
            rewards_msg.approach_angle_rew = 0.0
            rewards_msg.cumulated_approach_angle_rew = 0.0
            rewards_msg.exploration_rew = self.exploration_reward

            
            reward = reward + 2000*self.safe_landing_reward/2100 + 3000*self.exiting_boundaries_reward/3000 + 2000*self.velocity_reward/1800 + 1500*self.early_stop_reward/1000 + 0*shaping_reward/200 + 2000*self.goal_proximity_bonus/2000 + 200*flipping_reward/2500
            
            if reward<-50000:
#                rospy.logwarn("Exploded")
                reward = -50000
     
        
        
        self.publish_rewards(rewards_msg)
        
        self.cumulated_reward += reward
        if self.cumulated_reward<-100000:
            self.cumulated_reward = -100000
            
        self.cumulated_steps += 1
        
        return reward


    # Internal TaskEnv Methods
    def touched_ground(self, gt_pose):
        """
        TO BE IMPLEMENTED
        Simulates contact sensor readings
        """
        
        # z_ground based on ground truth topic readings when the drone is on the ground
        z_ground = 0.08
        if gt_pose.position.z <=z_ground:
            c = 1
        else:
            c = 0
            
        return c
    
    def approach_velocity_reward(self,velocity):
        if velocity>1.6:
            return -20.0*np.exp((0.45*(np.abs(velocity)))**1)
        if (velocity<=1.6) and (velocity >=0.1):
            return - 12.5 * np.exp(2.1*(velocity-0.1))
        if velocity < 0.1:
            return +55.0 * np.exp(20*(0.1-velocity))     
 
        
    def approach_angle_reward(self,roll,pitch):
        if np.abs(roll) + np.abs(pitch) < 0.174:
            return 100*np.exp((7.0*(0.174-np.abs(roll) - np.abs(pitch)))**1)
        if (np.abs(roll) + np.abs(pitch)<=1.55)and(np.abs(roll) + np.abs(pitch) >=0.174):
            return -6.0*(np.exp((3.2*(np.abs(roll) + np.abs(pitch)-0.174))**1))
        if (np.abs(roll) + np.abs(pitch)>1.55):
            return -500.0
    
    
    def flip_reward(self,angle,prev_angle):
        if np.abs(angle) < 0.26:
            return 0.05*np.exp(20*(0.26-np.abs(angle)))
        if (np.abs(angle)>=0.26):
            return -7.0*np.exp((2.1*(np.abs(angle)-0.26))**1)

    
    def small_roll(self):
        return (self.current_obs[3] < 0.2)and(self.current_obs[3]>-0.2)
    
    def small_pitch(self):
        return (self.current_obs[4] < 0.2)and(self.current_obs[4]>-0.2)
    
    def has_landed_safely(self,observations):
        """
        TO BE IMPLEMENTED
        Tests if the drone has successfully landed using the following criteria:
            1) engines thrust should be off
            2) z position should be around zero (look at the exact ground truth value from gazebo 
            simulation when the drone is standing still on the ground)
            3) appropriate roll and pitch
            
        This function will be called only after checking if the drone has flipped and if it is still inside the
        workspace
        """
        if ((self.current_act[3] <= 0.5) and (observations[2]<=0.075) and 
            (observations[3]<self.max_landing_roll) and 
            (observations[3]>-self.max_landing_roll) and
            (observations[4]>-self.max_landing_pitch) and
            (observations[4]<self.max_landing_pitch)):
            return True
        else:
            return False
    
    
    def update_pos_buffer(self,new_observation,new_scaled_obs):
        """
        Updates the current and previous observation list based on the new observation
        Keeping track of current and previous observations is necessary for the shaping function         
        """
        self.prev_obs = self.current_obs
        self.current_obs = new_observation
        self.prev_scaled_obs = self.current_scaled_obs
        self.current_scaled_obs = new_scaled_obs
        return
    
    def update_action_buffer(self,new_action,new_scaled_act):
        """
        Updates the current and previous action list based on the new action
        Keeping track of current and previous observations is necessary for the shaping function
        """
        self.prev_act = self.current_act
        self.current_act = new_action
        self.prev_scaled_act = self.current_scaled_act
        self.current_scaled_act = new_scaled_act
        return
        
    def shaping_function(self,obs,act):
        """
        Shaping function based on Sampedro et al. IROS 2018
        """
                
        # implementation for relative coordinates as observations
        
        return  (-1.0 *self.position_shaping_factor * np.sqrt((obs[0])**2 + 
                                                              (obs[1])**2 + 
                                                              (obs[2])**2 + 
                                                               obs[5]**2) 
                 -1.0 * self.approach_reward_factor * np.sqrt((obs[0] - self.prev_obs[0])**2 + 
                                                              (obs[1] - self.prev_obs[1])**2)
                 -1.0 * self.control_actions_shaping_factor * np.sqrt(act[0]**2 + act[1]**2 + act[2]**2 + act[3]**2)
                 )           
    
    def shaping_function_increment(self,current_obs,current_act,previous_obs,previous_act):
        
        current_shaping_function_val = self.shaping_function(current_obs,current_act)
        increment = current_shaping_function_val - self.prev_shaping_function_val 
        self.prev_shaping_function_val = current_shaping_function_val
        
        return increment
    
    
    def is_in_desired_position(self,current_position, epsilon=0.05):
        """
        It return True if the current position is similar to the desired poistion
        """
        
        is_in_desired_pos = False
        
        
        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon
        
        x_current = current_position.x
        y_current = current_position.y
        
        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)
        
        is_in_desired_pos = x_pos_are_close and y_pos_are_close
        
        return is_in_desired_pos
    
    def is_inside_workspace(self,current_position):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_inside = False

        if current_position.x > self.work_space_x_min and current_position.x <= self.work_space_x_max:
            if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
                if current_position.z > self.work_space_z_min and current_position.z <= self.work_space_z_max:
                    is_inside = True
        
        return is_inside
        
        
    def drone_has_flipped(self,current_orientation):
        """
        Based on the orientation RPY given states if the drone has flipped
        """
        has_flipped = True
        if np.abs(current_orientation.x)<=self.max_roll/2 and np.abs(current_orientation.y)<=self.max_pitch/2:        
        
                    has_flipped = False
        
        return has_flipped
    
    def average_velocity_at_contact(self,time_frame):
        """
        Calculates the average z-axis velocity over the specified time_frame
        time_frame should be given in training steps (currently 1 step = 0.3 seconds)
        """
        
#        length = len(self.height_log) 
        
#        if length >(time_frame + 1): 
#            return np.abs((self.height_log[-1] - self.height_log[-1-time_frame])/(time_frame * 1/30))
#        else:
        return np.abs((self.height_log[-1] - self.height_log[-2])/(0.05))
        

        
    def get_orientation_euler(self, quaternion_vector):
        # We convert from quaternions to euler
        orientation_list = [quaternion_vector.x,
                            quaternion_vector.y,
                            quaternion_vector.z,
                            quaternion_vector.w]
    
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        return roll, pitch, yaw

    def get_action_space(self):
        return self.action_space
    
    def get_observation_space(self):
        return self.observation_space
    
    