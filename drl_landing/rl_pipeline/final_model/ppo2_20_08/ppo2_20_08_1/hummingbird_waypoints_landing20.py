#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
import numpy as np
from gym import spaces
from openai_ros.robot_envs import hummingbird_env
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion
from hummingbird.msg import Rewards
import multiprocessing
import random
import tf



if __name__ == '__main__':
    multiprocessing.freeze_support()


class HummingbirdLandBasic(hummingbird_env.HummingbirdEnv):
    def __init__(self,env_id):
        """
        Make the drone learn how to land safely on a specified location in an empty world
        """
        # Here we will add any init functions prior to starting the MyRobotEnv
        super(HummingbirdLandBasic, self).__init__(env_id = env_id)
        self.episode_num = 0
        rospy.logwarn("STARTED LEE")
        # reading parameters from the yaml file
        os.environ['ROS_PACKAGE_PATH']="/home/imartinovic/drl/drl_landing/rl_pipeline/catkin_ws/src:/opt/ros/melodic/share"

#        self.listener = tf.TransformListener()

        # Get WorkSpace Cube Dimensions
        self.work_space_x_max = rospy.get_param("/drone/work_space/x_max")  
        self.work_space_x_min = rospy.get_param("/drone/work_space/x_min") 
        self.work_space_y_max = rospy.get_param("/drone/work_space/y_max") 
        self.work_space_y_min = rospy.get_param("/drone/work_space/y_min") 
        self.work_space_z_max = rospy.get_param("/drone/work_space/z_max") - 2.5
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
#        odom = self.get_odometry()

        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/drone/desired_pose/x")
        
        self.desired_point.y = rospy.get_param("/drone/desired_pose/y")
        
        self.desired_point.z = 0.065
        # the tolerance around the desired point
        self.desired_point_epsilon = rospy.get_param("/drone/desired_point_epsilon")
        
        # actions and their number are defined by the arrays of upper and lower bounds
        # 5 m/s max for each axis
        self.action_upper_bound = np.array([5.0,
                                            5.0,
                                            5.0])
                                        
        self.action_lower_bound = np.array([-5.0,
                                            -5.0,
                                            -5.0])
    
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
#        self.step_limit = 64;
        self.step_limit = 128;
        
        # Observations and their number are defined by their upper and lower bound
        # max roll and pitch are 2*pi/2
        self.max_yaw = 3.14
        # adding the simulated binary ground contact sensor C to the observation space 

        observation_upper_bound = np.array([1.0,
                                            1.0,
                                            1.0,
                                            10.0
                                          ])
                                        
        observation_lower_bound = np.array([-1.0,
                                            -1.0,
                                            -1.0,
                                             0.0
                                        ])
        
        # ddpg baseline does not work with non box spaces
        # creating a gym space for observations
        # always need to specify the dtype
        self.observation_space = spaces.Box(observation_lower_bound,
                                            observation_upper_bound, 
                                            dtype = np.float32)        
    
        
        # initializing observations for the current and the previous step 
        self.current_obs = np.zeros(4)
        self.current_scaled_obs = np.zeros(4)
        self.prev_obs =  np.zeros(4)
        self.prev_scaled_obs = np.zeros(4)

        
        # initializing position z log, starts empty for each episode
        # used for logging z at each time step and calculating the average landing
        # velocity right before contact
            
        self.height_log = []        
        self.pos_x_log = [0.0]
        self.pos_y_log = [0.0]
        self.pos_z_log = [0.0]
        self.current_gt_orientation = [0.0,0.0,0.0]
        self.prev_gt_orientation = [0.0,0.0,0.0]

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
        
        # reward scaling seems quite important
        self.reward_scale = 4.0
        
        self.contact_counter = 0
        
        print("ENV INIT DONE")

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
        self.current_obs = np.zeros(4)
        self.current_scaled_obs = np.zeros(4)
        self.current_scaled_act = np.ones(3)
        self.prev_obs =  np.zeros(4)
        self.prev_scaled_obs = np.zeros(4)
        self.current_act = np.ones(3)
        self.prev_act = np.ones(3)
        self.prev_scaled_act = np.ones(3)
        
        self.height_log = []
        self.pos_x_log = [0.0]
        self.pos_y_log = [0.0]
        self.pos_z_log = [0.0]
        self.current_gt_orientation = [0.0,0.0,0.0]
        self.prev_gt_orientation = [0.0,0.0,0.0]

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
        
#        self.contact_counter = 0
#        
#        self.desired_point.x = -5 + random.uniform(0,10)
#        self.desired_point.y = -5 + random.uniform(0,10)

        
        
    def _set_action(self, action):
        """
        Setting the setpoint parameters for the low level controller
        Input to this functions are action values generated by the neural network
        """
               
        
        # generating actions from the uniform distribution inside the specified allowed interval
        # used for initializing the networks and removing the training dependency on the initial parameters of the policy
#        if (self.episode_num <= 100): 
#            action = self.action_lower_bound + np.array([random.uniform(0,1),random.uniform(0,1),random.uniform(0,1),
#            random.uniform(0,1)]) * (self.action_upper_bound-self.action_lower_bound)
#            self.action = action
#        
   
                    # extracting actions from the neural network output
        x_vel = action[0]
        y_vel = action[1]
        z_vel = action[2]
#        yaw_rate = action[3] # correcting for the symmetric range
        
        # not allowing aggressive manovers close to the ground
#        gt_position = np.array([self.pos_x_log[-1],self.pos_y_log[-1],self.pos_z_log[-1]])
#        desired_position = np.array([self.desired_point.x,self.desired_point.y,self.desired_point.z])
#        distance_to_goal = np.linalg.norm(gt_position-desired_position)
#        
##        if distance_to_goal < 0.5:
##            cmd_roll = action[0] * 0.2
##            cmd_pitch = action[1] * 0.2
        
#        if self.ground_contact:
#            self.engine_off = True
            
        # We pass to the rpyt controller the setpoints

        self.move_base(action, update_rate=100, init= False)
        
        # move base publishes the parameters, as well as the update rate to my rpyt low level controller
        # it waits for 1/30 seconds (adjusted for the realtime factor of the simulation) for the action to execute
        

        # scaling the current action
        # the reward shaping function uses the scaled action values                 
        current_scaled_act = [x_vel/5.0,
                              y_vel/5.0,
                              z_vel/5.0,
#                              yaw_rate/2.0
                              ]

        # updating the action buffer for the shaping function
        
        self.update_action_buffer(action,current_scaled_act)
        
        # moving target
        

    def _get_obs(self):
        """
        Getting observations from the sensors and preparing them to be used
        as the input for the neural networks
        :return:
        """
#        print("GETTING OBS")
        # We get the current odometry
        odom = self.get_odometry()
        
        # Converting the orientation quaternion to roll, pitch and yaw for control
        roll, pitch, yaw = self.get_orientation_euler(odom.pose.pose.orientation)
        
        # Reading from the simulated contact sensor
        # Contact is inferred using ground truth pose information in the simulation
        # This is not necessary in the actual real-world implementation
        
        gt_pose = self.get_pose()
        gt_roll, gt_pitch, gt_yaw = self.get_orientation_euler(gt_pose.orientation)
        self.current_gt_orientation = [gt_roll,gt_pitch,gt_yaw]
        
        contact = self.touched_ground(gt_pose)
        self.ground_contact = contact
        
        if self.ground_contact:
            self.contact_counter +=1
        
#        gt_position = np.array([self.pos_x_log[-1],self.pos_y_log[-1]])
#        desired_position = np.array([self.desired_point.x,self.desired_point.y])
#        distance_to_goal = np.linalg.norm(gt_position-desired_position)
        # for neural network input
        # position observations are converted into position relative to the goal
        observations = [
                        odom.pose.pose.position.x - self.desired_point.x,
                        odom.pose.pose.position.y - self.desired_point.y,
                        odom.pose.pose.position.z - self.desired_point.z,
                        self.average_velocity_at_contact(1)
                        ]       
        
        # using ground truth data for reward calculation 
        self.pos_x_log.append(gt_pose.position.x - self.desired_point.x)
        self.pos_y_log.append(gt_pose.position.y  - self.desired_point.y)
        self.pos_z_log.append(gt_pose.position.z  - self.desired_point.z)
        
        current_scaled_obs = [
                                self.pos_x_log[-1]/12.0,
                                self.pos_y_log[-1]/12.0,
                                self.pos_z_log[-1]/12.0,
                                self.average_velocity_at_contact(1)/10.0
                                ]     
        
#        self.desired_point.y += 0.02

        
#        if (contact == 1): self.engine_off = True

        # updating values of current and previous observations
        self.update_pos_buffer(observations,current_scaled_obs,self.current_gt_orientation)
        
        # logging the current height
        self.height_log.append(odom.pose.pose.position.z)
#        print("obs received")
        return current_scaled_obs
        

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
        current_position.x = self.pos_x_log[-1] + self.desired_point.x
        current_position.y = self.pos_y_log[-1] + self.desired_point.y
        current_position.z = self.pos_z_log[-1] + self.desired_point.z
        
        current_orientation = Point()
        current_orientation.x = self.current_gt_orientation[0]
        current_orientation.y = self.current_gt_orientation[1]
        current_orientation.z = self.current_gt_orientation[2]
        contact = self.ground_contact
        
        # checking end conditions        
        self.is_inside_workspace_now = self.is_inside_workspace(current_position)
        
        self.drone_flipped = self.drone_has_flipped(current_orientation)
        
        self.desired_position_reached = self.is_in_desired_position(current_position,self.desired_point_epsilon)
        
        # checks if ground contact ensues at high velocity  (more than >1.2 m/s average, >0.8 still penalised)
        self.crashed = (contact == 1) and (self.average_velocity_at_contact(6) >= 1.2)
        
#        if (contact == 1)and(self.desired_position_reached): self.engine_off = True
        if self.engine_off:
            self.engine_off_counter += 1
        
        if (self.is_inside_workspace_now and (not self.drone_flipped) and (not(self.crashed))and (not(self.landed_safely)))and(contact==1) and(self.desired_position_reached): 
            self.landed_safely = self.has_landed_safely(observations)

        
        
        if self.drone_flipped:
            rospy.logerr("drone_flipped="+str(self.drone_flipped))
  
        # average velocity inside the last 0.3 seconds before contact
        # taking the velocity directly at contact seems very error prone
        if (contact == 1) and (self.average_velocity_at_contact(4) >= 1.2):
            rospy.logerr("drone_crashed")
        
        if (self.desired_position_reached):
            rospy.logwarn("DESIRED POSITION REACHED")
         
            
        if (self.landed_safely):
            rospy.logwarn("landed safely")
            
        gt_position = np.array([self.pos_x_log[-1],self.pos_y_log[-1]])
#        desired_position = np.array([self.desired_point.x,self.desired_point.y])
        distance_to_goal = np.linalg.norm(gt_position)
        
        # Determining if the episode is done based on the criteria
        episode_done = ( 

                        (self.cumulated_steps  == (self.step_limit-1)) 
#                        not(self.is_inside_workspace_now) or
#                        self.landed_safely or
#                        self.pos_z_log[-1] <= 0.075 or

#                        self.engine_off_counter >=3
#                        current_position.z < 0.075
                        
                        )
        
#        if self.pos_z_log[-1] <= 0.075 :
#            self.engine_off = True

        # reinitializing, might not be necessary
        if episode_done == True:
            rospy.logwarn("desired y = " + str(self.desired_point.y))    
            rospy.logwarn("current y = " + str(self.pos_y_log[-1]))
            rospy.logwarn("desired x = " + str(self.desired_point.x))    
            rospy.logwarn("current x = " + str(self.pos_x_log[-1]))
            rospy.logwarn("distancexy = " + str(distance_to_goal))
            rospy.logwarn("current z = " + str(self.pos_z_log[-1]))

            rospy.logwarn("drone velocity = " + str(self.average_velocity_at_contact(3)))   
        return episode_done

    def _compute_reward(self, observations, done):
        
        reward = 0
#        rospy.logerr(str(self.desired_point.x) + " " + str(self.desired_point.y) + " " + str(self.desired_point.z))
        # currently not publishing since it seemed to cause sync problems
        # but it would be worth to try to figure it out
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
            

            
            
            flipping_reward = 0.5 * ((self.flip_reward(self.current_gt_orientation[0],self.prev_gt_orientation[0])*0.5) + 
                                      self.flip_reward(self.current_gt_orientation[1],self.prev_gt_orientation[1] *0.5))
            if flipping_reward < - 2500:
                flipping_reward = -2500
                
            self.cumulated_flipping_reward += flipping_reward
            rewards_msg.flipping_rew = flipping_reward
            rewards_msg.cumulated_flipping_rew = self.cumulated_flipping_reward
            
            
            approach_angle_reward = (self.ground_contact) * (self.desired_position_reached) * self.approach_angle_reward(self.current_gt_orientation[0],
                                                                                       self.current_gt_orientation[1]) 
            self.cumulated_approach_angle_reward += approach_angle_reward
            rewards_msg.approach_angle_rew = approach_angle_reward
            rewards_msg.cumulated_approach_angle_rew = self.cumulated_approach_angle_reward
            
            self.exiting_boundaries_reward = (not self.is_inside_workspace_now)*self.exited_bounding_box
            rewards_msg.exiting_boundaries_rew = self.exiting_boundaries_reward
            
            self.exploration_reward += self.not_ending_point_reward
            rewards_msg.exploration_rew = self.exploration_reward
            
            # the ending rewards are being set to 0
            rewards_msg.safe_landing_rew = 0.0
            rewards_msg.approach_velocity_rew = 0.0
            rewards_msg.early_stop_rew = 0.0
            rewards_msg.goal_prox_bonus_rew = 0.0
            
            gt_position = np.array([self.pos_x_log[-1]/12.0,self.pos_y_log[-1]/12.0])
            distance_to_goal = np.linalg.norm(gt_position)
#            self.pos_x_log[-1]/np.abs(self.desired_point.x),
#            self.pos_y_log[-1]/np.abs(self.desired_point.y),
#            self.pos_z_log[-1]/np.abs(self.desired_point.z),
#            
#            rospy.logwarn("REWXY = "  + str(0.1 * (-distance_to_goal**2)/4))
            # reward slowing down 1m off the ground
            
            
            rewxy = - 0.5 * (distance_to_goal**2)
#            rospy.logwarn("REWXY = "  + str(rewxy))
#            rospy.logerr(self.pos_z_log[-1])
            
#            rewz = -0.25 * ((self.pos_z_log[-1]/12.0)**2)
#            rospy.logwarn("REWZ = "  + str(rewz))
            rewz = -0.5 * ((self.pos_z_log[-1]/12.0)**2)
            
            self.velocity_reward = self.approach_velocity_reward(self.average_velocity_at_contact(4))*(self.pos_z_log[-1] <0.5)*(distance_to_goal<=0.03 and distance_to_goal>=0.01)
            reward = reward + self.reward_scale*((shaping_reward)*0.0 +
                                                 (0.0)*((7-distance_to_goal)**2) * (distance_to_goal > 0.4) *(distance_to_goal < 7.0) +
                                                 
                                                  0*(40 + 0.1 *((100*(0.4-distance_to_goal))**2)) * (distance_to_goal < 0.4) +
                                                 rewxy + 
                                                 (0.02 + (distance_to_goal<=0.04 and distance_to_goal>=0.01)*(self.pos_z_log[-1] <0.5)* 0.15)  *flipping_reward/2500 +
                                                 0.0*self.exiting_boundaries_reward/3000 + 
                                                 0.5*self.velocity_reward/500  +
                                                 0.25 * approach_angle_reward/100 +
                                                 1.0*self.desired_position_reached*self.safe_landing_reward/70 +
                                                 rewz
                                                 )
            
            
#            rewflip = (0.1 + (distance_to_goal<=0.03 and distance_to_goal>=0.01) * 0.1)  * flipping_reward/250
#            rospy.logwarn("REWFLIP = "  + str(rewflip))

            
            
            if reward<-20.0:reward = -20.0
            if reward> 1000.0:reward = 1000.0
#            self.desired_point.x += 0.05
#            self.desired_point.y += 0.05

#            rospy.logwarn("REWTOTAL = " + str(reward))
        else:
            rospy.logerr("END EP")            
            self.safe_landing_reward = self.landed_safely * self.safe_landing_points
                                
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
            
            gt_position = np.array([self.pos_x_log[-1]/12.0,self.pos_y_log[-1]/12.0])
            distance_to_goal = np.linalg.norm(gt_position)
            self.velocity_reward = self.approach_velocity_reward(self.average_velocity_at_contact(2))*(self.ground_contact)*(self.pos_z_log[-1] <0.08)
            
            if self.velocity_reward < -2000:
                self.velocity_reward = -2000
            
            rewards_msg.approach_velocity_rew = self.velocity_reward
            

            
            self.early_stop_reward = (self.cumulated_steps<40)*(100-self.cumulated_steps)* (-5.0)
            rewards_msg.early_stop_rew = self.early_stop_reward
            
      
            rewards_msg.goal_prox_bonus_rew = self.average_velocity_at_contact(4)
            
            flipping_reward = 0.0 + ((self.flip_reward(self.current_gt_orientation[0],self.prev_gt_orientation[0])*0.5) + 
                                      self.flip_reward(self.current_gt_orientation[1],self.prev_gt_orientation[1] *0.5))
            if flipping_reward < - 2500:
                flipping_reward = -2500
                
            self.cumulated_flipping_reward += flipping_reward 
            rewards_msg.flipping_rew = flipping_reward
            rewards_msg.cumulated_flipping_rew = self.cumulated_flipping_reward
            
          
            
            
            
            # setting the rest of the rewards to zero
            rewards_msg.approach_angle_rew = 0.0
            rewards_msg.cumulated_approach_angle_rew = 0.0
            rewards_msg.exploration_rew = self.exploration_reward
            

            
            self.goal_proximity_bonus = -1.0* distance_to_goal**2
            
            reward = reward + self.reward_scale*(20.0*self.desired_position_reached*self.safe_landing_reward/70 +
                                                 0.0*self.desired_position_reached*(1+0.03*(100-self.pos_z_log[-1]**2)) +
                                                 0.0*self.exiting_boundaries_reward/3000 + 
                                                 0.5*self.velocity_reward/500 + 
                                                 (-0.5)*self.drone_flipped 
                                                 
                                                 )
            
            if reward<-20.0:reward = -20.0
            if reward> 1000.0:reward = 1000.0
            self.desired_point.x = random.uniform(0,5)
            self.desired_point.y = random.uniform(0,5)
#            rospy.logerr("REWTOTAL = " + str(reward))
#                                distance_to_goal>np.linalg.norm(desired_position) + 2.0 or
#                        np.linalg.norm(gt_position) > np.linalg.norm(desired_position) + 2.0
        
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
        z_ground = 0.06
        if gt_pose.position.z <=z_ground:
            c = 1
        else:
            c = 0
            
        return c
    
    def approach_velocity_reward(self,velocity):
        if velocity>7.6:
            return -2.5*np.exp((0.45*(np.abs(velocity)))**1)
        if (velocity<=7.6) and (velocity >=6.2):
            return - 15.5 * np.exp(2.1*(velocity-6.2))
        if (velocity>=0.1) and (velocity < 6.2):
            return 2 * np.exp(1*(6.2-velocity))     
        if velocity<0.1:
            return 5.0
        
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
        return (self.current_gt_orientation[0] < 0.2)and(self.current_gt_orientation[0] >-0.2)
    
    def small_pitch(self):
        return (self.current_gt_orientation[1] < 0.2)and(self.current_gt_orientation[1]>-0.2)
    
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
        if ( (self.pos_z_log[-1] + self.desired_point.z <= 0.075) and 
            (self.current_gt_orientation[0]<self.max_landing_roll) and 
            (self.current_gt_orientation[0]>-self.max_landing_roll) and
            (self.current_gt_orientation[1]>-self.max_landing_pitch) and
            (self.current_gt_orientation[1]<self.max_landing_pitch)):
            return True
        else:
            return False
    
    
    def update_pos_buffer(self,new_observation,new_scaled_obs,new_gt_orient):
        """
        Updates the current and previous observation list based on the new observation
        Keeping track of current and previous observations is necessary for the shaping function         
        """
        self.prev_obs = self.current_obs
        self.current_obs = new_observation
        self.prev_scaled_obs = self.current_scaled_obs
        self.current_scaled_obs = new_scaled_obs
        self.prev_gt_orientation = self.current_gt_orientation
        self.current_gt_orientation = new_gt_orient        
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
        
        return  (-1.0 * (obs[0]**2 + obs[1]**2) 
                 -0.0 * self.control_actions_shaping_factor * np.sqrt(act[0]**2 + act[1]**2 + act[2]**2)
                 -0.0* 10 * ((obs[0] - self.prev_obs[0])**2 + (obs[1] - self.prev_obs[1])**2)
                  
                 )           
    
    def shaping_function_increment(self,current_obs,current_act,previous_obs,previous_act):
        
        current_shaping_function_val = self.shaping_function(current_obs,current_act)
        increment = current_shaping_function_val - self.prev_shaping_function_val 
        self.prev_shaping_function_val = current_shaping_function_val
#        rospy.logwarn("val = " + str(current_shaping_function_val))
#        rospy.logwarn("inc = " + str(increment))
        if(np.abs(current_shaping_function_val>10)): current_shaping_function_val = -10
        if(increment>10) : increment = 10.0
        if(increment<-10) : increment = -10.0

        return (current_shaping_function_val + 10)/10 + increment/10
    
    
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
        
        is_in_desired_pos = x_pos_are_close and y_pos_are_close and self.pos_z_log[-1] <0.07
        
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
        Calculates the average velocity over the specified time_frame
        time_frame should be given in training steps (currently 1 step = 0.05 seconds)
        """
        if len(self.pos_x_log) > time_frame :
            return np.sqrt( ((self.pos_x_log[-1] - self.pos_x_log[-1-time_frame])**2 + 
                          (self.pos_y_log[-1] - self.pos_y_log[-1-time_frame])**2 + 
                          (self.pos_z_log[-1] - self.pos_z_log[-1-time_frame])**2)  / ((0.05*time_frame)**2))
        else:
            return 1.0

        
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
    
    

    