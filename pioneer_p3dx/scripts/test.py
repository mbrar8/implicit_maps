#!/usr/bin/env python3

import sys
import time
import math
import random
import threading

import cv2
import numpy as np
from numpy import inf
import torch
import torch.nn.functional as F
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Image
from squaternion import Quaternion
from std_srvs.srv import Empty as EmptySrv
from visualization_msgs.msg import Marker, MarkerArray
from gazebo_msgs.srv import SetEntityState
from tf_transformations import euler_from_quaternion
from cv_bridge import CvBridge

import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from model import AgentLSTMPolicy, ProbeLSTMPolicy

from utils import *

# print all np values
np.set_printoptions(threshold=inf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
MAX_EP_STEPS = 650  # Maximum number of steps per episode


class CustomGazeboEnv(gym.Env):
    def __init__(self):
        super(CustomGazeboEnv, self).__init__()
        
        self.env = GazeboEnv()
        self.ep_timesteps = 0
        self.eval_mode = True

        # Action space: 2 continuous actions for velocity and angular velocity
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: match the structure of the state returned by GazeboEnv
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32),
            'imu': spaces.Box(low=-10, high=10, shape=(9,), dtype=np.float32),
            'goal_distance': spaces.Box(low=0.0, high=15.0, shape=(1,), dtype=np.float32),
            'goal_angle': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        })
        print("CustomGazeboEnv Initialized")

    def reset(self, seed=42):
        # Reset the Gazebo environment and return the initial state
        self.ep_timesteps = 0
        state = self.env.reset()

        if TRAIN_PROBE and self.eval_mode:
            agent_model = RecurrentPPO.load(f"{MODEL_PATH}/{MODEL_NAME}")
            self.eval_mode = False
            model_reward, std_reward = evaluate_policy(agent_model, self, n_eval_episodes=1)
            state = self.env.state
            self.env.pioneer_goal_x = self.env.home_x
            self.env.pioneer_goal_y = self.env.home_y

        return state, {}

    def step(self, action):
        # Take a step in the Gazebo environment using the action provided
        state, reward, terminated, truncated = self.env.step(action, self.ep_timesteps)
        self.ep_timesteps += 1
        
        return state, reward, terminated, truncated, {}

    def close(self):
        # Close the ROS 2 node and clean up
        rclpy.shutdown()
        self.env.destroy()

class GazeboEnv(Node):
    """Superclass for all Gazebo environments."""

    def __init__(self):
        super().__init__('env')

        self.pioneer_odom_x = 0.0
        self.pioneer_odom_y = 0.0
        self.pioneer_goal_x = 0.0
        self.pioneer_goal_y = 0.0
        self.home_x = 0.0
        self.home_y = 0.0

        self.lower = -4.75
        self.upper = 4.75
        self.prev_distance = 20.0
        self.unavailable_positions = {}

        self.state = None
        self.blackout = False

        # ROS publishers and clients
        self.publish_vel_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.publish_goal_point = self.create_publisher(MarkerArray, "/goal_point", 1)
        self.publish_home_point = self.create_publisher(MarkerArray, "/home_point", 1)
        self.publish_fisheye_camera = self.create_publisher(Image, "/fisheye_camera/image_raw_intermittent", 1)

        self.unpause_client = self.create_client(EmptySrv, "/unpause_physics")
        self.pause_client = self.create_client(EmptySrv, "/pause_physics")
        self.reset_proxy_client = self.create_client(EmptySrv, "/reset_world")
        self.entity_state_client = self.create_client(SetEntityState, '/set_entity_state')

        self.pioneer_vel_cmd = Twist()
        
        self.robot_name = "pioneer"
        self.set_entity_state(self.robot_name, self.pioneer_odom_x, self.pioneer_odom_y, z=0.01)

        self.get_logger().info("Let's TEST!")

    def set_entity_state(self, entity_name, x, y, z=0.0, ori_x=0.0, ori_y=0.0, ori_z=0.0, ori_w=1.0):
        # Wait until the service is available
        while not self.entity_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service to become available...')

        # Create a request
        request = SetEntityState.Request()
        request.state.name = entity_name
        request.state.pose.position.x = x
        request.state.pose.position.y = y
        if ("cardboard_box_" in entity_name):
            request.state.pose.position.z = 0.15
        else:
            request.state.pose.position.z = z
        request.state.pose.orientation.x = ori_x
        request.state.pose.orientation.y = ori_y
        request.state.pose.orientation.z = ori_z
        request.state.pose.orientation.w = ori_w
        request.state.reference_frame = "world"

        # Call the service
        self.entity_state_client.call_async(request)

    # Perform an action and read a new state
    def step(self, action):
        global velodyne_data
        target = False
        
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0])
        vel_cmd.angular.z = float(action[1])
        self.publish_vel_cmd.publish(vel_cmd)
        self.publish_markers(action)

        #rospy.wait_for_service("/gazebo/unpause_physics")
        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.unpause.call_async(EmptySrv.Request())
        except:
            print("/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            pass
            self.pause.call_async(EmptySrv.Request())
        except (rclpy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(velodyne_data)
        v_state = []
        v_state[:] = velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = last_odom.pose.pose.position.x
        self.odom_y = last_odom.pose.pose.position.y
        quaternion = Quaternion(
            last_odom.pose.pose.orientation.w,
            last_odom.pose.pose.orientation.x,
            last_odom.pose.pose.orientation.y,
            last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            env.get_logger().info("GOAL is reached!")
            target = True
            done = True

        # This is where I can add battery, other robots and goals
        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, min_laser)

        return state, reward, done, target

    def reset(self):
        global img_data

        self.prev_distance = 20.0

        self.pioneer_vel_cmd.linear.x = 0.0        # Pitch
        self.pioneer_vel_cmd.linear.y = 0.0
        self.pioneer_vel_cmd.linear.z = 0.0
        self.pioneer_vel_cmd.angular.x = 0.0
        self.pioneer_vel_cmd.angular.y = 0.0
        self.pioneer_vel_cmd.angular.z = 0.0       # Yaw
        self.publish_vel_cmd.publish(self.pioneer_vel_cmd)

        # Resets the state of the environment and returns an initial observation.
        #rospy.wait_for_service("/gazebo/reset_world")
        while not self.reset_proxy_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.reset_proxy_client.call_async(EmptySrv.Request())
        except rclpy.ServiceException as e:
            self.get_logger().info("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)

        self.clear_markers()
        self.random_obstacles()
        self.spawn_robot(quaternion)
        self.spawn_goal()
        self.publish_markers()

        while not self.unpause_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')

        try:
            self.unpause_client.call_async(EmptySrv.Request())
        except:
            self.get_logger().info("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        while not self.pause_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')

        try:
            self.pause_client.call_async(EmptySrv.Request())
        except:
            self.get_logger().info("/gazebo/pause_physics service call failed")

        
        state = {
            'image': np.asarray(img_data, dtype=np.float32),
            'imu': np.array([0.0] * 9),
            'goal_distance': np.array([0.0]),
            'goal_angle': np.array([0.0])
        }

        self.state = state
        
        return state

    def spawn_goal(self):
        """
        Spawn the goal in a random location in the environment
        """
        goal_ok = False
        goal_name = "goal"
        while not goal_ok:
            goal_x = random.uniform(self.lower, self.upper)
            goal_y = random.uniform(self.lower, self.upper)
            if check_position(goal_x, goal_y, obstacle_sizes[goal_name], goal_name, self.unavailable_positions):
                goal_ok = True
                self.pioneer_goal_x = goal_x
                self.pioneer_goal_y = goal_y

    def spawn_robot(self, quaternion):
        """
        Spawn the robot in a random location in the environment
        """
        position_ok = False
        while not position_ok:
            robot_x = random.uniform(self.lower, self.upper)
            robot_y = random.uniform(self.lower, self.upper)
            if check_position(robot_x, robot_y, obstacle_sizes[self.robot_name], self.robot_name, self.unavailable_positions):
                position_ok = True
                self.pioneer_odom_x = robot_x
                self.pioneer_odom_y = robot_y
                self.home_x = robot_x
                self.home_y = robot_y

        self.set_entity_state("pioneer", self.pioneer_odom_x, self.pioneer_odom_y, z=0.01, ori_x=quaternion.x, ori_y=quaternion.y, ori_z=quaternion.z, ori_w=quaternion.w)

    def spawn_obstacles(self, object_name, count):
        """
        Spawn the obstacles in a random location in the environment
        """
        for i in range(count):
            obstacle_name = object_name + "_" + str(i)
            position_ok = False
            while not position_ok:
                obstacle_x = random.uniform(self.lower, self.upper)
                obstacle_y = random.uniform(self.lower, self.upper)
                if check_position(obstacle_x, obstacle_y, obstacle_sizes[obstacle_name], obstacle_name, self.unavailable_positions):
                    position_ok = True
            
            self.set_entity_state(obstacle_name, obstacle_x, obstacle_y)

    def random_obstacles(self):
        """
        Randomize the environment by spawning obstacles in random locations
        """
        # self.spawn_obstacles('cardboard_box', 4)
        # self.spawn_obstacles('fire_hydrant', 2)
        # self.spawn_obstacles('bookshelf', 2)
        # self.spawn_obstacles('table', 2)

    def clear_markers(self):
        markerArray = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        markerArray.markers.append(marker)
        # Publish to all relevant topics to clear existing markers
        self.publish_goal_point.publish(markerArray)
        self.publish_home_point.publish(markerArray)

    def create_marker(self, frame_id, marker_type, scale, color, pose, action=Marker.ADD):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = marker_type
        marker.action = action
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = pose[0]
        marker.pose.position.y = pose[1]
        marker.pose.position.z = pose[2]
        return marker

    def publish_markers(self):
        # Publish visual data in Rviz
        markerArray1 = MarkerArray()
        markerArray1.markers.append(self.create_marker("odom", Marker.CYLINDER, [0.3, 0.3, 0.01], [0.0, 1.0, 0.0, 1.0], [self.pioneer_goal_x, self.pioneer_goal_y, 0.0]))
        self.publish_goal_point.publish(markerArray1)

        markerArray2 = MarkerArray()
        markerArray2.markers.append(self.create_marker("odom", Marker.CYLINDER, [0.3, 0.3, 0.01], [1.0, 1.0, 1.0, 1.0], [self.home_x, self.home_y, 0.0]))
        self.publish_home_point.publish(markerArray2)

    @staticmethod
    def get_reward(target, action, distance, prev_distance):
        # forward_reward_weight = 1.0   # Penalty weight for forward movement
        distance_reward_weight = 2.0    # Weight for the distance reward
        turning_penalty_weight = 0.15  # Penalty weight for turning
        step_penalty = 0.25             # Penalty for each step
        reward = 0.0

        if target:
            reward = 1000.0  # Reward for reaching a goal
        else:
            # # Reward for forward movement
            # forward_reward = action[0] * forward_reward_weight

            # Penalty for turning too much to prevent getting stuck in a loop
            turning_penalty = abs(action[1]) * turning_penalty_weight

            # Reward for reducing distance to the goal
            reward = max(0, (prev_distance - distance)) * distance_reward_weight
            if reward > 0:
                reward += action[0] * 1.5

            # Penalty for increasing distance from the goal
            reward -= max(0, (distance - prev_distance)) * 1.75

            reward += -step_penalty - turning_penalty

        return reward

class OdomSubscriber(Node):
    def __init__(self):
        super().__init__('odom_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.has_received_message = False

    def odom_callback(self, odom_msg):
        global odom_data
        odom_data = odom_msg
        self.has_received_message = True

class FisheyeCameraSubscriber(Node):
    def __init__(self):
        super().__init__('depth_camera_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/fisheye_camera/image_raw',
            self.camera_callback,
            10)
        self.has_received_message = False

    def camera_callback(self, img_msg):
        global img_data
        
        # Convert 3-channels ROS Image message to OpenCV image
        img = self.bridge.imgmsg_to_cv2(img_msg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img_data = np.transpose(img, (2, 0, 1))
        img_data = np.asarray(img_data, dtype=np.float32)

        self.has_received_message = True

class IMUSubscriber(Node):
    def __init__(self):
        super().__init__('imu_subscriber')
        self.subscription = self.create_subscription(
            Imu,
            '/imu_plugin/out',
            self.imu_callback,
            10)
        self.has_received_message = False

    def imu_callback(self, imu_msg):
        global imu_data
        imu_data = imu_msg
        self.has_received_message = True

def wait_for_subscribers(subscribers, timeout=30):
    start_time = time.time()
    rate = subscribers[0].create_rate(10)  # 10 Hz check rate
    
    while not all(sub.has_received_message for sub in subscribers):
        if time.time() - start_time > timeout:
            raise TimeoutError("Timed out waiting for subscribers to receive messages")
        rate.sleep()

    print("All subscribers have received their first messages")

if __name__ == '__main__':
    rclpy.init(args=None)

    # Set the parameters for the implementation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
    seed = 0  # Random seed number
    max_ep = 500  # maximum number of steps per episode
    file_name = "model"  # name of the file to load the policy from
    environment_dim = 30
    robot_dim = 4

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2

    # Create the network
    network = dream(state_dim, action_dim)
    try:
        # network.load(file_name, "/home/dipam/dream_ws/src/dream/dream/trials/trial_1/pytorch_models")
        network.load(file_name, "/home/dipam/dream_ws/src/dream/dream/model/")
    except:
        raise ValueError("Could not load the stored model parameters")

    done = True
    episode_timesteps = 0

    # Create the testing environment
    env = GazeboEnv()
    odom_subscriber = OdomSubscriber()
    velodyne_subscriber = VelodyneSubscriber()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(velodyne_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    rate = odom_subscriber.create_rate(2)

    # Begin the testing loop
    while rclpy.ok():

        # On termination of episode
        if done:
            state = env.reset()

            action = network.get_action(np.array(state))
            # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
            a_in = [(action[0] + 1) / 2, action[1]]
            next_state, reward, done, target = env.step(a_in)
            done = 1 if episode_timesteps + 1 == max_ep else int(done)

            done = False
            episode_timesteps = 0
        else:
            action = network.get_action(np.array(state))
            # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
            a_in = [(action[0] + 1) / 2, action[1]]
            next_state, reward, done, target = env.step(a_in)
            done = 1 if episode_timesteps + 1 == max_ep else int(done)

            state = next_state
            episode_timesteps += 1

    rclpy.shutdown()