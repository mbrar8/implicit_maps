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
from torch.utils.tensorboard import SummaryWriter
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
MAX_EP_STEPS = 600  # Maximum number of steps per episode

class CustomGazeboEnv(gym.Env):
    def __init__(self):
        super(CustomGazeboEnv, self).__init__()
        
        self.env = GazeboEnv()
        self.ep_timesteps = 0
        self.total_timesteps = 0
        self.ep_rewards = 0
        self.eval_mode = True

        # Action space: 2 continuous actions for velocity and angular velocity
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        # Observation space: match the structure of the state returned by GazeboEnv
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32),
            'imu': spaces.Box(low=-20, high=20, shape=(9,), dtype=np.float32),
            'goal_distance': spaces.Box(low=0.0, high=15.0, shape=(1,), dtype=np.float32),
            'goal_angle': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
            # 'imu': spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32),
            # 'goal_distance': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            # 'goal_angle': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        print("CustomGazeboEnv Initialized")

    def reset(self, seed=42):
        # Reset the Gazebo environment and return the initial state
        print(f"Episode Rewards: {self.ep_rewards}")
        self.ep_timesteps = 0
        self.ep_rewards = 0
        state = self.env.reset()

        if TRAIN_PROBE and self.eval_mode:
            agent_model = RecurrentPPO.load(f"{MODEL_PATH}/{MODEL_NAME}")
            self.eval_mode = False
            model_reward, std_reward = evaluate_policy(agent_model, self, n_eval_episodes=1)
            state = self.env.state
            self.env.goal_x = self.env.home_x
            self.env.goal_y = self.env.home_y

        return state, {}

    def step(self, action):
        # Take a step in the Gazebo environment using the action provided
        state, reward, terminated, truncated = self.env.step(action, self.ep_timesteps, self.total_timesteps)
        self.ep_timesteps += 1
        self.total_timesteps += 1
        self.ep_rewards += reward

        return state, reward, terminated, truncated, {}

    def close(self):
        # Close the ROS 2 node and clean up
        rclpy.shutdown()
        self.env.destroy()

class GazeboEnv(Node):
    """Superclass for all Gazebo environments."""

    def __init__(self):
        super().__init__('env')
        
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.goal_x = 3.0
        self.goal_y = 3.0
        self.home_x = 0.0
        self.home_y = 0.0

        self.lower = -4.25      # -4.75
        self.upper = 4.25       # 4.75
        self.collision_lower = -4.75
        self.collision_upper = 4.75
        self.min_distance = 20.0
        self.prev_distance = 20.0
        self.prev_theta = 0.0
        self.erratic_penalty = 0.0
        self.unavailable_positions = {}

        self.state = None
        self.blackout = False
        self.reward_writer = SummaryWriter(log_dir=REWARDS_PATH)

        # ROS publishers and clients
        self.publish_vel_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.publish_goal_point = self.create_publisher(MarkerArray, "/goal_point", 1)
        self.publish_home_point = self.create_publisher(MarkerArray, "/home_point", 1)
        self.publish_fisheye_camera = self.create_publisher(Image, "/fisheye_camera/image_raw_intermittent", 1)

        self.unpause_client = self.create_client(EmptySrv, "/unpause_physics")
        self.pause_client = self.create_client(EmptySrv, "/pause_physics")
        self.reset_proxy_client = self.create_client(EmptySrv, "/reset_world")
        self.entity_state_client = self.create_client(SetEntityState, '/set_entity_state')

        self.vel_cmd = Twist()
        
        # self.robot_name = "pioneer"
        self.robot_name = "turtlebot3"

        self.set_entity_state(self.robot_name, self.odom_x, self.odom_y, z=0.01)

        self.get_logger().info("Let's TRAIN!")

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

    def normalize_imu(self, imu):
        max_angular_velocity = 2 * math.pi
        max_acceleration = 19.62

        normalized_imu = [
            imu[0] / math.pi,
            imu[1] / math.pi,
            imu[2] / math.pi,
            imu[3] / max_angular_velocity,
            imu[4] / max_angular_velocity,
            imu[5] / max_angular_velocity,
            imu[6] / max_acceleration,
            imu[7] / max_acceleration,
            imu[8] / max_acceleration
        ]

        return normalized_imu

    # Perform an action and read a new state
    def step(self, action, ep_timesteps, total_timesteps):
        global odom_data
        global img_data
        global imu_data
        target = False
        terminated = False
        truncated = False
        current_timestep = 0
        reward = 0.0
        linear_x = 0.0
        angular_z = 0.0

        if action == 0:
            linear_x = 1.0
        elif action == 1:
            linear_x = -0.5
        elif action == 2:
            angular_z = 0.5
        elif action == 3:
            angular_z = -0.5

        # self.vel_cmd.linear.x = float(action[0])        # Pitch
        self.vel_cmd.linear.x = linear_x
        self.vel_cmd.linear.y = 0.0
        self.vel_cmd.linear.z = 0.0
        self.vel_cmd.angular.x = 0.0
        self.vel_cmd.angular.y = 0.0
        self.vel_cmd.angular.z = angular_z
        # self.vel_cmd.angular.z = float(action[1])       # Yaw
        self.publish_vel_cmd.publish(self.vel_cmd)

        while not self.unpause_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.unpause_client.call_async(EmptySrv.Request())
        except:
            self.get_logger().info("/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        while not self.pause_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            pass
            self.pause_client.call_async(EmptySrv.Request())
        except (rclpy.ServiceException) as e:
            self.get_logger().info("/gazebo/pause_physics service call failed")

        # Process IMU data
        orientation_q = imu_data.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        roll, pitch, yaw = euler_from_quaternion(orientation_list)
        
        angular_velocity = imu_data.angular_velocity
        linear_acceleration = imu_data.linear_acceleration
        
        # Combine into state vector
        imu = [
                roll, pitch, yaw,
                angular_velocity.x, angular_velocity.y, angular_velocity.z,
                linear_acceleration.x, linear_acceleration.y, linear_acceleration.z
            ]
        
        # imu = self.normalize_imu(imu)

        self.odom_x = odom_data.pose.pose.position.x
        self.odom_y = odom_data.pose.pose.position.y
        quaternion = Quaternion(
            odom_data.pose.pose.orientation.w,
            odom_data.pose.pose.orientation.x,
            odom_data.pose.pose.orientation.y,
            odom_data.pose.pose.orientation.z,
        )

        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        roll = round(euler[0], 4)
        pitch = round(euler[1], 4)

        distance = np.linalg.norm([self.goal_x - self.odom_x, self.goal_y - self.odom_y])

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        mag = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        beta = math.acos(skew_x / mag)
        
        if skew_y < 0:
            beta = -beta
        theta = beta - angle
        
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if total_timesteps < 200000:
            dist_thresh = GOAL_REACHED_DIST - (0.00000375 * total_timesteps)
        else:
            dist_thresh = 0.25
        if distance <= dist_thresh:
            self.get_logger().info(f"*********************** GOAL ***********************")
            target = True
            terminated = True
            # reward += (MAX_EP_STEPS - ep_timesteps) * 0.1

        reward, erratic_penalty = self.get_reward(target, action, distance, self.min_distance, self.prev_distance, theta, self.prev_theta, roll, pitch)
        self.erratic_penalty += erratic_penalty

        if distance < self.min_distance:
            self.min_distance = distance

        if ep_timesteps >= MAX_EP_STEPS:
            self.get_logger().info(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxx TRUNCATED xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            truncated = True

        if self.erratic_penalty >= 25.0:
            self.get_logger().info(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxx ERRATIC xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            reward -= 500.0
            terminated = True

        # Penalize if the agent goes beyond the boundaries of the environment
        if self.odom_x < self.collision_lower or self.odom_x > self.collision_upper or self.odom_y < self.collision_lower or self.odom_y > self.collision_upper:
            # self.get_logger().info(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxx COLLISION xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            # terminated = True
            reward -= 0.05

        if ENV_SETUP == 4 or ENV_SETUP == 5:
            if not self.blackout and random.random() < 0.20:
                self.blackout = True
                current_timestep = ep_timesteps
                self.get_logger().info(f"~~~~~~~~~~~~~~~~~~~~~~~~~~ BLACKOUT ~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
        if self.blackout and ep_timesteps - current_timestep < 50:
            # Create a black image using opencv
            img_data = np.zeros((3, 64, 64), dtype=np.float32)
        else:
            self.blackout = False

        img_data_msg = (img_data * 255.0).astype(np.uint8)  # Scale float32 to uint8 (0-255)
        img_data_msg = np.transpose(img_data_msg, (1, 2, 0))  # Transpose to (height, width, channels)

        # Publish the camera image
        img_msg = Image()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "numpy_frame"
        img_msg.height = img_data_msg.shape[0]
        img_msg.width = img_data_msg.shape[1]
        img_msg.encoding = "rgb8"
        img_msg.step = img_msg.width * 3
        img_msg.data = img_data_msg.tobytes()
        self.publish_fisheye_camera.publish(img_msg)

        state = {
            'image': np.asarray(img_data, dtype=np.float32),
            'imu': np.array(imu),
            # 'goal_distance': np.array([distance/15.0]),
            # 'goal_angle': np.array([theta/np.pi])
            'goal_distance': np.array([distance]),
            'goal_angle': np.array([theta])
        }

        self.state = state
        self.prev_distance = distance
        self.prev_theta = theta

        return state, reward, terminated, truncated

    def reset(self):
        global img_data

        self.min_distance = 20.0
        self.prev_distance = 20.0
        self.prev_theta = 0.0
        self.erratic_penalty = 0.0

        self.vel_cmd.linear.x = 0.0        # Pitch
        self.vel_cmd.linear.y = 0.0
        self.vel_cmd.linear.z = 0.0
        self.vel_cmd.angular.x = 0.0
        self.vel_cmd.angular.y = 0.0
        self.vel_cmd.angular.z = 0.0       # Yaw
        self.publish_vel_cmd.publish(self.vel_cmd)

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
                self.goal_x = goal_x
                self.goal_y = goal_y

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
                self.odom_x = robot_x
                self.odom_y = robot_y
                self.home_x = robot_x
                self.home_y = robot_y

        self.set_entity_state(self.robot_name, self.odom_x, self.odom_y, z=0.01, ori_x=quaternion.x, ori_y=quaternion.y, ori_z=quaternion.z, ori_w=quaternion.w)

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
        self.spawn_obstacles('bookshelf', 4)

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
        markerArray1.markers.append(self.create_marker("odom", Marker.CYLINDER, [0.3, 0.3, 0.01], [0.0, 1.0, 0.0, 1.0], [self.goal_x, self.goal_y, 0.0]))
        self.publish_goal_point.publish(markerArray1)

        markerArray2 = MarkerArray()
        markerArray2.markers.append(self.create_marker("odom", Marker.CYLINDER, [0.3, 0.3, 0.01], [1.0, 1.0, 1.0, 1.0], [self.home_x, self.home_y, 0.0]))
        self.publish_home_point.publish(markerArray2)

    def get_reward(self, target, action, distance, min_distance, prev_distance, theta, prev_theta, roll, pitch):
        forward_reward_weight = 1.0   # Penalty weight for forward movement
        distance_reward_weight = 0.2  # 0.25  # Weight for the distance reward
        distance_penalty_weight = 1.5
        turning_penalty_weight = 0.0001    # Penalty weight for turning
        theta_reward_weight = 0.001
        reward = 0.0
        linear_reward = 0.0
        turning_penalty = 0.0
        theta_reward = 0.0
        theta_penalty = 0.0
        stuck_penalty = 0.0
        erratic_penalty_weight = 0.5  # Weight for penalty
        erratic_penalty = 0.0

        if target:
            reward = 100.0  # Reward for reaching a goal
        else:
            # # reward = action[0] * 0.2 - abs(action[1]) * 0.45

            # reward = -abs(action[1]) * 0.02
            
            # # Reward for forward movement
            # forward_reward = action[0] * forward_reward_weight

            # # Penalty for turning too much to prevent getting stuck in a loop
            # # turning_penalty = abs(action[1]) * turning_penalty_weight
            # if action == 1 or action == 2:
            #     turning_penalty = turning_penalty_weight

            # reward += max(0, (min_distance - distance)) * distance_reward_weight
            
            # Reward for reducing distance to the goal
            distance_reward = max(0, (prev_distance - distance)) * distance_reward_weight
            # if distance_reward > 0.0 and action == 0:
            #     distance_reward += 0.5
            # else:
            #     distance_reward = 0.0
                # linear_reward = min(0.5, distance_reward * 0.5)

            # NEXT - Only action == 0

            # if abs(theta) < 0.1:
            #     theta_reward = 0.5
            # else:
            #     theta_penalty = abs(theta) * 0.01
            
            # Penalize roll and pitch outside the acceptable range
            # roll_penalty = max(0, abs(roll) - 0.1)
            # pitch_penalty = max(0, abs(pitch) - 0.1)
            
            # erratic_penalty = erratic_penalty_weight * (roll_penalty + pitch_penalty)

            # if np.isclose(distance, prev_distance, atol=0.01):
            #     stuck_penalty = 0.2  # Penalize lack of progress

            # if prev_theta < 0:
            #     theta_reward = theta - prev_theta
            # else:
            #     theta_reward = prev_theta - theta
            # theta_reward *= theta_reward_weight

            # Penalty for increasing distance from the goal
            # distance_penalty = max(0, (distance - prev_distance)) * distance_penalty_weight
            
            reward = distance_reward #- erratic_penalty #- distance_penalty #+ theta_reward - theta_penalty  # - stuck_penalty #- distance_penalty #- turning_penalty

        return reward, erratic_penalty

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

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('depth_camera_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            # '/fisheye_camera/image_raw',
            '/camera/image_raw',
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
            # '/imu_plugin/out',
            '/imu',
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

    counter = 0
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    odom_subscriber = OdomSubscriber()
    camera_subscriber = CameraSubscriber()
    imu_subscriber = IMUSubscriber()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(camera_subscriber)
    executor.add_node(imu_subscriber)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # Wait for subscribers to receive messages
    try:
        wait_for_subscribers([odom_subscriber, camera_subscriber, imu_subscriber], timeout=30)
    except TimeoutError as e:
        print(f"Error: {e}")
        sys.exit(1)

    rate = odom_subscriber.create_rate(2)

    try:
        # Initialize the environment, policy, and model outside the loop
        agent_env = CustomGazeboEnv()
        agent_model = RecurrentPPO(AgentLSTMPolicy, agent_env, verbose=2, tensorboard_log=TENSORBOARD_PATH, ent_coef=0.01)

        if TRAIN_PROBE:
            probe_env = CustomGazeboEnv()
            probe_model = RecurrentPPO(ProbeLSTMPolicy, probe_env, verbose=2, tensorboard_log=TENSORBOARD_PATH)

        checkpoint_callback = CheckpointCallback(save_freq=25000, save_path=MODEL_PATH, name_prefix=MODEL_NAME)

        while rclpy.ok():
            if TRAIN_PROBE:
                probe_model.learn(5e6, callback=checkpoint_callback)
                model_reward, std_reward = evaluate_policy(probe_model, probe_env, n_eval_episodes=1)
            else:
                agent_model.learn(5e6, callback=checkpoint_callback)
                model_reward, std_reward = evaluate_policy(agent_model, agent_env, n_eval_episodes=1)
            
            rate.sleep()
            break

    except KeyboardInterrupt:
        pass