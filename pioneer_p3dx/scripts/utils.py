#!/usr/bin/env python3

import numpy as np

ENV_SETUP = 2
GOAL_REACHED_DIST = 1.0 # 0.25
COLLISION_DIST = 0.35   # DO NOT CHANGE
HOME_REACHED_DIST = 0.3
TIME_DELTA = 0.05       # OG - 0.2
MIN_DISTANCE = 0.8
TRAIN_PROBE = False
MODEL_NAME = f"agent_{ENV_SETUP}"
RNN_STATE_PATH = "/home/dipam/blind_ws/src/pioneer_p3dx/logs/eval"
MODEL_PATH = "/home/dipam/blind_ws/src/pioneer_p3dx/logs/models"
TENSORBOARD_PATH = "/home/dipam/blind_ws/src/pioneer_p3dx/logs/tensorboard"
REWARDS_PATH = "/home/dipam/blind_ws/src/pioneer_p3dx/logs/rewards"

# Map of object names to their sizes (length, width, height)
obstacle_sizes = {
    "Wall_11": (2.5, 0.15, 1),
    "Wall_12": (2.5, 0.15, 1),
    # "Wall_24": (1.97704, 0.15, 1),
    # "Wall_24_2": (1.97704, 0.15, 1),
    # "Wall_28": (1.75, 0.15, 1),
    "cardboard_box_0": (0.5, 0.4, 0.3),
    "cardboard_box_1": (0.5, 0.4, 0.3),
    "cardboard_box_2": (0.5, 0.4, 0.3),
    "cardboard_box_3": (0.5, 0.4, 0.3),
    "bookshelf_0": (0.9, 0.4, 1.2),
    "bookshelf_1": (0.9, 0.4, 1.2),
    "bookshelf_2": (0.9, 0.4, 1.2),
    "bookshelf_3": (0.9, 0.4, 1.2),
    "table_0": (1.5, 0.8, 0.03),
    "table_1": (1.5, 0.8, 0.03),
    "fire_hydrant_0": (0.3, 0.3, 0.6),
    "fire_hydrant_1": (0.3, 0.3, 0.6),
    "goal": (1.0, 1.0, 0.01),
    "home": (0.1, 0.1, 0.01),
    "pioneer": (0.3, 0.3, 0.3),
    "turtlebot3": (0.3, 0.3, 0.3),
}

# def check_position(x, y, size, identifier, diff, unavailable_positions):
def check_position(x, y, size, identifier, unavailable_positions):
    obj_length, obj_width, _ = size

    for key, value in unavailable_positions.items():
        obj_x, obj_y = value
        distance_x = abs(obj_x - x)
        distance_y = abs(obj_y - y)

        # Calculate distance and check if there is an overlap
        # if diff == 0:
        #     diff = 1
        # if distance_x < (obj_length/2 + size[0]/2 + (MIN_DISTANCE/diff)) and distance_y < (obj_width/2 + size[1]/2 + (MIN_DISTANCE/diff)):
        
        if distance_x < (obj_length/2 + size[0]/2 + MIN_DISTANCE) and distance_y < (obj_width/2 + size[1]/2 + MIN_DISTANCE):
            # If the checking position is for the same object, allow to update its position.
            if key == identifier:
                continue
            return False
        # # Check if the suggested (x, y) is too close to the object location
        # corner1 = (obj_pose[0] - obj_length / 2, obj_pose[1] - obj_width / 2)
        # corner2 = (obj_pose[0] + obj_length / 2, obj_pose[1] + obj_width / 2)
        # if (corner1[0] - MIN_DISTANCE <= x <= corner2[0] + MIN_DISTANCE) and \
        # (corner1[1] - MIN_DISTANCE <= y <= corner2[1] + MIN_DISTANCE):
    
    # Update or add the position for the given identifier
    unavailable_positions[identifier] = (x, y)

    return True