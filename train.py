import tensorflow as tf
import numpy as np
import pandas as pd
import psutil
import itertools
import random
from typing import Tuple
import scipy.ndimage
import scipy.signal
import time
import os
import gym
from gym import spaces
from gym.spaces import Box, Discrete
import math
import imageio
import joblib

import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as gf1d
from scipy.signal import lfilter

from IPython.display import clear_output
import tensorflow.contrib.slim as slim
from spinup.exercises.common import print_result
from spinup import ppo
from spinup.utils.logx import restore_tf_graph

learning_rate = 0.0005
gamma = 0.99
num_rows = 8
num_cols = 8
num_states = 2
num_channels = 4

to_train = False
exp_id = 1544401745

# Grid World Environment 

# One hot encoding (background, R, G, B, an agent)
def encode(x):
    return np.array([[0,0,0,0],
                     [0,0,1,0],
                     [0,1,0,0],
                     [1,0,0,0],
                     [1,1,1,1]], dtype='float32')[x]

# Commands
class Walk:
    GO = 1      # 5
    AVOID = 2   # 6

# Colors
class Color:
    BLACK = 0
    GREEN = 1
    RED = 2
    BLUE = 3
    WHITE = 4

# Objects
class Obj:
    BLACK = 0
    TRIANGLE = 1 # 7
    SQUARE = 2   # 8
    CIRCLE = 3   # 9
    WHITE = 4




def construct_cmd(train=True):
    walk = random.choice([Walk.GO, Walk.AVOID])
    color = random.choice([Color.GREEN, Color.RED, Color.BLUE])
    obj = random.choice([Obj.TRIANGLE, Obj.SQUARE, Obj.CIRCLE])

    if train: # Removing the  ‘Blue’ color  and ‘Circle’ object from the training 
        if color == Color.BLUE and obj == Obj.CIRCLE:
            return construct_cmd(train)

    return [walk, color, obj]

# Translating commands to adjust to object-color pairs

def translate_cmd(target):
    walk = 5 if target[0] == Walk.GO else 6
    if target[2] == 1:
        object = 7
    elif target[2] == 2:
        object = 8
    elif target[2] == 3:
        object = 9
    return [walk, target[1], object]

class GoalGridWorld(gym.Env):
    DX = [-1, 0, +1, 0] #The way an agent moves on the ‘x’ axis 
    DY = [0, +1, 0, -1] #The way an agent moves on the ‘y’ axis 

    EYE2 = np.eye(2) 
    EYE3 = np.eye(3)

    # Reward and penalize to adjust an agent’s behavior 

    STEP_REWARD = -1
    CORRECT_TARGET_REWARD = 100
    INCORRECT_TARGET_REWARD = -36
    AVOID_TARGET_REWARD = 10
    BOUNDARY_REWARD = -100
    NUM_STEPS_EXCEEDED_REWARD = 0

    # Number of columns and rows of the gridworld environment an agent navigates in

    def __init__(self,
                 num_rows = 8,
                 num_cols = 8):

        self.action_space_size = 4 # up, right, down, left
        self.max_steps = 100  
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.size = num_rows, num_cols
        self.state = np.zeros(self.size, dtype='int32')
        self.object_state = np.zeros(self.size, dtype='int32')
        self.object_color_map = {}

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(520,), dtype=np.float32)

    def set_objects(self, target):
        # Ensure that target color-object pairs exist in the environment 

        self.object_color_map[target[1]] = target[2] 
        colors = [Color.GREEN, Color.RED, Color.BLUE]
        colors.remove(target[1])

        objects = [Obj.TRIANGLE, Obj.SQUARE, Obj.CIRCLE]
        objects.remove(target[2])

        # Assign to each color a random object that is not from the target

        self.object_color_map[colors[0]] = random.choice([objects[0], objects[1]])
        self.object_color_map[colors[1]] = random.choice([objects[0], objects[1]])

    # Simplifying the environment 

    def get_obs(self):
        
        walk_cmd = self.EYE2[self.target[0]-1]
        color_cmd = self.EYE3[self.target[1]-1]
        obj_cmd = self.EYE3[self.target[2]-1]
        
        # Concatenating colors state, object state, commands to simplify the environment

        cmd = np.concatenate([walk_cmd, color_cmd, obj_cmd])
        a = np.concatenate([encode(self.state).flat, encode(self.object_state).flat, cmd]) # reshape((1,1,520))
        return a 

    # Reset the environment 

    def reset(self, target=None, train=True): # [Walk, Color, Object]
        if target is None:
            target = construct_cmd(train=train)
        assert target[0] is Walk.GO or target[0] is Walk.AVOID
        assert target[1] is Color.GREEN or target[1] is Color.RED or target[1] is Color.BLUE
        assert target[2] is Obj.TRIANGLE or target[2] is Obj.SQUARE or target[2] is Obj.CIRCLE

        self.target = target
        self.count_steps = 0
        self.reward = 0
        self.done = False

        self.state[...] = 0
        self.object_state[...] = 0
        all_points = [(i, j) for i in range(self.num_rows)
                             for j in range(self.num_cols)]

        green_pos, red_pos, blue_pos, player_pos = random.sample(all_points, k=4)

    
        self.state[green_pos] = Color.GREEN
        self.state[red_pos] = Color.RED
        self.state[blue_pos] = Color.BLUE
        self.state[player_pos] = Color.WHITE

        self.set_objects(target)
        self.object_state[green_pos] = self.object_color_map[Color.GREEN]
        self.object_state[red_pos] = self.object_color_map[Color.RED]
        self.object_state[blue_pos] = self.object_color_map[Color.BLUE]
        self.object_state[player_pos] = Obj.WHITE

        self.position = player_pos #Agent's location
        return self.get_obs()


    def offset(self, point, direction):
        x, y = point
        return x + self.DX[direction], y + self.DY[direction]

    # Check to see if the agent is outside or inside the board

    def outside(self, point):
        x, y = point
        if x < 0 or x >= self.num_rows:
            return True
        if y < 0 or y >= self.num_cols:
            return True
        return False

    def step(self, action):
        if self.done:
            raise ValueError('Game is over. Environment is frozen.')

        if action not in range(self.action_space_size):
            raise ValueError('The action is not included in the action space.')

        self.state[self.position] = Color.BLACK
        self.position = self.offset(self.position, action)
        self.count_steps += 1
        self.reward += self.STEP_REWARD  

        # Making sure the agent will not go outside the board

        if self.outside(self.position):
            if action == 0:
                new_action = 2
            if action == 1:
                new_action = 3
            if action == 2: 
                new_action = 0
            if action == 3:
                new_action = 1
            self.position = self.offset(self.position, new_action)

        
        #   Reward conditions
         
        elif self.state[self.position] == self.target[1] \
        and self.target[0] == Walk.GO \
        and self.object_state[self.position] == self.target[2]: #  If the object is the same e.g. Circle
            '''Agent hits the target'''
            self.reward += self.CORRECT_TARGET_REWARD
            self.done = True
        elif self.target[0] == Walk.AVOID  \
        and self.state[self.position] == self.target[1] \
        and self.object_state[self.position] == self.target[2]:
            ''''Agent hits the target that should be avoided'''
            self.reward += self.AVOID_TARGET_REWARD
            self.done = True
        elif self.count_steps > self.max_steps:
            '''Agent exceeded max number of steps'''
            self.reward += self.NUM_STEPS_EXCEEDED_REWARD
            self.done = True
        elif self.state[self.position] != Color.BLACK:
            '''Agent hits incorrect target. Game continues'''
            self.reward += self.INCORRECT_TARGET_REWARD

        if not self.done:
            ''''Put the agent back onto the board to continue the game'''
            self.state[self.position] = Color.WHITE

        return self.get_obs(), self.reward if self.done else 0, self.done, None

# Multilayer perceptron 

# The code for mlp, mlp_categorical, mlp_actor_critic is taken from OpenAI Spinning Up https://github.com/openai/spinningup

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):

    with tf.variable_scope('ob'):
        x = slim.flatten(x)
        x = slim.fully_connected(x, 128, activation_fn=activation)
        x = slim.fully_connected(x, 128, activation_fn=activation)

    logits = slim.fully_connected(x, hidden_sizes[-1], activation_fn=output_activation)
    return logits

"""
Policy 
"""

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh, 
                     output_activation=None, policy=None, action_space=None):

    # Default policy builder depends on action space

    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy
    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, v

class Color:
    BLACK = 0
    GREEN = 1
    RED = 2
    BLUE = 3
    WHITE = 4

    # Rendering 
    
def render(state, images, choice=0): 

    # Colors
    image = np.pad(state, ((1,1), (1,1)), 'constant')
    image = scipy.ndimage.zoom(image, 64, order=0, mode='reflect') 
    image = np.expand_dims(image, -1) 
    image = np.repeat(image, 3, axis=2) 
    
    # Set green
    (image[:,:,0])[image[:,:,0]==1] = 0
    (image[:,:,2])[image[:,:,2]==1] = 0
    
    # Set red
    (image[:,:,1])[image[:,:,0]==2] = 0
    (image[:,:,2])[image[:,:,2]==2] = 0
    
    # Set blue
    (image[:,:,0])[image[:,:,0]==3] = 0
    (image[:,:,1])[image[:,:,1]==3] = 0
    
    images.append(image)
    return images

def simulate(path, color_output="color_state.gif", object_output="object_state.gif"):
    sess = tf.Session()
    graph = restore_tf_graph(sess, path)
    env = GoalGridWorld()

    state = env.reset(train=False)
    running = True
    count = 0
    color_images, object_images = [], []
    
    while running:
        a, _ = sess.run([graph['pi'], graph['v']], feed_dict={graph['x']: state.reshape(1,-1)})
        state, reward, done, _ = env.step(a[0])

        color_images = render(env.state, color_images)
        object_images = render(env.object_state, object_images)
        running = not done

        count += 1 
        if count > 100:
            break

    save_gif(color_images, path=color_output)
    save_gif(object_images, path=object_output)
    print("____________________________")
    print("Target: {}".format(env.target))
    print("Reward: {}".format(reward))
    print("____________________________")

def save_gif(images, path="example.gif"):
    with imageio.get_writer(path, mode='I') as writer:
        for image in images:
            writer.append_data(image)

if __name__ == '__main__':
    """
    Run the code to verify the solution
    """
    if to_train:
        logdir = "data/experiments/%i"%int(time.time())
        ppo(env_fn = GoalGridWorld,
            actor_critic=mlp_actor_critic,
            steps_per_epoch=100000, epochs=100, logger_kwargs=dict(output_dir=logdir))
    else:
        logdir = "data/experiments/%i/simple_save/"%int(exp_id)
        simulate(path=logdir)
