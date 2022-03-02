import gym
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, GRU, TimeDistributed
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time
import heapq
import copy
# from FIELD import FIELD
# from PSO import PSO
# from utility import get_position, path_prediction

import configparser
config = configparser.ConfigParser()
config['DEFAULT'] = {'agent_num': '2',
                     'obs_num': '1',
                     'agent_vel': '5',
                     'obs_vel': '5',
                     'angle_bound': '90',
                     'obstacle_range': '20',
                     'v2v_range': '20',
                     'reward_0_if_done': 'False',
                     'ngh_in_state': 'True',
                     'obs_in_state': 'True',
                     'relative_position': 'False',
                     'average_state': 'True',
                     'v2v_comms': 'True',
                     'v2v_collision': 'True',
                     'id_concat': 'False',
                     'action_space': 'Continuous'}
with open('default.ini', 'w') as configfile:
    config.write(configfile)
from maca_env import *
from utilities import *
env = uav_swarm_env()

filename = 'maca_' + str(env.agent_num) + 'U' + str(env.obs_num) + 'O'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

entry, features, action_dim = env.entry, env.property, env.num_actions
angle_bound, state_layers, agent_num = env.angle_bound, env.layers, env.agent_num
state_size = (entry, features, state_layers)
action_size = (action_dim,)
swarm_state_size = (agent_num, entry, features, state_layers)
swarm_action_size = (agent_num, action_dim)
memory_depth = 4
memory_state_size = (memory_depth, entry, features, state_layers)
memory_action_size = (memory_depth, action_dim)
action_upper_bound = env.action_upper_bound
action_lower_bound = env.action_lower_bound
angle_upper_bound = env.angle_upper_bound
angle_lower_bound = env.angle_lower_bound
obstacle_safe_range = env.obstacle_range
vev_safe_range = env.v2v_range

class Buffer:
    def __init__(self, buffer_capacity, batch_size=10):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = {}
        self.action_buffer = {}
        self.reward_buffer = []
        self.next_state_buffer = {}
        self.done_buffer = []

        for i in range(agent_num):
            self.state_buffer[i] = []
            self.action_buffer[i] = []
            self.next_state_buffer[i] = []

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update_critic(
            self, state_batch_swarm, action_batch_swarm, reward_batch, next_state_batch_swarm, done_batch
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = []
            for i in range(agent_num):
                target_actions.append(target_actor(next_state_batch_swarm[i], training=True))

            target_critic_value = target_critic(
                [tf.transpose(next_state_batch_swarm, perm=[1, 0, 2, 3, 4]),
                 tf.squeeze(tf.transpose(tf.stack(target_actions), perm=[1, 0, 2]))], training=True
            )
            # done_batch_not = tf.math.logical_not(done_batch)
            # done_batch_not = tf.cast(done_batch_not, dtype=tf.float32)
            # target_critic_value *= done_batch_not

            y = reward_batch + gamma * target_critic_value

            critic_value = critic([tf.transpose(state_batch_swarm, perm=[1, 0, 2, 3, 4]),
                                   tf.transpose(action_batch_swarm, perm=[1, 0])], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic.trainable_variables)
        )

    @tf.function
    def update_actor_policy(
            self, state_batch_swarm, action_batch_swarm, reward_batch, next_state_batch_swarm,
    ):
        for i in range(agent_num):
            with tf.GradientTape() as tape:
                actions_update, actions_compare, states_update, states_compare = [], [], [], []
                for ii in range(agent_num):
                    actions_update.append(actor(state_batch_swarm[ii], training=True))
                    actions_compare.append(actor(state_batch_swarm[ii], training=True))
                    states_update.append(state_batch_swarm[ii])
                    states_compare.append(state_batch_swarm[ii])
                states_update_swarm = tf.stack(states_update)

                actions_update[i] = actor(states_update_swarm[i], training=True)
                critic_value = critic([tf.transpose(states_update_swarm, perm=[1, 0, 2, 3, 4]),
                                       tf.squeeze(tf.transpose(tf.stack(actions_update), perm=[1, 0, 2]))], training=True)
                actions_compare[i] *= 0
                states_compare[i] *= 0
                critic_baseline = critic([tf.transpose(tf.stack(states_compare), perm=[1, 0, 2, 3, 4]),
                                          tf.squeeze(tf.transpose(tf.stack(actions_compare), perm=[1, 0, 2]))], training=True)
                critic_advantage = critic_value - critic_baseline
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                actor_loss = -tf.math.reduce_mean(critic_advantage)

                # actions_update = actor(state_batch_swarm[i], training=True)
                # critic_value = critic([tf.transpose(state_batch_swarm, perm=[1, 0, 2, 3, 4]),
                #                        tf.expand_dims(actions_update, axis=-1)], training=True)
                # actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
            actor_optimizer_policy.apply_gradients(
                zip(actor_grad, actor.trainable_variables)
            )

    @tf.function
    def update_actor_expert(
            self, state_batch_swarm, action_batch_swarm, reward_batch, next_state_batch_swarm,
    ):
        action_batch_swarm = tf.cast(action_batch_swarm, tf.float32)
        for i in range(agent_num):
            with tf.GradientTape() as tape:
                actions = actor(state_batch_swarm[i], training=True)
                cost = actions - tf.expand_dims(action_batch_swarm[i], axis=-1)
                # Used `-value` as we want to maximize the value given
                # by the critic for our actions
                actor_loss = tf.math.reduce_mean(tf.math.square(cost))

            actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
            actor_optimizer_expert.apply_gradients(
                zip(actor_grad, actor.trainable_variables)
            )

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        # index = self.buffer_counter % self.buffer_capacity
        for i in range(agent_num):
            self.state_buffer[i].append(obs_tuple[0][i])
            self.action_buffer[i].append(obs_tuple[1][i])
            self.next_state_buffer[i].append(obs_tuple[3][i])
        self.reward_buffer.append(obs_tuple[2])
        self.done_buffer.append(obs_tuple[4])

        self.buffer_counter += 1

        # return index

    # We compute the loss and update parameters
    def learn(self, actor_flag):
        # Get sampling range
        record_range = np.min([self.buffer_counter, self.buffer_capacity])
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        # batch_indices = [-1]
        # batch_indices0 = np.array(heapq.nlargest(np.int(self.batch_size/2), range(len(self.R_episodic)),
        #                                          np.asarray(self.R_episodic).take))
        # batch_indices1 = np.random.choice(record_range, np.int(self.batch_size/2))
        # batch_indices = np.concatenate([batch_indices0, batch_indices1])

        # Sampling batches
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = {}, {}, [], {}, []
        for i in range(agent_num):
            state_batch[i] = []
            action_batch[i] = []
            next_state_batch[i] = []

        batch_len = len(batch_indices)
        for b in range(batch_len):
            index_temp = batch_indices[b]
            reward_batch.append(self.reward_buffer[index_temp])
            done_batch.append(self.done_buffer[index_temp])
            for i in range(agent_num):
                state_batch[i].append(self.state_buffer[i][index_temp])
                action_batch[i].append(self.action_buffer[i][index_temp])
                next_state_batch[i].append(self.next_state_buffer[i][index_temp])

        state_batch_swarm, action_batch_swarm, next_state_batch_swarm = [], [], []
        for i in range(agent_num):
            state_batch_swarm.append(tf.convert_to_tensor((state_batch[i])))
            action_batch_swarm.append(tf.convert_to_tensor((action_batch[i])))
            next_state_batch_swarm.append(tf.convert_to_tensor((next_state_batch[i])))

        state_batch_swarm = tf.stack(state_batch_swarm)
        action_batch_swarm = tf.stack(action_batch_swarm)
        next_state_batch_swarm = tf.stack(next_state_batch_swarm)

        # Convert to tensors
        reward_batch = tf.expand_dims(tf.convert_to_tensor(np.hstack(reward_batch)), axis=-1)
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        done_batch = tf.expand_dims(tf.convert_to_tensor(np.hstack(done_batch)), axis=-1)

        self.update_critic(state_batch_swarm, action_batch_swarm, reward_batch, next_state_batch_swarm, done_batch)
        if actor_flag == 'policy':
            self.update_actor_policy(state_batch_swarm, action_batch_swarm, reward_batch, next_state_batch_swarm)
        elif actor_flag == 'expert':
            self.update_actor_expert(state_batch_swarm, action_batch_swarm, reward_batch, next_state_batch_swarm)
        else:
            pass


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.00003, maxval=0.00003)

    inputs = Input(shape=state_size)
    out = Flatten()(inputs)
    out = Dense(256, activation="relu")(out)
    out = Dense(256, activation="relu")(out)
    outputs = Dense(1, activation="tanh", kernel_initializer=last_init)(out)
    #
    # outputs = Dense(1, activation="linear", kernel_initializer=last_init)(out)
    # outputs = tf.clip_by_value(outputs, clip_value_min=action_lower_bound, clip_value_max=action_upper_bound)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * action_upper_bound
    model = tf.keras.Model(inputs, outputs)

    return model


def get_critic():
    # State as input
    state_input = Input(shape=swarm_state_size)
    state_out = Flatten()(state_input)
    state_out = Dense(16, activation="relu")(state_out)
    state_out = Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = Input(shape=swarm_action_size)
    action_out = Flatten()(action_input)
    action_out = Dense(32, activation="relu")(action_out)

    # Both are passed through seperate layer before concatenating
    concat = Concatenate()([state_out, action_out])

    out = Dense(256, activation="relu")(concat)
    out = Dense(256, activation="relu")(out)
    outputs = Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state_var, ou_noise, decay_step):
    explore_probability = epsilon_min + (epsilon - epsilon_min) * np.exp(
        -epsilon_decay * decay_step)

    sampled_actions = tf.squeeze(actor(state_var))
    # legal_noise = ou_noise()
    legal_noise = np.random.normal(loc=0, scale=.1, size=sampled_actions.shape)
    # legal_noise = np.clip(legal_noise, action_lower_bound, action_upper_bound)
    # legal_noise = np.random.uniform(low=action_lower_bound*1, high=action_upper_bound*1,
    #                                 size=sampled_actions.shape)
    # if explore_probability > np.random.rand():
    #     noise_actions = sampled_actions * 0.5 + legal_noise * 0.5
    #     print('Noise action')
    # else:
    #     noise_actions = sampled_actions
    #     print('-----> Policy action')

    # We make sure action is within bounds
    noise_actions = sampled_actions * (1-explore_probability) + legal_noise * explore_probability
    # noise_actions = sampled_actions + legal_noise
    # print('explore_probability: {}'.format(explore_probability))
    legal_action = np.clip(noise_actions, action_lower_bound, action_upper_bound)

    return legal_action


actor = get_actor()
critic = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())

# Learning rate for actor-critic models
critic_lr = 3e-3
actor_lr_expert = 1e-3
actor_lr_policy = 1e-5

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer_expert = tf.keras.optimizers.Adam(actor_lr_expert)
actor_optimizer_policy = tf.keras.optimizers.Adam(actor_lr_policy)

total_episodes = 10_000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.01

buffer = Buffer(1000, 64)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

epsilon_min = 0.01
epsilon = 1
epsilon_decay = 1e-05

gamma1, gamma2 = .5, .5
maxiter, N = 5, 10
guard = env.obstacle_range + 10
field_size = env.size_env
obs_num = env.obs_num
# field = FIELD(agent_num, obs_num, guard, field_size)

create_time = time.ctime()
mon = create_time[4:7]
day = create_time[8:10]
hour = create_time[11:13]
minute = create_time[14:16]
test = False
if test:
    actor.load_weights('actor_' + filename + '.h5')
    # critic.load_weights("critic_Feb_0 _1425.h5")
    # target_actor.load_weights("target_actor_Feb_0 _1425.h5")
    # target_critic.load_weights("target_critic_Feb_0 _1425.h5")
behavior_clone = False
behavior_filename = 'expert_' + str(env.agent_num) + 'U' + str(env.obs_num) + 'O'
if behavior_clone:
    actor.load_weights('actor_' + behavior_filename + '.h5')
    # critic.load_weights('critic_' + behavior_filename + '.h5')
    target_actor.load_weights('actor_' + behavior_filename + '.h5')
    # target_critic.load_weights('target_critic_' + behavior_filename + '.h5')

policy_decay_step = 0
expert_decay_step = 0
# Takes about 4 min to train
# plt.figure(1)
# plt.xlabel("Episode")
# plt.ylabel("Avg. Epsiodic Reward")
# # plt.axis([0, total_episodes, 0, 100])
with open('ep_reward_list_' + filename + '.txt', 'w') as f:
    for ep in range(total_episodes):

        prev_state = env.reset()

        episodic_reward = 0
        step = 0

        done = False

        while not done:
            explore_probability = epsilon_min + (epsilon - epsilon_min) * np.exp(
                -epsilon_decay * expert_decay_step)
            img, img_global = env.render()
            # plt.figure(1001)
            # plt.imshow(img_global)
            # for plt_agent in range(agent_num):
            #     plt.plot(env.swarm[plt_agent].x_rec[-10:], env.swarm[plt_agent].y_rec[-10:], 'r')
            # plt.pause(0.01)
            # plt.clf()

            if not test:
                ou_noise = 0
                actionPolicy = policy(prev_state[:, 0, ...], ou_noise, policy_decay_step)
                action = actionPolicy
                expert_flag = False
            else:
                action_pred = tf.squeeze(actor(prev_state[:, 0, ...]))
                # swarm_x, swarm_y, swarm_vel, swarm_angle_rec, obs_x, obs_y = get_swarm(env)
                # action, emergency_flag, obs_flag = hyper_guarantee(env, action_pred, obs_x, obs_y, swarm_x, swarm_y,
                #                                                    swarm_vel)
                action = action_pred

                expert_flag = False
                # print('Testing ...')

            actionPre = action

            # Recieve state and reward from environment.
            emergency_flag = np.zeros(agent_num, dtype=bool)
            # if agent_num == 1:
            #     action = [action]
            # else:
            #     pass
            state, reward, done, info, local_reward = env.step(action, step, emergency_flag)

            buffer.record((prev_state[:, 0, ...], action, reward, state[:, 0, ...], done))
            episodic_reward += reward

            # End this episode when `done` is True
            # if done:
            #     break

            prev_state = state

            print(filename)
            print("Episode {} step {} time {}".format(ep, step, time.ctime()))
            # print("action: ")
            # print(action)
            # print("reward: ")
            # print(reward)

            step += 1

            # if cv.waitKey(25) & 0xFF == ord('q'):
            #     cv.destroyWindow()
            #     break

            expert_decay_step += 1

            if not expert_flag:
                policy_decay_step += 1

        if test:
            pass
        else:
            if ep >= 10:
                if ep % 3 == 0:
                    if expert_flag:
                        actor_flag = 'expert'
                    else:
                        actor_flag = 'policy'
                else:
                    actor_flag = 'NA'
                buffer.learn(actor_flag)
                update_target(target_actor.variables, actor.variables, tau)
                update_target(target_critic.variables, critic.variables, tau)
            # # Save the weights
            # actor.save_weights('actor_' + filename + '.h5')
            # critic.save_weights('critic_' + filename + '.h5')
            #
            # target_actor.save_weights('target_actor_' + filename + '.h5')
            # target_critic.save_weights('target_critic_' + filename + '.h5')

        ep_reward_list.append(episodic_reward)
        f.write(str(episodic_reward) + '\n')

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        # print("Episode * {} * action: {} Avg Reward is ==> {}".format(ep, action, avg_reward))
        avg_reward_list.append(avg_reward)

        # plt.figure(1)
        # plt.plot(np.arange(0, ep+1), ep_reward_list, 'r')
        # plt.plot(np.arange(0, ep+1), avg_reward_list, 'b')
        # plt.pause(0.01)

        # Save the weights
        actor.save_weights('actor_' + filename + '.h5')
        critic.save_weights('critic_' + filename + '.h5')

        target_actor.save_weights('target_actor_' + filename + '.h5')
        target_critic.save_weights('target_critic_' + filename + '.h5')

