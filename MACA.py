import gym
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
import numpy as np
import pylab
import pickle
from scipy import signal
import time
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions

import configparser
config = configparser.ConfigParser()
config['DEFAULT'] = {'agent_num': '3',
                     'obs_num': '1',
                     'agent_vel': '5',
                     'obs_vel': '5',
                     'angle_bound': '45',
                     'reward_0_if_done': 'True',
                     'ngh_in_state': 'True',
                     'obs_in_state': 'True',
                     'relative_position': 'True',
                     'v2v_comms': 'True',
                     'v2v_collision': 'True',
                     'id_concat': 'False',
                     'action_space': 'Continuous'}
with open('default.ini', 'w') as configfile:
    config.write(configfile)
# from uav_swarm_env_soft_random_obs import *
# from uav_swarm_env_persistence import *
from MACA_Env import *
from EAS import *
env = uav_swarm_env()

state_size = (env.entry, env.property, env.layers)
action_size = (env.num_actions,)
swarm_state_size = (env.agent_num, env.entry, env.property, env.layers)
swarm_action_size = (env.agent_num, env.num_actions)

action_upper_bound = env.action_upper_bound
action_lower_bound = env.action_lower_bound
angle_upper_bound = env.angle_upper_bound
angle_lower_bound = env.angle_lower_bound
obstacle_range = env.obstacle_range
vev_range = env.v2v_range

script_name = 'Null_' + str(env.agent_num) + 'A' + str(env.obs_num) + 'O_' + str(angle_upper_bound)

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

class Buffer:
    def __init__(self, buffer_capacity=10000, batch_size=64):
        
        self.buffer_capacity = buffer_capacity
       
        self.batch_size = batch_size

        self.buffer_counter = 0
        self.buffer_index = 0

        self.state_buffer = np.zeros((env.agent_num,) + (self.buffer_capacity,) + state_size)
        self.action_buffer = np.zeros((env.agent_num,) + (self.buffer_capacity, env.num_actions))
        self.local_reward_buffer = np.zeros((env.agent_num,) + (self.buffer_capacity, 1))
        self.global_reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.self_perception_buffer = np.zeros((env.agent_num,) + (self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((env.agent_num,) + (self.buffer_capacity,) + state_size)
        self.done_buffer = np.zeros((self.buffer_capacity, 1), dtype=bool)

    def record(self, obs_tuple):
        self.buffer_index = self.buffer_counter % self.buffer_capacity
        index = self.buffer_index
        self.state_buffer[:, index] = np.squeeze(obs_tuple[0])
        self.action_buffer[:, index] = obs_tuple[1]
        self.local_reward_buffer[:, index] = obs_tuple[2]
        self.global_reward_buffer[index, :] = obs_tuple[3]
        self.self_perception_buffer[:, index] = obs_tuple[4]
        self.next_state_buffer[:, index] = np.squeeze(obs_tuple[5])
        self.done_buffer[index, :] = obs_tuple[6]

        self.buffer_counter += 1

    @tf.function
    def update_critic(
        self, state_batch, action_batch, local_reward_batch, global_reward_batch, self_perception_batch, next_state_batch, done_batch):
        with tf.GradientTape() as tape1:
            tape1.watch(Critic1.trainable_variables)
            actions_ = []
            for i in range(env.agent_num):
                actions_.append(tf.math.atan(target_Actor[i](next_state_batch[i])))
                # actions_.append(target_Actor[i](next_state_batch[i]))
            critic_ = target_Critic1([tf.transpose(next_state_batch, perm=[1, 0, 2, 3, 4]),
                                             tf.transpose(tf.stack(actions_), perm=[1, 0, 2])],
                                            training=True)
            done_batch_not = tf.math.logical_not(done_batch)
            done_batch_not = tf.cast(done_batch_not, dtype=tf.float32)
            critic_ *= done_batch_not

            critic = Critic1([tf.transpose(state_batch, perm=[1, 0, 2, 3, 4]),
                                     tf.transpose(action_batch, perm=[1, 0, 2])], training=True)
            credit = Credit([tf.transpose(state_batch, perm=[1, 0, 2, 3, 4]),
                                     tf.transpose(action_batch, perm=[1, 0, 2])], training=True)

            y1 = global_reward_batch + gamma * critic_
            critic_loss1 = huber_loss(y1, critic)

        critic_grad = tape1.gradient(critic_loss1, Critic1.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, Critic1.trainable_variables)
        )

    @tf.function
    def update_actor(
            self, state_batch, action_batch, local_reward_batch, global_reward_batch, self_perception_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        local_reward_batch = tf.transpose(tf.squeeze(local_reward_batch), perm=[1, 0])
        actions_n_orig = []
        for i in range(env.agent_num):
            actions_n_orig.append(tf.cast(tf.math.atan(Actor[i](state_batch[i], training=True)), dtype=tf.float64))
            # actions_n_orig.append(
            #     tf.cast(Actor[i](state_batch[i], training=True), dtype=tf.float64))

        critic_subs = []
        swarm_ind = set(np.arange(env.agent_num))
        # noise_rand_acts = tfd.Normal(loc=0., scale=1)
        for ego in range(env.agent_num):
            # action_subs = tf.math.atan(Actor[ego](state_batch[ego], training=True)) * 2 / np.pi
            ego_ind = set([ego])
            sub_ind = list(swarm_ind - ego_ind)[0]
            action_ego = action_batch[ego]
            state_ego = state_batch[ego]
            action_subs = action_batch[sub_ind]
            state_subs = state_batch[sub_ind]
            # state_subs = state_subs.numpy()
            # state_subs[:, 0, 1, 0] = 0
            # state_subs = tf.convert_to_tensor(state_subs)

            # rand_act = tf.cast(noise_rand_acts.sample(action_ego.shape), dtype=tf.float64)
            states_subs, actions_subs = [], []
            for j in range(env.agent_num):
                if j == ego:
                    actions_subs.append(action_ego * 0)
                    states_subs.append(state_ego * 0)
                else:
                    actions_subs.append(action_batch[j] * 1)
                    states_subs.append(state_batch[j] * 1)

            critic_subs.append(Critic1([tf.transpose(tf.stack(states_subs),
                                                          perm=[1, 0, 2, 3, 4]),
                                             tf.transpose(tf.stack(actions_subs),
                                                          perm=[1, 0, 2])]))

        for i in range(env.agent_num):
            actions_n = actions_n_orig
            with tf.GradientTape() as tape:
                action_ego = tf.cast(tf.math.atan(Actor[i](state_batch[i], training=True)), tf.float64)
                # action_ego = tf.cast(Actor[i](state_batch[i], training=True), tf.float64)

                actions_n[i] = action_ego
                critic = Critic1([tf.transpose(state_batch, perm=[1, 0, 2, 3, 4]),
                                          tf.transpose(tf.stack(actions_n), perm=[1, 0, 2])],
                                         training=True)

                counterfactual_advantage = critic - critic_subs[i]
                # noise = tfd.Normal(loc=0., scale=.1)
                # counterfactual_advantage += noise.sample(1)
                # counterfactual_advantage = tf.minimum(counterfactual_advantage, critic)
                # counterfactual_advantage = tf.maximum(counterfactual_advantage, 0)
                actor_loss = -tf.math.reduce_mean(counterfactual_advantage)
                # actor_loss = -tf.math.reduce_mean(critic * self_perception_batch_uni_mod[:,i])
                # actor_loss = -tf.math.reduce_mean(critic_n * tf.expand_dims(reward_ratio[:, i], axis=1))
                # credit_loss += (critic/(critic_n + 1e-8) - tf.expand_dims(credit[:, i], axis=1))**2
                # credit_loss += huber_loss(critic/(critic_n + 1e-8), tf.expand_dims(credit[:, i], axis=1))
                # credit_loss += actor_loss

            actor_grad = tape.gradient(actor_loss, Actor[i].trainable_variables)
            actor_optimizer.apply_gradients(
                zip(actor_grad, Actor[i].trainable_variables)
            )

            # credit_loss = tf.math.reduce_mean(tf.sqrt(credit_loss))

    def learn(self, update_flag):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)
        # batch_indices = np.random.choice(np.arrange(record_range), self.batch_size, p=priority)

        state_batch = tf.convert_to_tensor(self.state_buffer[:, batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[:, batch_indices])
        local_reward_batch = tf.cast(tf.convert_to_tensor(self.local_reward_buffer[:, batch_indices]), dtype=tf.float32)
        global_reward_batch = tf.cast(tf.convert_to_tensor(self.global_reward_buffer[batch_indices, :]), dtype=tf.float32)
        self_perception_batch = tf.cast(tf.convert_to_tensor(self.self_perception_buffer[:, batch_indices]), dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[:, batch_indices])
        done_batch = tf.cast(tf.convert_to_tensor(self.done_buffer[batch_indices, :]), dtype=tf.bool)

        if update_flag == 'critic':
            # print('critic update. ')
            self.update_critic(state_batch, action_batch, local_reward_batch, global_reward_batch,
                               self_perception_batch, next_state_batch, done_batch)
        elif update_flag == 'actor':
            # print('actor update. ')
            self.update_actor(state_batch, action_batch, local_reward_batch, global_reward_batch,
                              self_perception_batch, next_state_batch)
        else:
            print('no network to update. ')

@tf.function
def update_target(target_weights, weights, var_tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * var_tau + a * (1 - var_tau))

def actor_network(input_shape, optimizer):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003)

    state_input = Input(input_shape)
    state_x = Flatten()(state_input)

    x = Dense(256, activation="relu")(state_x)
    x = Dense(256, activation="relu")(x)
    throttle = Dense(1, activation="tanh", kernel_initializer=last_init)(x)
    angle = Dense(1, activation="linear", kernel_initializer=last_init)(x)

    # throttle = tf.multiply(throttle, throttle_up_bound)
    # angle = tf.multiply(angle, angle_up_bound)

    model = Model(inputs=state_input, outputs=angle)
    model.compile(loss='mse', optimizer=optimizer)

    return model

def critic_network(input_shape, action_shape, agent_num, optimizer):
    state_input = Input(input_shape)
    state_x = Flatten()(state_input)
    state_x = Dense(32, activation="relu")(state_x)

    action_input = Input(action_shape)
    action_x = Flatten()(action_input)
    action_x = Dense(32, activation="relu")(action_x)

    out_x = Concatenate()([state_x, action_x])

    out_x = Dense(256, activation="relu")(out_x)
    out_x = Dense(256, activation="relu")(out_x)
    value = Dense(1, kernel_initializer='he_uniform')(out_x)
    credit = Dense(agent_num, activation="softmax")(out_x)
    value_weight = value * credit

    model = Model(inputs=[state_input, action_input], outputs=[value])
    model.compile(loss='mse', optimizer=optimizer)

    return model

def credit_network(input_shape, action_shape, agent_num, optimizer):
    state_input = Input(input_shape)
    state_x = Flatten()(state_input)
    state_x = Dense(32, activation="relu")(state_x)

    action_input = Input(action_shape)
    action_x = Flatten()(action_input)
    action_x = Dense(32, activation="relu")(action_x)

    out_x = Concatenate()([state_x, action_x])
    # #
    # out_x = state_x

    out_x = Dense(256, activation="relu")(out_x)
    out_x = Dense(256, activation="relu")(out_x)
    value = Dense(agent_num, activation="softmax")(out_x)

    model = Model(inputs=[state_input, action_input], outputs=value)
    model.compile(loss='mse', optimizer=optimizer)

    return model

def policy(var_state, decay_step, evaluate):
    explore_probability = epsilon_min + (epsilon - epsilon_min) * np.exp(
        -epsilon_decay * decay_step)
    if env.agent_num == 1:
        sampled_actions = Actor[0](var_state[0])
    else:
        sampled_actions = np.zeros([env.agent_num, 1])
        for i in range(env.agent_num):
            sampled_actions[i] = Actor[i](np.stack(var_state[i]))
    sampled_actions = np.arctan(sampled_actions)

    if not evaluate:
        noise = np.random.normal(loc=0.0, scale=1, size=sampled_actions.shape)
        noise_actions = sampled_actions + noise * explore_probability
    else:
        noise_actions = sampled_actions
    # if explore_probability > np.random.rand():
    #     noise_actions = sampled_actions + noise
    # else:
    #     noise_actions = sampled_actions

    legal_action = np.clip(noise_actions, action_lower_bound, action_upper_bound)

    return legal_action

# std_dev = 0.1
# ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

critic_lr = 0.001
actor_lr = 0.001
credit_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
credit_optimizer = tf.keras.optimizers.Adam(credit_lr)

Actor, target_Actor, local_Critic, local_target_Critic = [], [], [], []
for agent in range(env.agent_num):
    Actor.append(actor_network(input_shape=state_size, optimizer=actor_optimizer))
    target_Actor.append(actor_network(input_shape=state_size, optimizer=actor_optimizer))
    # local_Critic.append(critic_network(input_shape=state_size, action_shape=action_size,
    #                                    agent_num=1, optimizer=critic_optimizer))
    # local_target_Critic.append(critic_network(input_shape=state_size,
    #                                           action_shape=action_size,
    #                                           agent_num=1, optimizer=critic_optimizer))
Critic1 = critic_network(input_shape=swarm_state_size, action_shape=swarm_action_size,
                         agent_num=env.agent_num, optimizer=critic_optimizer)
target_Critic1 = critic_network(input_shape=swarm_state_size, action_shape=swarm_action_size,
                                agent_num=env.agent_num, optimizer=critic_optimizer)
Credit = credit_network(input_shape=swarm_state_size, action_shape=swarm_action_size,
                        agent_num=env.agent_num, optimizer=credit_optimizer)

for agent in range(env.agent_num):
    target_Actor[agent].set_weights(Actor[agent].get_weights())
    # local_target_Critic[agent].set_weights(local_Critic[agent].get_weights())
target_Critic1.set_weights(Critic1.get_weights())

total_episodes = 100
gamma = 0.99
tau = 0.005

buffer = Buffer(10000, 64)
# with open('Buffer.txt', 'rb') as f:
#     buffer = pickle.load(f)

pylab.figure(1, figsize=(8, 5))
scores, episodes, averages = [], [], []
def plot_model(score, episode):
    scores.append(score)
    episodes.append(episode)
    average = sum(scores[-50:]) / len(scores[-50:])
    averages.append(average)
    pylab.plot(episodes, averages, 'r')
    pylab.plot(episodes, scores, 'g')
    pylab.xlabel('Steps')
    pylab.ylabel('Score')
    try:
        pylab.savefig('test_VDN.png')
    except OSError:
        pass
    return average

ep_reward_list = []
avg_reward_list = []

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid1 = cv2.VideoWriter('DQNRun.mp4', fourcc, 10, (300, 300))
file = open('DDPG_results_scores_averages' + script_name + '.txt', 'w')
file.write('scores ' + ' episodes ' + ' average ' + "\n")
# Critic1 = load_model('imitation_critic1.h5')
# Actor = load_model('imitation_actor.h5')
epsilon_min = 0.01
epsilon = 1
epsilon_decay = 1e-04
decay_step = 0
best_score = 0
evaluate = True
online_lr = False
continue_train = False
cloned_behavior = False

if not evaluate and online_lr:
    raise Exception('You have to enable evaluate first and then yse online learning! ')

if evaluate:
    if not online_lr:
        for agent in range(env.agent_num):
            Actor[agent] = load_model('DDPG_Actor' + str(agent) + script_name + '.h5')
    else:
        for agent in range(env.agent_num):
            Actor[agent] = load_model('DDPG_Actor' + str(agent) + script_name + '.h5')
            target_Actor[agent] = load_model('DDPG_Target_Actor' + str(agent) + script_name + '.h5')
        Critic1 = load_model('DDPG_Critic' + script_name + '.h5')
        target_Critic1 = load_model('DDPG_Target_Critic' + script_name + '.h5')

if continue_train:
    for agent in range(env.agent_num):
        Actor[agent] = load_model('DDPG_Actor' + str(agent) + script_name + '.h5')
        target_Actor[agent] = load_model('DDPG_Target_Actor' + str(agent) + script_name + '.h5')
    Critic1 = load_model('DDPG_Critic' + script_name + '.h5')
    target_Critic1 = load_model('DDPG_Target_Critic' + script_name + '.h5')

if cloned_behavior:
    # for agent in range(env.agent_num):
    #     Actor[agent] = load_model('test_Imitation_Learning_Actor' + str(agent) + '.h5')
    #     target_Actor[agent] = load_model('test_Imitation_Learning_target_Actor' + str(agent) + '.h5')
    Critic1 = load_model('test_Imitation_Learning_Critic.h5')
    target_Critic1 = load_model('test_Imitation_Learning_target_Critic.h5')

num_success, num_failure, avoid_times = 0, 0, 0
buffer_counter_success_all = 0
combine2buffers = False
for ep in range(total_episodes):

    if env.agent_num == 2:
        fileA0 = open('Trajectory_' + script_name + str(ep) + '_agent0.txt', 'w')
        fileA0.write('x ' + ' y ' + "\n")
        fileA1 = open('Trajectory_' + script_name + str(ep) + '_agent1.txt', 'w')
        fileA1.write('x ' + ' y ' + "\n")
    elif env.agent_num == 3:
        fileA0 = open('Trajectory_' + script_name + str(ep) + '_agent0.txt', 'w')
        fileA0.write('x ' + ' y ' + "\n")
        fileA1 = open('Trajectory_' + script_name + str(ep) + '_agent1.txt', 'w')
        fileA1.write('x ' + ' y ' + "\n")
        fileA2 = open('Trajectory_' + script_name + str(ep) + '_agent2.txt', 'w')
        fileA2.write('x ' + ' y ' + "\n")
    elif env.agent_num == 4:
        fileA0 = open('Trajectory_' + script_name + str(ep) + '_agent0.txt', 'w')
        fileA0.write('x ' + ' y ' + "\n")
        fileA1 = open('Trajectory_' + script_name + str(ep) + '_agent1.txt', 'w')
        fileA1.write('x ' + ' y ' + "\n")
        fileA2 = open('Trajectory_' + script_name + str(ep) + '_agent2.txt', 'w')
        fileA2.write('x ' + ' y ' + "\n")
        fileA3 = open('Trajectory_' + script_name + str(ep) + '_agent3.txt', 'w')
        fileA3.write('x ' + ' y ' + "\n")

    if env.obs_num == 1:
        fileO0 = open('Trajectory_' + script_name + str(ep) + '_obs0.txt', 'w')
        fileO0.write('x ' + ' y ' + "\n")
    elif env.obs_num == 2:
        fileO0 = open('Trajectory_' + script_name + str(ep) + '_obs0.txt', 'w')
        fileO0.write('x ' + ' y ' + "\n")
        fileO1 = open('Trajectory_' + script_name + str(ep) + '_obs1.txt', 'w')
        fileO1.write('x ' + ' y ' + "\n")

    fileT = open('Time_' + script_name + str(ep) + '.txt', 'w')
    fileT.write('time ' + "\n")

    state = env.reset()
    episodic_reward = 0
    step = 0
    done = False
    while not done:
        decay_step += 1
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        img, img_global = env.render()
        # env.render()
        swarm_x, swarm_y, swarm_vel, swarm_angle_rec, obs_x, obs_y = get_swarm(env)

        if not evaluate:
            start = time.time()
            action = policy(state, decay_step, evaluate)
            end = time.time()
            time_step = end - start
            emergency_flag = np.zeros([env.agent_num, 1], dtype=bool)
        else:
            start = time.time()
            action_pred = policy(state, decay_step, evaluate)
            action, emergency_flag, obs_flag = hyper_guarantee(env, action_pred, obs_x, obs_y,
                                                     swarm_x, swarm_y, swarm_vel)
            action = action_pred
            end = time.time()
            time_step = end - start

        fileT.write('%f \n' % (time_step))

        # Recieve state and reward from environment.
        state_, local_reward, global_reward, self_perception, done, info = env.step(action, step, emergency_flag)

        vid1.write(np.uint8(img_global))

        buffer.record((state, action, local_reward, global_reward, self_perception, state_, done))
        episodic_reward += np.sum(global_reward)
        if not evaluate:
            buffer.learn('critic')
            if ep % 3 == 0 and ep >= 10:
                buffer.learn('actor')
            for agent in range(env.agent_num):
                update_target(target_Actor[agent].variables, Actor[agent].variables, tau)
                # update_target(local_target_Critic[agent].variables, local_Critic[agent].variables, tau)
            update_target(target_Critic1.variables, Critic1.variables, tau)
            print("Episode {} step {} time {}".format(ep, step, time.ctime()))
            print("local_reward: ")
            print(local_reward)
            # print("global_reward {}".format(global_reward))
            # print("credit_pred: {}".format(critic_w))
            # print("credit_true: {}".format(self_perception_uni_mod))
            # print("action: {}".format(action))
        else:
            if online_lr:
                buffer.learn('critic')
                if ep % 3 == 0 and ep >= 10:
                    buffer.learn('actor')
                for agent in range(env.agent_num):
                    update_target(target_Actor[agent].variables, Actor[agent].variables, tau)
                    # update_target(local_target_Critic[agent].variables, local_Critic[agent].variables, tau)
                update_target(target_Critic1.variables, Critic1.variables, tau)
            else:
                pass
            if info == 'crash':
                num_failure += 1
            else:
                if obs_flag:
                    num_success += 1
                    avoid_times += 1

        state = state_
        step += 1

        if env.agent_num == 2:
            fileA0.write('%f  %f \n' % (swarm_x[0], swarm_y[0]))
            fileA1.write('%f  %f \n' % (swarm_x[1], swarm_y[1]))
        elif env.agent_num == 3:
            fileA0.write('%f  %f \n' % (swarm_x[0], swarm_y[0]))
            fileA1.write('%f  %f \n' % (swarm_x[1], swarm_y[1]))
            fileA2.write('%f  %f \n' % (swarm_x[2], swarm_y[2]))
        elif env.agent_num == 4:
            fileA0.write('%f  %f \n' % (swarm_x[0], swarm_y[0]))
            fileA1.write('%f  %f \n' % (swarm_x[1], swarm_y[1]))
            fileA2.write('%f  %f \n' % (swarm_x[2], swarm_y[2]))
            fileA3.write('%f  %f \n' % (swarm_x[3], swarm_y[3]))

        if env.obs_num == 1:
            fileO0.write('%f  %f \n' % (obs_x[0], obs_y[0]))
        elif env.obs_num == 2:
            fileO0.write('%f  %f \n' % (obs_x[0], obs_y[0]))
            fileO1.write('%f  %f \n' % (obs_x[1], obs_y[1]))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # # End this episode when `done` is True
        # if done:
        #     break
        if step >= env._max_episode_steps - 1:
            break

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = plot_model(episodic_reward, ep)
    if not evaluate:
        if avg_reward > best_score:
            # Save the models
            for agent in range(env.agent_num):
                Actor[agent].save('DDPG_Actor' + str(agent) + script_name + '.h5')
            Critic1.save('DDPG_Critic' + script_name + '.h5')

            for agent in range(env.agent_num):
                target_Actor[agent].save('DDPG_Target_Actor' + str(agent) + script_name + '.h5')
            target_Critic1.save('DDPG_Target_Critic' + script_name + '.h5')
            best_score = avg_reward
    else:
        if online_lr:
            if avg_reward > best_score:
                # Save the models
                for agent in range(env.agent_num):
                    Actor[agent].save('oneline_DDPG_Actor' + str(agent) + script_name + '.h5')
                Critic1.save('oneline_DDPG_Critic' + script_name + '.h5')

                for agent in range(env.agent_num):
                    target_Actor[agent].save('oneline_DDPG_Target_Actor' + str(agent) + script_name + '.h5')
                target_Critic1.save('oneline_DDPG_Target_Critic' + script_name + '.h5')
                best_score = avg_reward
        else:
            pass
        rate_failure = num_failure / (avoid_times + 1)
        rate_success = num_success / (avoid_times + 1)
        print('episode:{}, success rate:{}, failure rate:{}'.format(ep, rate_success, rate_failure))

    # avg_reward = np.mean(ep_reward_list[-40:])
    avg_reward_list.append(avg_reward)
    file.write('%f  %f  %f\n' % (episodic_reward, ep, avg_reward))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    if env.agent_num == 2:
        fileA0.close()
        fileA1.close()
    elif env.agent_num == 3:
        fileA0.close()
        fileA1.close()
        fileA2.close()
    elif env.agent_num == 4:
        fileA0.close()
        fileA1.close()
        fileA2.close()
        fileA3.close()
    fileT.close()
    if env.obs_num == 1:
        fileO0.close()
    elif env.obs_num == 2:
        fileO0.close()
        fileO1.close()
vid1.release()
file.close()