import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import combinations
import os
import configparser
config = configparser.ConfigParser()
# config['DEFAULT'] = {'agent_num': '2',
#                      'obs_num': '1',
#                      'reward_0_if_done': 'True',
#                      'ngh_in_state': 'True',
#                      'v2v_comms': 'True',
#                      'v2v_collision': 'True',
#                      'id_concat': 'False',
#                      'action_space': 'Continuous'}
# with open('default.ini', 'w') as configfile:
#     config.write(configfile)

if os.path.isfile('default.ini'):
    config.read('default.ini')
    agent_num = np.int(config['DEFAULT']['agent_num'])
    obs_num = np.int(config['DEFAULT']['obs_num'])
    agent_vel = np.int(config['DEFAULT']['agent_vel'])
    obs_vel = np.int(config['DEFAULT']['obs_vel'])
    obstacle_range = np.int(config['DEFAULT']['obstacle_range'])
    v2v_range = np.int(config['DEFAULT']['v2v_range'])
    angle_bound = np.int(config['DEFAULT']['angle_bound'])
    reward_0_if_done = config['DEFAULT']['reward_0_if_done']
    ngh_in_state = config['DEFAULT']['ngh_in_state']
    obs_in_state = config['DEFAULT']['obs_in_state']
    relative_position = config['DEFAULT']['relative_position']
    average_state = config['DEFAULT']['average_state']
    v2v_comms = config['DEFAULT']['v2v_comms']
    v2v_collision = config['DEFAULT']['v2v_collision']
    id_concat = config['DEFAULT']['id_concat']
    action_space = config['DEFAULT']['action_space']
    if reward_0_if_done == 'False': reward_0_if_done = False
    elif reward_0_if_done == 'True': reward_0_if_done = True
    if ngh_in_state == 'False': ngh_in_state = False
    elif ngh_in_state == 'True': ngh_in_state = True
    if obs_in_state == 'False': obs_in_state = False
    elif obs_in_state == 'True': obs_in_state = True
    if relative_position == 'True': relative_position = True
    elif relative_position == 'False': relative_position = False
    if average_state == 'True': average_state = True
    elif average_state == 'False': average_state = False
    if v2v_comms == 'False': v2v_comms = False
    elif v2v_comms == 'True': v2v_comms = True
    if v2v_collision == 'False': v2v_collision = False
    elif v2v_collision == 'True': v2v_collision = True
    if id_concat == 'False': id_concat = False
    elif id_concat == 'True': id_concat = True
else:
    agent_num = 2
    obs_num = 1
    agent_vel = 3
    obs_vel = 3
    angle_bound = 45
    obstacle_range = 20
    v2v_range = 5
    reward_0_if_done = True
    ngh_in_state = True
    relative_position = False
    average_state = False
    v2v_comms = True
    v2v_collision = False
    id_concat = False
    action_space = 'Continuous'

size_env = 300
entry = agent_num + obs_num + 1 # ID + (self + neighbor) + obs + target x/y
# entry = 5
layers = 2  # x/y
property_num = 4  # 1: presence, 2: x/y, 3: current vx/current vy, 4: target vx/vy

state_size = (entry, property_num, layers)
num_actions = 1

action_upper_bound = 1
action_lower_bound = -1
angle_upper_bound = angle_bound
angle_lower_bound = -angle_bound

class Object:
    def __init__(self, x, y, vel, angle_record):
        self.x_orig = x
        self.y_orig = y
        self.x = x
        self.y = y
        self.vel = vel
        self.action_upper_bound = action_upper_bound
        self.action_lower_bound = action_lower_bound
        self.angle_upper_bound = angle_upper_bound
        self.angle_lower_bound = angle_lower_bound
        self.observe_range = 50
        self.angle_bound = angle_bound
        self.size_env = size_env
        self.x_rec = []
        self.y_rec = []
        self.N = 100
        self.field = np.zeros((self.size_env, self.size_env))
        self.power_init = 1000
        self.power_remain = 1000
        self.power_map = np.zeros(self.size_env * self.size_env)
        self.angle_record = angle_record
        self.target_x = x + 200
        self.target_y = y
        self.target_vx = 1
        self.target_vy = 0
        self.vx = 0
        self.vy = 0
        self.detected = False
        self.crashed = False
        self.angle_action_range = 90
        self.angle_action_step = 10

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def power(self, power_consume):
        self.power_remain -= power_consume
        power_level = np.floor(self.power_remain / self.power_init * (self.size_env * self.size_env))
        # if power_level == self.size_env * self.size_env:
        #     self.power_map[0:] = 1
        # elif power_level <= 0:
        #     self.power_map[0:] = 0
        # else:
        #     self.power_map[0:power_level.astype(int)] = 1
        #     self.power_map[power_level.astype(int):] = 0

    def action(self, choice, emergency_flag):
        '''
        Gives us 21 total movement options. (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)
        Choice = [-10:10] degrees
        '''
        if emergency_flag:
            self.angle_upper_bound, self.angle_lower_bound = 90, -90
        else:
            self.angle_upper_bound, self.angle_lower_bound = angle_upper_bound, angle_lower_bound
        if action_space == 'Continuous':
            # angle_map = self.angle_upper_bound - (self.action_upper_bound - choice) \
            #             * (self.angle_upper_bound - self.angle_lower_bound) \
            #             / (self.action_upper_bound - self.action_lower_bound)
            angle_map = choice * self.angle_upper_bound
            self.angle_record += angle_map
            self.vx = np.cos(self.angle_record/180*np.pi)
            self.vy = np.sin(self.angle_record/180*np.pi)
            dx = self.vel * np.cos(self.angle_record/180*np.pi)
            dy = self.vel * np.sin(self.angle_record/180*np.pi)
        elif action_space == 'Discrete':
            angle = np.arange(-self.angle_action_range, self.angle_action_range+1, self.angle_action_step)
            self.angle_record += angle[choice]
            self.vx = np.cos(self.angle_record/180*np.pi)
            self.vy = np.sin(self.angle_record/180*np.pi)
            dx = self.vel * np.cos(self.angle_record/180*np.pi)
            dy = self.vel * np.sin(self.angle_record/180*np.pi)
        else:
            print('Action space is either continuous or discrete! ')

        self.move(x=dx, y=dy)

    def move(self, x, y):
        self.x += x
        self.y += y

        # If we are out of bounds, fix!
        if self.x < 1:
            self.x = 1
        # elif self.x > self.size_env-1:
        #     self.x = self.size_env-1
        if self.y < 1:
            self.y = 1
        elif self.y > self.size_env-1:
            self.y = self.size_env-1

        self.x_rec.append(self.x)
        self.y_rec.append(self.y)

    def internal_reward(self):
        # Interpolation
        x_temp = self.x_rec
        y_temp = self.y_rec
        # x_temp = self.x_rec[-3:]
        # y_temp = self.y_rec[-3:]
        # for i in range(len(x_temp)-1):
        #     x_temp_left = x_temp[i]
        #     x_temp_right = x_temp[i+1]
        #     x_temp_interp_temp = np.linspace(x_temp_left, x_temp_right, num=self.N)
        #     x_temp_interp = np.concatenate((x_temp_interp, x_temp_interp_temp), axis=0)
        # y_temp_interp = np.interp(x_temp_interp, x_temp, y_temp)
        if len(x_temp) >= 3:
            curvature = np.mean(np.sqrt((np.array(x_temp[3 - 1::1]) - 2 * np.array(x_temp[2 - 1:-1:1]) + np.array(x_temp[1 - 1:-2:1])) ** 2 + (np.array(y_temp[3 - 1::1]) - 2 * np.array(y_temp[2 - 1:-1:1]) + np.array(y_temp[1 - 1:-2:1])) ** 2))
        else:
            curvature = 0
        continuity = 0
        # curvature = np.sum(np.sqrt((x_temp_interp[3 - 1::1] - 2 * x_temp_interp[2 - 1:-1:1] + x_temp_interp[1 - 1:-2:1]) ** 2 + (y_temp_interp[3 - 1::1] - 2 * y_temp_interp[2 - 1:-1:1] + y_temp_interp[1 - 1:-2:1]) ** 2))
        # curvature = np.sum((x_temp_interp[3 - 1::1] - 2 * x_temp_interp[2 - 1:-1:1] + x_temp_interp[1 - 1:-2:1]) ** 2 + (y_temp_interp[3 - 1::1] - 2 * y_temp_interp[2 - 1:-1:1] + y_temp_interp[1 - 1:-2:1]) ** 2)
        internal_reward = continuity + curvature

        return internal_reward

    def path_len(self):
        x_temp = self.x_rec
        y_temp = self.y_rec
        x_temp_diff = np.diff(x_temp)
        y_temp_diff = np.diff(y_temp)
        path_len = np.sum(np.sqrt(x_temp_diff ** 2 + y_temp_diff ** 2))
        return path_len

class uav_swarm_env:
    def __init__(self):
        self._max_episode_steps = 100
        self.object_size = 5
        self.size_env = size_env
        self.entry = entry  # self + neighbor + obs + target x/y
        self.layers = layers  # x/y
        self.property = property_num  # 1: presence, 2: x/y, 3: current vx/current vy, 4: target vx/vy
        self.agent_num = agent_num
        self.num_actions = num_actions
        self.angle_bound = angle_bound
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound = action_upper_bound
        self.angle_upper_bound = angle_upper_bound
        self.angle_lower_bound = angle_lower_bound
        self.obs_num = obs_num
        self.swarm = {}
        for i in range(self.agent_num):
            self.swarm[i] = Object(0, 0, 0, 0)
        self.obstacle = {}
        self.target_range = 50
        self.obstacle_range = obstacle_range
        self.v2v_range =  v2v_range
        self.reward_0_if_done = reward_0_if_done
        self.ngh_in_state = ngh_in_state
        self.obs_in_state = obs_in_state
        self.v2v_comms = v2v_comms
        self.v2v_collision = v2v_collision
        self.id_concat = id_concat
        if average_state:
            self.scale_factor = self.size_env
        else:
            self.scale_factor = 1
        self.reset_obs_flag = False
        self.init_oind = 0

    def reset_obs(self, x_cent=250, y_cent=150, vel_ang_init=180):
        #### Reset obstacle ####
        # if obs_num <= 3:
        #     self.obs_num = np.random.random_integers(1, obs_num)
        # else:
        #     self.obs_num = np.random.random_integers(obs_num - 3, obs_num)
        self.obs_num = obs_num
        # x_cent, y_cent = 250, 150
        for i in range(self.obs_num):
            obs_x_init = [250, 150, 150]
            obs_y_init = [150, 50, 250]
            vel_ang_init = [180, 90, -90]
            # self.init_oind = np.random.randint(3)
            self.init_oind = 0
            if self.init_oind == 0:
                x_cent = obs_x_init[self.init_oind]
                y_cent = obs_y_init[self.init_oind] + (-1) ** np.random.randint(2) * np.random.randint(20)
                obs_angl = vel_ang_init[self.init_oind]
            elif self.init_oind == 1:
                x_cent = obs_x_init[self.init_oind] + (-1) ** np.random.randint(2) * np.random.randint(50)
                y_cent = obs_y_init[self.init_oind]
                obs_angl = vel_ang_init[self.init_oind]
            else:
                x_cent = obs_x_init[self.init_oind] + (-1) ** np.random.randint(2) * np.random.randint(50)
                y_cent = obs_y_init[self.init_oind]
                obs_angl = vel_ang_init[self.init_oind]

            self.obstacle[i] = Object(0, 0, 0, obs_angl)
            self.obstacle[i].x = x_cent
            self.obstacle[i].y = y_cent + (-1) ** np.random.randint(2) * np.random.randint(50)
            self.obstacle[i].vel = obs_vel
            # self.obstacle[i].vx = self.obstacle[i].vel * np.cos(obs_angl / 180 * np.pi)
            # self.obstacle[i].vy = self.obstacle[i].vel * np.sin(obs_angl / 180 * np.pi)
            self.obstacle[i].vx = np.cos(obs_angl / 180 * np.pi)
            self.obstacle[i].vy = np.sin(obs_angl / 180 * np.pi)
            self.obstacle[i].detected = False

    def reset_swarm(self):
        #### Reset swarm ####
        r = 40
        theta = np.linspace(-np.pi/2, 3 * np.pi/2, self.agent_num + 1)
        x_cent, y_cent = 50, 150
        for i in range(self.agent_num):
            self.swarm[i].x = np.int(np.ceil(r * np.cos(theta[i]) + x_cent))
            self.swarm[i].y = np.int(np.ceil(r * np.sin(theta[i]) + y_cent))
            self.swarm[i].target_x = self.swarm[i].x + 200
            self.swarm[i].target_y = self.swarm[i].y
            self.swarm[i].x_orig = self.swarm[i].x
            self.swarm[i].y_orig = self.swarm[i].y
            self.swarm[i].vel = agent_vel
            self.swarm[i].x_rec = []
            self.swarm[i].y_rec = []
            self.swarm[i].x_rec.append(self.swarm[i].x)
            self.swarm[i].y_rec.append(self.swarm[i].y)
            self.swarm[i].target_vx = np.cos(0)
            self.swarm[i].target_vy = np.sin(0)
            self.swarm[i].angle_record = 0
            self.swarm[i].vx = np.cos(self.swarm[i].angle_record / 180 * np.pi)
            self.swarm[i].vy = np.sin(self.swarm[i].angle_record / 180 * np.pi)
            self.swarm[i].crashed = False

    def reset(self):
        self.reset_swarm()
        self.reset_obs()
        img, img_global = self.get_image()

        return img

    def render(self):
        img, img_global = self.get_image()
        ######################################################################
        img_global = img_global * 255
        img_global = cv2.resize(img_global, (300, 300), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow('Time_' + str(0), img_global)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyWindow()
        #     return
        ######################################################################

        return img, img_global

    def step(self, action, step, emergency_flag):
        # Take actions and calculate distances
        pos_obs = np.zeros((2, self.obs_num)) # 1st row x, 2ne row y.
        for i in range(self.obs_num):
            self.obstacle[i].action(0, emergency_flag=False)
            pos_obs[0][i], pos_obs[1][i] = self.obstacle[i].x, self.obstacle[i].y

        pos_swarm = np.zeros((2, self.agent_num)) # 1st row x, 2ne row y.
        for i in range(self.agent_num):
            pos_swarm[0][i], pos_swarm[1][i] = self.swarm[i].x, self.swarm[i].y

        # if len(self.obstacle[0].x_rec) >= 100:
        # # if np.mean(pos_obs[0]) - np.min(pos_swarm[0]) < -10:
        #     # self.reset_obs_flag = True
        #     self.reset_obs(x_cent=round(np.mean(pos_swarm[0]) + 2 * self.size_env / 3),
        #                    y_cent=150)
        #     # self.reset_obs_flag = False
        #     pos_obs = np.zeros((2, self.obs_num))  # 1st row x, 2ne row y.
        #     for i in range(self.obs_num):
        #         self.obstacle[i].action(0, emergency_flag=False)
        #         pos_obs[0][i], pos_obs[1][i] = self.obstacle[i].x, self.obstacle[i].y

        v2v_comb = np.array(list(combinations(np.arange(self.agent_num), 2)))
        dist_v2v_before_action = np.zeros((v2v_comb.shape[0], 1))
        for i in range(len(v2v_comb)):
            dist_v2v_before_action[i] = np.sqrt((self.swarm[v2v_comb[i][0]].x - self.swarm[v2v_comb[i][1]].x)**2 + (self.swarm[v2v_comb[i][0]].y - self.swarm[v2v_comb[i][1]].y)**2)

        gravity_penalty_before_action = np.zeros((self.agent_num, 1))
        dist2obs_before_action = np.zeros((self.agent_num, 1))
        for i in range(self.agent_num):
            vec2obs_before_action = (self.swarm[i].x - pos_obs[0], self.swarm[i].y - pos_obs[1])
            dist2obs_before_action[i] = np.min([np.min(np.linalg.norm(vec2obs_before_action, axis=0)), self.swarm[i].observe_range])
            gravity_penalty_before_action[i] = np.abs(self.swarm[i].y - self.swarm[i].y_orig)
            self.swarm[i].action(action[i], emergency_flag[i])

        dist_v2v_after_action = np.zeros((v2v_comb.shape[0], 1))
        for i in range(len(v2v_comb)):
            dist_v2v_after_action[i] = np.sqrt((self.swarm[v2v_comb[i][0]].x - self.swarm[v2v_comb[i][1]].x)**2 + (self.swarm[v2v_comb[i][0]].y - self.swarm[v2v_comb[i][1]].y)**2)

        # Generate state_
        state_, img_global = self.render()

        # Calculate local rewards
        dist2obs_after_action = np.zeros((self.agent_num, 1))
        dist2ngh_after_action = np.zeros((self.agent_num, 1))
        dist_after_action = np.zeros((self.agent_num, 1))
        dist2obs_grad = np.zeros((self.agent_num, 1))
        path_len = np.zeros((self.agent_num, 1))
        vel2tar_inner_product = np.zeros((self.agent_num, 1))
        vel2tar_inner_product_pre = vel2tar_inner_product
        done = np.zeros((self.agent_num, 1), dtype=bool)
        collide = np.zeros((self.agent_num, 1), dtype=bool)
        local_reward = np.zeros((self.agent_num, 1))
        internal_reward = np.zeros((self.agent_num, 1))
        internal_reward_pre = np.zeros((self.agent_num, 1))
        external_reward = np.zeros((self.agent_num, 1))
        gravity_penalty = np.zeros((self.agent_num, 1))
        dist2target = np.zeros((self.agent_num, 1))
        gravity_penalty_pre = gravity_penalty
        action_pre = gravity_penalty
        self_perception = np.zeros((self.agent_num, 1))
        for i in range(self.agent_num):
            obs_ind = np.where(state_[i][0][:, 0, 0] == 2)[0]
            if len(obs_ind) == 0:
                dist2obs_temp_min = self.swarm[i].observe_range
            else:
                dist2obs_temp = []
                vec2obs_temp = []
                for obs in obs_ind:
                    observation_obs = np.squeeze(state_[i][0, obs])
                    if observation_obs[0][0] == 2:
                        if not relative_position:
                            dist_temp = np.sqrt((observation_obs[1, 0]*self.scale_factor
                                                - state_[i][0, 0+np.int(self.id_concat), 1, 0]*self.scale_factor) ** 2
                                                + (observation_obs[1, 1]*self.scale_factor
                                                - state_[i][0, 0+np.int(self.id_concat), 1, 1]*self.scale_factor) ** 2)
                            vec_temp = [observation_obs[1, 0]*self.scale_factor
                                        - state_[i][0, 0+np.int(self.id_concat), 1, 0]*self.scale_factor,
                                        observation_obs[1, 1]*self.scale_factor
                                        - state_[i][0, 0+np.int(self.id_concat), 1, 1]*self.scale_factor]
                        else:
                            dist_temp = np.sqrt((observation_obs[1, 0]) ** 2
                                                + (observation_obs[1, 1]) ** 2)
                            vec_temp = [observation_obs[1, 0], observation_obs[1, 1]]
                        dist2obs_temp.append(dist_temp)
                        vec2obs_temp.append(vec_temp / dist_temp)
                dist2obs_temp_min = np.min([np.min(dist2obs_temp), self.swarm[i].observe_range])
            dist2obs_after_action[i] = dist2obs_temp_min

            # dist2obs_grad[i] = 1/(dist2obs_after_action[i] - self.obstacle_range + 1e-8)**2 \
            #                    - 1/(dist2obs_before_action[i] - self.obstacle_range + 1e-8)**2
            dist2obs_grad[i] = (dist2obs_after_action[i] - dist2obs_before_action[i])\
                               / (np.abs(dist2obs_before_action[i] - (self.obstacle_range + 10)) + 1e-8)

            # print('dist_before {}'.format(dist2obs_before_action[i]))
            # print('dist_after {}'.format(dist2obs_after_action[i]))
            # print('dist_grad {}'.format(dist2obs_grad[i]))

            ngh_ind = np.where(state_[i][0][:, 0, 0] == 1)[0]
            if len(ngh_ind) == 0:
                dist2ngh_after_action[i] = self.swarm[i].observe_range
            elif len(ngh_ind) == 1 and ngh_ind[0] == 0:
                dist2ngh_after_action[i] = self.swarm[i].observe_range
            else:
                dist2ngh_temp = []
                vec2ngh_temp = []
                for ngh in ngh_ind:
                    if ngh == 0:
                        pass
                    else:
                        observation_ngh = np.squeeze(state_[i][0, ngh])
                        if not relative_position:
                            dist_temp = np.sqrt((observation_ngh[1, 0]*self.scale_factor
                                                 - state_[i][0, 0+np.int(self.id_concat), 1, 0]*self.scale_factor) ** 2
                                                + (observation_ngh[1, 1]*self.scale_factor
                                                   - state_[i][0, 0+np.int(self.id_concat), 1, 1]*self.scale_factor) ** 2)
                            vec_temp = [observation_ngh[1, 0]*self.scale_factor
                                        - state_[i][0, 0+np.int(self.id_concat), 1, 0]*self.scale_factor,
                                        observation_ngh[1, 1]*self.scale_factor
                                        - state_[i][0, 0+np.int(self.id_concat), 1, 1]*self.scale_factor]
                        else:
                            dist_temp = np.sqrt((observation_ngh[1, 0]) ** 2
                                                + (observation_ngh[1, 1]) ** 2)
                            vec_temp = [observation_ngh[1, 0], observation_ngh[1, 1]]
                        dist2ngh_temp.append(dist_temp)
                        vec2ngh_temp.append(vec_temp / dist_temp)
                dist2ngh_after_action[i] = np.min(dist2ngh_temp)

            # gravity_penalty[i] = np.abs(state_[i][0, 0+np.int(self.id_concat), 1, 1] \
            #                      - state_[i][0, -1, 1, 1])/(self.size_env/2)
            gravity_penalty[i] = np.abs(state_[i][0, 0+np.int(self.id_concat), 1, 1]*self.scale_factor \
                                 - state_[i][0, -1, 1, 1]*self.scale_factor)

            dist2target[i] = np.sqrt((state_[i][0, 0+np.int(self.id_concat), 1, 0]*self.scale_factor - state_[i][0, -1, 1, 0]*self.scale_factor)**2 \
                             + (state_[i][0, 0+np.int(self.id_concat), 1, 1]*self.scale_factor - state_[i][0, -1, 1, 1]*self.scale_factor)**2)

            path_len[i] = self.swarm[i].path_len()

            vel2tar_inner_product[i] = state_[i][0, 0+np.int(self.id_concat), 2, 0] \
                                       * state_[i][0, -1, 2, 0] \
                                       + state_[i][0, 0+np.int(self.id_concat), 2, 1] \
                                       * state_[i][0, -1, 2, 1]

            # if dist2obs_after_action[i] > self.obstacle_range and dist2ngh_after_action[i] > self.v2v_range * np.int(self.v2v_collision):
            #     pass
            #     # collide[i] = False
            #     # done[i] = False
            # else:
            #     self.swarm[i].crashed = True
            #     # Only the crash step is penalized, after that penalty is 0.
            #     # collide_penalty[i] = 10

            dist_after_action[i] = np.min([dist2ngh_after_action[i] - self.v2v_range,
                                           dist2obs_after_action[i] - self.obstacle_range])

            if dist_after_action[i] <= 0:
                self.swarm[i].crashed = True
                external_reward[i] = 0
            else:
                external_reward[i] = 0

            # Once marked as crashed, always marked as crashed.
            if self.swarm[i].crashed:
                collide[i] = True
                done[i] = True

            # if step >= self._max_episode_steps - 1:
            #     done[i] = True

            internal_reward[i] = vel2tar_inner_product[i] * 2 - np.tanh(gravity_penalty[i] / 10)
            # internal_reward[i] = vel2tar_inner_product[i] * 1
            self_perception[i] = internal_reward[i] + dist_after_action[i]

            local_reward[i] = internal_reward[i]
            # local_reward[i] = 1 - np.tanh(gravity_penalty[i] / 10)
            # local_reward[i] = 3 + (vel2tar_inner_product[i]) * 2 - np.abs(action[i]) - np.tanh(gravity_penalty[i]) * 2
            # if internal_reward[i] >= internal_reward_pre[i]:
            #     local_reward[i] = 1
            # else:
            #     local_reward[i] = -1

            internal_reward_pre[i] = internal_reward[i]
            vel2tar_inner_product_pre[i] = vel2tar_inner_product[i]
            action_pre[i] = action[i]
            gravity_penalty_pre[i] = gravity_penalty[i]

            # if done[i]:
            #     # local_reward[i] = 0
            #     # self_perception[i] = 1e-8
            #     internal_reward[i] = 3 + (vel2tar_inner_product[i]) * 2 - np.abs(action[i][0]) \
            #                           - np.tanh(gravity_penalty[i]) * 2
            #     external_reward[i] = - collide_penalty[i] - np.abs(np.tanh(dist2ngh_after_action[i])) * 0
            #                          # - np.abs(np.tanh(dist2ngh_after_action[i] - self.v2v_range))
            #                          # + dist_after_action[i]/self.swarm[i].observe_range \
            #     local_reward[i] = internal_reward[i] + external_reward[i]
            #     self_perception[i] = internal_reward[i] + dist_after_action[i]
            # else:
            #     # if vel2tar_inner_product[i] < 0.5:
            #     #     vel2tar_inner_product[i] = 0
            #     internal_reward[i] = 3 + (vel2tar_inner_product[i]) * 2 - np.abs(action[i][0]) \
            #                           - np.tanh(gravity_penalty[i]) * 2
            #     external_reward[i] = - collide_penalty[i] - np.abs(np.tanh(dist2ngh_after_action[i])) * 0
            #                          # - np.abs(np.tanh(dist2ngh_after_action[i] - self.v2v_range))
            #                          # + dist_after_action[i]/self.swarm[i].observe_range
            #     local_reward[i] = internal_reward[i] + external_reward[i]
            #     self_perception[i] = internal_reward[i] + dist_after_action[i]

            # print('inner_product:{}, gravity:{}, internal:{}'.format(vel2tar_inner_product[i], -gravity_penalty[i], -np.abs(action[i][0])))
            # print('reward:{}'.format(local_reward[i]))

        C = 0
        if any(collide) or step >= self._max_episode_steps - 1:
            info = 'crash'
            penalty = -C
            target_reward = 0
        else:
            info = 'n_crash'
            penalty = 0
            if all(dist2target < self.target_range):
                target_reward = C
            else:
                target_reward = 0

        global_reward = np.sum(local_reward) + penalty + target_reward
        # global_reward = penalty + target_reward
        # if any(collide):
        #     # global_reward = np.zeros((self.agent_num, 1))
        #     global_reward *= 0

        # if self.init_oind == 0:
        #     if np.mean(pos_obs[0]) - 10 < -5:
        #         obs_done = True
        #     else:
        #         obs_done = False
        # elif self.init_oind == 1:
        #     if np.mean(pos_obs[1]) - 290 > 5:
        #         obs_done = True
        #     else:
        #         obs_done = False
        # elif self.init_oind == 2:
        #     if np.mean(pos_obs[1]) - 10 < -5:
        #         obs_done = True
        #     else:
        #         obs_done = False

        if np.max(pos_obs[0]) <= 20:
            obs_done = True
        else:
            obs_done = False

        if all(dist2target < self.target_range) or step >= self._max_episode_steps - 1:
            done = True
        else:
            done = any(done)

        return state_, global_reward, done, info, local_reward

    def get_image(self):
        # Local observations
        img_observed_set = np.zeros((self.agent_num, 1, self.entry, self.property, self.layers))
        for i in range(self.agent_num):
            id_onehot = np.zeros((1, self.property))
            # id_onehot[0, i] = 1
            img = np.zeros((self.entry, self.property, self.layers)) # agent:1/obs:0, position(x,y), vel(vx,vy), target_vel(vx,vy)
            if self.id_concat:
                img[0, :, 0] = id_onehot
            img[0 + np.int(self.id_concat), :, 0] = [0, self.swarm[i].x/self.scale_factor , self.swarm[i].vx, 0]
            img[0 + np.int(self.id_concat), :, 1] = [0, self.swarm[i].y/self.scale_factor , self.swarm[i].vy, 0]
            near_ind = 0+np.int(self.id_concat)
            ###################################################################################
            ## UAVs (ego and neighbour) velocity in state only have directions.
            ###################################################################################
            if self.ngh_in_state:
                for j in range(self.agent_num):
                    if i != j:
                        dist = np.sqrt((self.swarm[i].x-self.swarm[j].x)**2
                                       + (self.swarm[i].y-self.swarm[j].y)**2)
                        if dist <= self.swarm[i].observe_range:
                            near_ind += 1
                            if not relative_position:
                                img[near_ind, :, 0] = [1, self.swarm[j].x/self.scale_factor, self.swarm[j].vx, 0]
                                img[near_ind, :, 1] = [1, self.swarm[j].y/self.scale_factor, self.swarm[j].vy, 0]
                            else:
                                img[near_ind, :, 0] = [1, (self.swarm[j].x - self.swarm[i].x)/self.scale_factor,
                                                       self.swarm[j].vx, 0]
                                img[near_ind, :, 1] = [1, (self.swarm[j].y - self.swarm[i].y)/self.scale_factor,
                                                       self.swarm[j].vy, 0]
            ###################################################################################
            ## Obstacles velocity in state have both directions and magnitudes.
            ###################################################################################
            if self.obs_in_state:
                for k in range(self.obs_num):
                    dist = np.sqrt((self.swarm[i].x - self.obstacle[k].x) ** 2
                                   + (self.swarm[i].y - self.obstacle[k].y) ** 2)
                    if dist <= self.swarm[i].observe_range or self.obstacle[k].detected:
                        if self.v2v_comms:
                            self.obstacle[k].detected = True
                        else:
                            pass
                        near_ind += 1
                        if not relative_position:
                            img[near_ind, :, 0] = [2, self.obstacle[k].x/self.scale_factor, self.obstacle[k].vx, 0]
                            img[near_ind, :, 1] = [2, self.obstacle[k].y/self.scale_factor, self.obstacle[k].vy, 0]
                        else:
                            img[near_ind, :, 0] = [2, (self.obstacle[k].x - self.swarm[i].x)/self.scale_factor,
                                                   self.obstacle[k].vx, 0]
                            img[near_ind, :, 1] = [2, (self.obstacle[k].y - self.swarm[i].y)/self.scale_factor,
                                                   self.obstacle[k].vy, 0]

            img[-1, :, 0] = [0, self.swarm[i].target_x/self.scale_factor, self.swarm[i].target_vx, 0]
            img[-1, :, 1] = [0, self.swarm[i].target_y/self.scale_factor, self.swarm[i].target_vy, 0]

            img_exp = np.expand_dims(img, axis=0)
            img_observed_set[i] = img_exp

        # Output images
        img_global = np.zeros((self.size_env, self.size_env, 3))
        swarm_x = []
        for i in range(self.agent_num):
            swarm_x.append(round(self.swarm[i].x))
        for i in range(self.agent_num):
            shift = 0
            img_global[round(self.swarm[i].y) - 5:round(self.swarm[i].y) + 5, round(self.swarm[i].x - shift) - 5:round(self.swarm[i].x - shift) + 5, 0] = 1  # Drones first layers
            img_global[round(self.swarm[i].target_y) - 5:round(self.swarm[i].target_y) + 5, round(self.swarm[i].target_x - shift) - 5:round(self.swarm[i].target_x - shift) + 5, 0] = 1  # Drones first layers
            r = self.target_range
            x_cent = round(self.swarm[i].target_x)
            y_cent = round(self.swarm[i].target_y)
            for angle in np.linspace(-np.pi, np.pi, 10):
                tp_x = np.int(np.ceil(r * np.cos(angle) + x_cent))
                tp_y = np.int(np.ceil(r * np.sin(angle) + y_cent))
                img_global[round(tp_y) - 1: round(tp_y) + 1, round(tp_x) - 1: round(tp_x) + 1, 0] = 1  # Target range first layers

            r = self.v2v_range
            x_cent = round(self.swarm[i].x)
            y_cent = round(self.swarm[i].y)
            for angle in np.linspace(-np.pi, np.pi, 10):
                tp_x = np.int(np.ceil(r * np.cos(angle) + x_cent))
                tp_y = np.int(np.ceil(r * np.sin(angle) + y_cent))
                img_global[round(tp_y) - 1: round(tp_y) + 1, round(tp_x) - 1: round(tp_x) + 1, 0] = 1  # V2V range first layers
            # if np.mean(swarm_x) <= self.size_env/2:
            #     shift = 0
            #     img_global[round(self.swarm[i].y) - 5:round(self.swarm[i].y) + 5,
            #     round(self.swarm[i].x - shift) - 5:round(self.swarm[i].x - shift) + 5, 0] = 1 # Drones first layers
            # else:
            #     shift = np.mean(swarm_x) - self.size_env/2
            #     img_global[round(self.swarm[i].y) - 5:round(self.swarm[i].y) + 5,
            #     round(self.swarm[i].x - shift) - 5:round(self.swarm[i].x - shift) + 5, 0] = 1  # Drones first layers
            ############################################################################
            # r_observe = self.swarm[i].observe_range
            # r_thr = self.obstacle_range
            # theta = np.linspace(-np.pi, np.pi, 21)
            # x_cent, y_cent = round(self.swarm[i].y), round(self.swarm[i].x)
            # for c in range(len(theta)-1):
            #     x_observe = round(r_observe * np.cos(theta[c]) + x_cent)
            #     y_observe = round(r_observe * np.sin(theta[c]) + y_cent)
            #     x_thr = round(r_thr * np.cos(theta[c]) + x_cent)
            #     y_thr = round(r_thr * np.sin(theta[c]) + y_cent)
            #     if x_observe >= self.size_env:
            #         x_observe = self.size_env - 1
            #     elif x_observe < 0:
            #         x_observe = 0
            #     if y_observe >= self.size_env:
            #         y_observe = self.size_env - 1
            #     elif y_observe < 0:
            #         y_observe = 0
            #
            #     if x_thr >= self.size_env:
            #         x_thr = self.size_env - 1
            #     elif x_thr < 0:
            #         x_thr = 0
            #     if y_thr >= self.size_env:
            #         y_thr = self.size_env - 1
            #     elif y_thr < 0:
            #         y_thr = 0
            #
            #     img_global[x_observe, y_observe, 0] = 1
            #     img_global[x_thr, y_thr, 2] = 1
            ############################################################################

        for k in range(self.obs_num):
            img_global[round(self.obstacle[k].y)-5:round(self.obstacle[k].y)+5, round(self.obstacle[k].x - shift)-5:round(self.obstacle[k].x - shift)+5, 1] = 1  # Obstacle successive layers
            r = self.obstacle_range
            x_cent = round(self.obstacle[k].x)
            y_cent = round(self.obstacle[k].y)
            for angle in np.linspace(-np.pi, np.pi, 10):
                tp_x = np.int(np.ceil(r * np.cos(angle) + x_cent))
                tp_y = np.int(np.ceil(r * np.sin(angle) + y_cent))
                img_global[round(tp_y) - 1: round(tp_y) + 1, round(tp_x) - 1: round(tp_x) + 1, 1] = 1  # Obstacle range 2nd layers
        # for g in range(self.agent_num):
        #     img_global[round(self.target[g].y) - 5:round(self.target[g].y) + 5, round(self.target[g].x) - 5:round(self.target[g].x) + 5, 2] = 1

        # swarm_x = []
        # for i in range(self.agent_num):
        #     swarm_x.append(round(self.swarm[i].x))
        #
        # if np.mean(swarm_x) > size_env/2:
        #     img_global = np.concatenate([img_global[:, np.max(swarm_x)-np.int(size_env/2):, :],
        #                                  img_global[:, 0:np.max(swarm_x)-np.int(size_env/2), :]], axis=1)

        return img_observed_set, img_global


