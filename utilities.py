import numpy as np
# import tensorflow_probability as tfp
# tfd = tfp.distributions
import scipy.stats

def get_swarm(env):
    swarm_x, swarm_y, swarm_vel, swarm_angle_rec = [], [], [], []
    obs_x, obs_y = [], []
    for agent in range(env.agent_num):
        swarm_x.append(env.swarm[agent].x)
        swarm_y.append(env.swarm[agent].y)
        swarm_vel.append(env.swarm[agent].vel)
        swarm_angle_rec.append(env.swarm[agent].angle_record)

    for obs in range(env.obs_num):
        obs_x.append(env.obstacle[obs].x)
        obs_y.append(env.obstacle[obs].y)

    return swarm_x, swarm_y, swarm_vel, swarm_angle_rec, obs_x, obs_y

def hyper_guarantee(env, action_pred, obs_x, obs_y, swarm_x, swarm_y, swarm_vel):
    agent_num = env.agent_num
    action_upper_bound = env.action_upper_bound
    action_lower_bound = env.action_lower_bound
    action_cal = action_pred.numpy()
    dist2obs_set = np.zeros([agent_num, 1])
    emergency_angle_upper_bound = 90
    emergency_angle_lower_bound = -90
    emergency_flag = np.zeros([agent_num, 1], dtype=bool)
    swarm_ind = set(np.arange(agent_num))
    for agent in range(agent_num):
        ego_ind = set([agent])
        sub_ind = list(swarm_ind - ego_ind)
        ego_ind = list(ego_ind)

        ego_x, ego_y = swarm_x[ego_ind[0]], swarm_y[ego_ind[0]]
        sub_x, sub_y = [], []
        for i in range(len(sub_ind)):
            sub_x.append(swarm_x[sub_ind[i]])
            sub_y.append(swarm_y[sub_ind[i]])
        ego_vel = swarm_vel[ego_ind[0]]

        action_noise = np.concatenate([np.linspace(-1, action_pred[agent], num=10),
                                       np.linspace(action_pred[agent], 1, num=10)])
        # action_noise_distribution = tfd.Normal(loc=action_pred[agent], scale=0.1)
        # action_noise_prob = action_noise_distribution.prob(action_noise).numpy()
        action_noise_distribution = scipy.stats.norm(loc=action_pred[agent], scale=0.1)
        action_noise_prob = action_noise_distribution.pdf(action_noise)
        angle_map = emergency_angle_upper_bound - (action_upper_bound - action_noise) \
                    * (emergency_angle_upper_bound - emergency_angle_lower_bound) \
                    / (action_upper_bound - action_lower_bound)
        angle_result = env.swarm[agent].angle_record + angle_map
        dx = ego_vel * np.cos(angle_result / 180 * np.pi)
        dy = ego_vel * np.sin(angle_result / 180 * np.pi)
        x_result = ego_x + dx
        y_result = ego_y + dy
        dist2obs = np.sqrt((ego_x - np.stack(obs_x)) ** 2
                           + (ego_y - np.stack(obs_y)) ** 2)
        dist2obs_set[agent] = np.min(dist2obs)
        nearest_obs = np.argmin(dist2obs)
        dist2obs_noise = np.sqrt((x_result - obs_x[nearest_obs]) ** 2
                                 + (y_result - obs_y[nearest_obs]) ** 2)

        dist2ngh = np.sqrt((ego_x - np.stack(sub_x)) ** 2
                           + (ego_y - np.stack(sub_y)) ** 2)
        nearest_ngh = np.argmin(dist2ngh)
        dist2ngh_noise = np.sqrt((x_result - sub_x[nearest_ngh]) ** 2
                                 + (y_result - sub_y[nearest_ngh]) ** 2)

        obs_collision_index = np.where(np.squeeze(dist2obs_noise) <= env.obstacle_range + 10)[0]
        ngh_collision_index = np.where(np.squeeze(dist2ngh_noise) <= env.v2v_range + 10)[0]
        # collision_index = np.concatenate([obs_collision_index, ngh_collision_index])
        collision_index = obs_collision_index

        if len(collision_index) > 0:
            emergency_flag[agent] = 1
            action_noise_prob[collision_index] = 0
            action_cal[agent] = action_noise[np.argmax(action_noise_prob)]
            angle_deg = angle_map[np.argmax(action_noise_prob)]
            # print('Calibrating! action:{}'.format(angle_deg))

        if any(dist2obs_set < env.obstacle_range*2):
            obs_flag = True
        else:
            obs_flag = False

    return action_cal, emergency_flag, obs_flag