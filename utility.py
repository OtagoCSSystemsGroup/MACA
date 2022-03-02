import numpy as np
import matplotlib.pyplot as plt
# import matlab
# import matlab.engine
# eng = matlab.engine.start_matlab()
import cv2 as cv

def snakeCostFun(gamma1, gamma2, swarmIndAvoiding, arcxTemp, arcyTemp, arcxPre, arcyPre, EextField, interpFlag):
    # x = arcxTemp
    # y = arcyTemp
    # z = arczTemp
    # xPre = np.zeros([N, 1, 2])
    # yPre = np.zeros([N, 1, 2])
    # zPre = np.zeros([N, 1, 2])
    # posIniIndiv = swarm_position
    # posIniSwarmRel, posIniSwarmRelNext = virtual_position, virtual_position
    # smoothField = field_env
    # EextField = np.concatenate([np.expand_dims(smoothField, -1), np.expand_dims(smoothField, -1)], axis=2) # To be done.

    N, PopulationSize, swarmSize = arcxTemp.shape
    arcxPre = arcxPre * np.ones([1, PopulationSize, 1])
    arcyPre = arcyPre * np.ones([1, PopulationSize, 1])

    x1 = np.concatenate([arcxPre[0:-1, :, :], arcxTemp[0:, :, :]], axis=0)
    y1 = np.concatenate([arcyPre[0:-1, :, :], arcyTemp[0:, :, :]], axis=0)

    Econt = 0
    # Econt = np.sum(np.abs((x1[0:-1, :, :] - x1[1:, :, :])) + np.abs((y1[0:-1, :, :] - y1[1:, :, :])), axis=0, keepdims=True)
    Ecurv = np.sum(np.abs((x1[2:, :, :] - 2 * x1[1:-1, :, :] + x1[0:-2, :, :])) + np.abs((y1[2:, :, :] - 2 * y1[1:-1, :, :] + y1[0:-2, :, :])), axis=0, keepdims=True)
    Eint = Econt + Ecurv

    if interpFlag:
        S1 = x1.shape[0]
        z1 = np.ones([S1, PopulationSize, 1]) * swarmIndAvoiding
        x1_mat = matlab.double(x1.tolist())
        y1_mat = matlab.double(y1.tolist())
        z1_mat = matlab.double(z1.tolist())
        EextField_mat = matlab.double(EextField.tolist())
        Eext1_mat = eng.interp3(EextField_mat, x1_mat, y1_mat, z1_mat)
        Eext1 = np.asarray(Eext1_mat)
        Eext1[np.isnan(Eext1)] = 0
        # Eext1 = np.reshape(Eext1, [S1, PopulationSize, swarmSize])
        Eext = np.sum(np.abs(Eext1), axis=0, keepdims=True)
        E = gamma1 * Eint - gamma2 * Eext
    else:
        E = Eint

    return E

def get_position(env, field_size):
    agent_num = len(env.swarm)
    obs_num = len(env.obstacle)

    swarm_position = np.zeros([agent_num, 2]) # col1: x, col2: y
    for idx in range(agent_num):
        swarm_position[idx] = [env.swarm[idx].x, env.swarm[idx].y]

    virtual_position = np.mean(swarm_position, axis=0) + [20., 0.]

    obstacle_position = np.zeros([obs_num, 2]) # col1: x, col2: y
    for idx in range(obs_num):
        obstacle_position[idx] = [env.obstacle[idx].x, env.obstacle[idx].y]

    return swarm_position, virtual_position, obstacle_position

def path_prediction(swarm_position, arcxPre, arcyPre, killing_intensity, swarmIndAvoiding, field_env, N):
    gamma1, gamma2 = 0.5, 0.5
    swarm_intensity = field_env[np.round(swarm_position[:, 0] - 1).astype(int), np.round(swarm_position[:, 1] - 1).astype(int)]
    swarm_intensity[swarm_intensity >= np.min(killing_intensity)] = np.min(killing_intensity) * 0.99
    agent_num = swarm_position.shape[0]
    field_size = field_env.shape[0]
    EextField = np.zeros([field_size, field_size, agent_num])
    predx, predy = [], []
    for ind_i in range(agent_num):
        agentx = swarm_position[ind_i, 0]
        agenty = swarm_position[ind_i, 1]
        ret, thresh = cv.threshold(field_env, swarm_intensity[ind_i], 255, 0)
        # EextField[:, :, ind_i] = thresh
        contours, hierarchy = cv.findContours(thresh.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # plt.imshow(thresh)
        # plt.plot(contours[0][:, :, 0], contours[0][:, :, 1], 'w')
        # plt.plot(contours[1][:, :, 0], contours[1][:, :, 1], 'w')
        if len(contours) > 1:
            distCont = np.zeros(len(contours))
            for ind_p in range(len(contours)):
                contxTemp = contours[ind_p][:, :, 1]
                contyTemp = contours[ind_p][:, :, 0]
                distCont[ind_p] = np.min(np.sqrt((contxTemp - agentx) ** 2 + (contyTemp - agenty) ** 2))
            contInd = np.argmin(distCont)
            contx = contours[contInd][:, :, 1]
            conty = contours[contInd][:, :, 0]
        else:
            contInd = 0
            contx = contours[contInd][:, :, 1]
            conty = contours[contInd][:, :, 0]

        minDistInd = np.argmin(np.sqrt((contx - agentx) ** 2 + (conty - agenty) ** 2))
        maxDistInd = np.argmax(np.sqrt((contx - agentx) ** 2 + (conty - agenty) ** 2))

        contxPart1 = contx[minDistInd: 0: -1]
        contxPart2 = contx[minDistInd: ]
        contyPart1 = conty[minDistInd: 0: -1]
        contyPart2 = conty[minDistInd: ]

        if len(contxPart1) >= N:
            contxPart1 = contxPart1[0:N, :]
        elif 0 < len(contxPart1) < N:
            contxPart1 = np.concatenate([contxPart1[0:N, :], contxPart1[-1, :] * np.ones([N - len(contxPart1), 1])])
        else:
            contxPart1 = np.ones([N, 1]) * arcxPre[-1, :, ind_i]

        if len(contxPart2) >= N:
            contxPart2 = contxPart2[0:N, :]
        elif 0 < len(contxPart2) < N:
            contxPart2 = np.concatenate([contxPart2[0:N, :], contxPart2[-1, :] * np.ones([N - len(contxPart2), 1])])
        else:
            contxPart2 = np.ones([N, 1]) * arcxPre[-1, :, ind_i]

        if len(contyPart1) >= N:
            contyPart1 = contyPart1[0:N, :]
        elif 0 < len(contyPart1) < N:
            contyPart1 = np.concatenate([contyPart1[0:N, :], contyPart1[-1, :] * np.ones([N - len(contyPart1), 1])])
        else:
            contyPart1 = np.ones([N, 1]) * arcyPre[-1, :, ind_i]

        if len(contyPart2) >= N:
            contyPart2 = contyPart2[0:N, :]
        elif 0 < len(contyPart2) < N:
            contyPart2 = np.concatenate([contyPart2[0:N, :], contyPart2[-1, :] * np.ones([N - len(contyPart2), 1])])
        else:
            contyPart2 = np.ones([N, 1]) * arcyPre[-1, :, ind_i]

        E1 = np.mean(contxPart1 * arcxPre[:, :, ind_i] + contyPart1 * arcyPre[:, :, ind_i])
        E2 = np.mean(contxPart2 * arcxPre[:, :, ind_i] + contyPart2 * arcyPre[:, :, ind_i])

        # E1 = snakeCostFun(gamma1, gamma2, swarmIndAvoiding,
        #                   np.expand_dims(contxPart1, axis=-1),
        #                   np.expand_dims(contyPart1, axis=-1),
        #                   np.expand_dims(arcxPre[:, :, ind_i], axis=-1),
        #                   np.expand_dims(arcyPre[:, :, ind_i], axis=-1), EextField * 0, False)
        # E2 = snakeCostFun(gamma1, gamma2, swarmIndAvoiding,
        #                   np.expand_dims(contxPart2, axis=-1),
        #                   np.expand_dims(contyPart2, axis=-1),
        #                   np.expand_dims(arcxPre[:, :, ind_i], axis=-1),
        #                   np.expand_dims(arcyPre[:, :, ind_i], axis=-1), EextField * 0, False)

        if E1 >= E2:
            predx.append(contxPart1)
            predy.append(contyPart1)
        else:
            predx.append(contxPart2)
            predy.append(contyPart2)

    return predx, predy





