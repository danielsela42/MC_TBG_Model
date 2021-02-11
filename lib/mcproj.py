import numpy as np
import math
import matplotlib.pyplot as plt


def init(n_points):
    # create config
    config = np.random.uniform(low=-1.0, high=1.0, size=(n_points, 4, 2))

    # Fix phase
    for i in range(n_points):
        config[i, 0, 1] = 0
        norm = sum(sum(config[i]*config[i]))
        config[i] *= 1/norm
    return config



def total_E(config, graph):
    total_energy = 0
    for i in range(len(config)):
        psi = config[i]
        for ind, _ in graph[i][1]:
            diff = psi - config[ind]
            total_energy += sum(diff*diff)
    return total_energy


def energy_diff(cand, curr, neighbors):
    energy = 0
    for neighbor in neighbors:
        cand_diff = cand - neighbor
        curr_diff = curr - neighbor
        energy += sum(cand_diff*cand_diff) - sum(curr_diff*curr_diff)
    return energy


def MC_step(n_points, config, graph, beta):
    '''Monte Carlo move using Metropolis algorithm '''
    for i in range(n_points):
        cand = np.random.randint(0, n_points) # looping over i & j therefore use a & b
        curr =  config[i]
        upd = curr
        neighbors = graph[i][1]
        del_E = energy_diff(cand, curr, neighbors)
        if del_E < 0:
            upd = cand
        elif np.random.rand() < np.exp(-del_E*beta):
            upd = cand
        config[i] = upd
    return config


