import numpy as np
import math
import multiprocessing as mp
import matplotlib.pyplot as plt
from mc_project.utilities import LatticeStructure
import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs


def total_E(config, graph):
     return config

def energy_diff(cand, curr, config):
    energy = cand - curr
    return energy


def MC_step(config, energy_diff, n_points, beta, n_accepted_list, sigma):
    '''Monte Carlo move using Metropolis algorithm '''
    curr = config

    # Choose and normalize candidate
    cand = np.random.normal(curr, sigma)
    if cand > 10 or cand < -10: return curr

    # Accept or deny candidate
    upd = curr
    del_E = energy_diff(cand, curr, config)
    if del_E < 0:
        upd = cand
        n_accepted_list[0] += 1
    elif np.random.rand() < np.exp(-del_E*beta):
        upd = cand
        n_accepted_list[0] += 1
    return upd


def calculation(nt, beta_range, eq_steps, mc_steps, energy_diff, group, cutoff, convergence=False, n_chains=12, sigma_list=list(), n_processes=12, ddof=0, cutoff_type='m', size=1, error=10**(-8), log_to_consol=False):
    ''' Perform Monte Carlo calculation for the model

    Inputs: nt - # temperature points
            eq_steps - steps to equilibriate system after temperature change
            mc_steps - # of sweeps and data collection to perform
            kappa - standard deviation/step size for monte carlo
            group - lattice type
            cutoff - magnitude cutoff for lattice
            func_list - list of function names to calculate from data (objects not strings)
            cutoff_type - type of cutoff (default: 'm')
            size - size of basis vectors for lattice
            error - allowed error in calculation

    return: [T, quantity_dict]. T is temperature points, quantity_dict is q a dictionary for each func_list containing a dictionary
            of values, relaxation times, and errors
    '''
    # Shift cutoff to smallest multiple of 3 above current
    cutoff += 3 - (cutoff % 3)

    # Create lattice and graph (graph provides neighboring points)
    lattice = LatticeStructure(group=group, cutoff=cutoff, cutoff_type=cutoff_type, size=size, error=error)
    graph = lattice.periodic_graph()
    n_points = len(graph)

    # Initialize configuration
    config = 3

    # Get coupling constant (beta)
    beta_T = np.linspace(beta_range[0], beta_range[1], nt)

    acceptance_rates =list()

    chains = list()

    for t in range(nt):
        # initialize temperature
        beta = beta_T[t]
        sigma = sigma_list[t]

        if log_to_consol: print("Beginning temp step: ", t+1)

        configs = list()

        # evolve the system to equilibrium
        n_accepted_list = [0]
        for _ in range(eq_steps):
            config = MC_step(config, energy_diff, n_points, beta, n_accepted_list, sigma)

        # Perform sweeps
        n_accepted_list = [0]
        for _ in range(mc_steps):
            config = MC_step(config, energy_diff, n_points, beta, n_accepted_list, sigma)
            configs.append(config)
        n_accepted = n_accepted_list[0]
        acceptance_rate = n_accepted/(mc_steps*n_points)
        print("     ", acceptance_rate)

        acceptance_rates.append(acceptance_rate)
        chains.append(configs)

    results = chains

    return results


def plots(nt, beta_range, eq_steps, mc_steps, group, cutoff, func_list, sigma_list=list(), ddof=0, cutoff_type='m', size=1, error=10**(-8)):

    chains = calculation(nt, beta_range, eq_steps, mc_steps, energy_diff, group, cutoff, func_list, sigma_list=sigma_list, ddof=ddof,cutoff_type='m', size=1, error=10**(-8), log_to_consol=True)

    for i in range(len(chains)):
        plt.figure(i+1)
        # energy_list = [chains[i][j]**2 for j in range(len(chains[i]))]
        energy_list = chains[i]
        plt.hist(energy_list, bins=100, density=True)
        # plt.scatter(energy_list, np.exp(-1*beta[i]*np.array(energy_list)))
    plt.show()


if __name__ == "__main__":
    eqSteps = 1000
    group = 'h'
    cutoff = 9
    size = 1
    # mc_steps = 2**15
    mc_steps = 100000
    nt = 5
    ddof = 1
    beta_range = [1, 5]
    func_list = list()

    # kappa_list = [14.5, 15.87, 17.2, 18.8, 20.5, 22.65, 25.05, 27.7, 30.8, 34.8]
    # kappa_list = [14.5, 15.1, 15.8, 16.4, 17.1, 17.8, 18.7, 19.4, 20.1, 21.1, 22.2, 23.3, 24.4, 25.8, 26.9, 28.2, 30, 31.5, 33.6, 35.8]
    
    #kappa_list = [3, 4, 5, 6, 7, 8, 11, 12, 13, 14.5]
    #kappa_list = [0.3, 0.6, 1, 2, 3]
    sigma_list = [1, 1, 1, 1, 1]
    plots(nt, beta_range, eqSteps, mc_steps, group, sigma_list=sigma_list, cutoff=cutoff, func_list=func_list, ddof=ddof, size=size, error=10**(-8))