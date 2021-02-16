import numpy as np
import math
import matplotlib.pyplot as plt
from mc_project.lattice import LatticeStructure


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
            total_energy += sum(sum(diff*diff))
    return total_energy


def energy_diff(cand, curr, neighbors, config):
    energy = 0
    for j, _ in neighbors:
        psi_neigh = config[j]
        cand_diff = cand - psi_neigh
        curr_diff = curr - psi_neigh
        energy += sum(sum(cand_diff*cand_diff)) - sum(sum(curr_diff*curr_diff))
    return energy


def MC_step(n_points, config, graph, beta):
    '''Monte Carlo move using Metropolis algorithm '''
    for i in range(n_points):
        rand_pos = np.random.randint(0, n_points)
        curr = config[rand_pos]
        cand = np.random.uniform(low=-1, high=1, size=(4, 2))
        norm = sum(sum(cand*cand))
        cand *= 1/norm
        upd = curr
        neighbors = graph[i][1]
        del_E = energy_diff(cand, curr, neighbors, config)
        if del_E < 0:
            upd = cand
        elif np.random.rand() < np.exp(-del_E*beta):
            upd = cand
        config[i] = upd
    return config


def calculation(eqSteps, err_runs, group, cutoff, cutoff_type='m', size=1, error=10**(-8)):

    lattice = LatticeStructure(group=group, cutoff=cutoff, cutoff_type=cutoff_type, size=size, error=error)
    graph = lattice.periodic_graph()
    n_points = len(graph)

    config = init(n_points)
        
    nt      = 5         #  number of temperature points
    mcSteps = 100
        
    # the number of MC sweeps for equilibrium should be at least equal to the number of MC sweeps for equilibrium

    # initialization of all variables
    T = np.linspace(1., 7., nt)
    E = np.zeros(nt)
        
    Energies = []
    delEnergies = []

    for t in range(nt):
        # initialize total energy and mag
        beta = 1./T[t]

        print("Beginning temp step: ", t+1)
        # evolve the system to equilibrium
        for _ in range(eqSteps):
            MC_step(n_points, config, graph, beta)

        # Perform sweeps
        Ez = []
        for _ in range(err_runs):
            E = 0
            for _ in range(mcSteps):
                MC_step(n_points, config, graph, beta)           
                energy = total_E(config, graph) # calculate the energy at time stamp

                # sum up total energy and mag after each time steps

                E += energy
            # mean (divide by total time steps)

            E_mean = E/mcSteps

            # calculate macroscopic properties (divide by # sites) and append

            Energy = E_mean/n_points

            Ez.append(Energy)

        Energy = np.mean(Ez)
        Energies.append(Energy)
        delEnergy = np.std(Ez)
        delEnergies.append(float(delEnergy))

    results = [T, Energies, delEnergies]

    return results


def plots(eqSteps, err_runs, group, cutoff, cutoff_type='m', size=1, error=10**(-8)):

    print("Performing calculation")
    T, Energies, delEnergies = calculation(eqSteps, err_runs, group, cutoff, cutoff_type='m', size=1, error=10**(-8))
    print("Plotting")

    # shift_T = [i for i in range(len(T))]

    plt.errorbar(x=T, y=Energies, yerr=delEnergies)

    plt.show()



if __name__ == "__main__":
    eqSteps = 500
    err_runs = 20
    group = 'h'
    cutoff = 9
    size = 1
    sequential = False
    initial = 20
    plots(eqSteps, err_runs, group, cutoff=cutoff, size=size, error=10**(-8))


