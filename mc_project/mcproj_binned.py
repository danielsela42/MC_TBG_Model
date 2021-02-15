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


def binning(data, binned_series=list()):
    if not binned_series:
        binned_series.append(data)
    if len(data) <= 2:
        return
    binned = list()
    length = int(len(data)/2)
    for i in range(length):
        binned.append((1/2)*(data[2*i] + data[2*i+1]))

    binned_series.append(binned)
    binning(binned, binned_series)
        


def calculation(eq_steps, mcSteps, n_bins, group, cutoff, cutoff_type='m', size=1, error=10**(-8)):

    lattice = LatticeStructure(group=group, cutoff=cutoff, cutoff_type=cutoff_type, size=size, error=error)
    graph = lattice.periodic_graph()
    n_points = len(graph)

    config = init(n_points)
        
    nt      = 10         #  number of temperature points
        
    # the number of MC sweeps for equilibrium should be at least equal to the number of MC sweeps for equilibrium

    # initialization of all variables
    T = np.linspace(1., 7., nt)
    E = np.zeros(nt)
        
    Energies = list()
    Energies_squared = list()
    delEnergies = list()
    error_list = list()
    relaxation_times = list()

    for t in range(nt):
        # initialize total energy and mag
        beta = 1./T[t]

        print("Beginning temp step: ", t+1)
        # evolve the system to equilibrium
        for _ in range(eq_steps):
            MC_step(n_points, config, graph, beta)

        # Perform sweeps
        E = list()
        for _ in range(mcSteps):
            MC_step(n_points, config, graph, beta)           
            energy = total_E(config, graph)/n_points # calculate the energy at time stamp

            # sum up total energy and mag after each time steps

            E.append(energy)

        print(len(E))

        binned_series = list()
        binning(E, binned_series)

        print(len(binned_series))

        errors = list()
        for bin_list in binned_series:
            Ml = len(bin_list)
            if Ml == 1:
                continue
            avg = np.mean(bin_list)
            summation = np.sum([(p - avg)**2 for p in bin_list])
            bin_error = np.sqrt((1/(Ml*(Ml-1)))*summation)
            errors.append(bin_error)

        # Approximate limit
        error = errors[-1]

        rel_t = (1/2)*((error/errors[0])**2 -1)
        relaxation_times.append(rel_t)

        Energy = np.mean(E)
        Energy_squared = np.mean(np.array(Energy)*np.array(Energy))
        Energies.append(Energy)
        Energies_squared.append(Energy_squared)
        error_list.append(errors)
        delEnergy = error
        delEnergies.append(float(delEnergy))

    results = [T, Energies, delEnergies, Energies_squared, relaxation_times, error_list]

    return results


def plots(eq_steps, mcSteps, n_bins, group, cutoff, cutoff_type='m', size=1, error=10**(-8)):

    print("Performing calculation")
    T, Energies, delEnergies, Energies_squared, rel_times, error_list = calculation(eq_steps, mcSteps, n_bins, group, cutoff, cutoff_type='m', size=1, error=10**(-8))
    print("Plotting")
    l_list = [i for i in range(len(error_list[0]))]

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()

    ax = axs[0, 0]
    ax.errorbar(x=T, y=Energies, yerr=delEnergies)
    ax.set_title('Energies')
    ax = axs[0, 1]
    ax.errorbar(x=T, y=Energies_squared)
    ax.set_title('Energies Squared')
    ax = axs[1, 0]
    ax.errorbar(x=l_list, y = error_list[0])
    ax.set_title('Error for T = 1')

    plt.show()

    print(rel_times)


if __name__ == "__main__":
    eqSteps = 1000
    group = 'h'
    cutoff = 9
    size = 1
    n_bins = 10
    mcSteps = 2**12
    plots(eqSteps, mcSteps, n_bins, group, cutoff=cutoff, size=size, error=10**(-8))


