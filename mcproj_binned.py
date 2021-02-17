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


def parameter_estimation(par_list, batch_size, num_batches):
    means = [np.mean(par) for par in par_list]
    variances = [np.var(par) for par in par_list]

    batch_means = list()
    for i in range(num_batches):
        par_vars = list()
        for par in par_list:
            batch_mean = (1/batch_size)*sum([par[j] for j in range(batch_size*i, batch_size*(i+1))])
            par_vars.append(batch_mean)
        batch_means.append(par_vars)

    rel_times = list()
    errors = list()
    count = 0
    for par in par_list:
        variance = variances[count]
        mean = means[count]
        batch_var = (1/num_batches)*sum([(batch_means[i] - mean)**2 for i in range(num_batches)])

        # Approximate autocorrelation time
        rel_t = batch_size*batch_var/variance
        rel_times.append(rel_t)

        error = variance*np.sqrt((1+2*rel_t)/mcSteps)
        error.append(errors)
        count += 1
    
    return rel_times, errors
        


def calculation(eq_steps, mcSteps, group, cutoff, cutoff_type='m', size=1, error=10**(-8)):
    ''' Perform Monte Carlo calculation for the model

    Inputs: eq_steps - steps to equilibriate system after temperature change
            mcSteps - # of sweeps and data collection to perform
    '''
    
    # Shift cutoff to smallest multiple of 3 above current
    cutoff += 3 - (cutoff % 3)

    # Batch size
    M = int(np.rint(mcSteps**(1/3)))
    num_batches = int(mcSteps/M)

    print(mcSteps, M)
    print(mcSteps/M, int(mcSteps/M))

    # Create lattice and graph (graph provides neighboring points)
    lattice = LatticeStructure(group=group, cutoff=cutoff, cutoff_type=cutoff_type, size=size, error=error)
    graph = lattice.periodic_graph()
    n_points = len(graph)

    config = init(n_points)
        
    nt = 5         #  number of temperature points
        
    # the number of MC sweeps for equilibrium should be at least equal to the number of MC sweeps for equilibrium

    # initialization of all variables
    T = np.linspace(1., 7., nt)
    E = np.zeros(nt)
        
    Energies = list()
    Energies_squared = list()
    delEnergies = list()
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

            # Add to the list of energies after each sweep

            E.append(energy)

        mean = np.mean(E)
        variance = (1/mcSteps)*sum([(E_val- mean)**2 for E_val in E])
        var2 = np.var(E)
        print(var2, variance)

        batch_means = list()
        for i in range(num_batches):
            batch_mean = (1/M)*sum([E[j] for j in range(M*i, M*(i+1))])
            batch_means.append(batch_mean)
    
        batch_var = (M/mcSteps)*sum([(batch_means[i] - mean)**2 for i in range(num_batches)])

        # Approximate autocorrelation time
        rel_t = M*batch_var/variance
        relaxation_times.append(rel_t)

        error = variance*np.sqrt((1+2*rel_t)/mcSteps)

        # Calculate energies and get error as limit of binned errors
        Energy = np.mean(E)
        Energy_squared = np.mean(np.array(Energy)*np.array(Energy))
        Energies.append(Energy)
        Energies_squared.append(Energy_squared)
        delEnergy = error
        delEnergies.append(float(delEnergy))

    results = [T, Energies, delEnergies, Energies_squared, relaxation_times]

    return results


def plots(eq_steps, mcSteps, group, cutoff, cutoff_type='m', size=1, error=10**(-8)):

    print("Performing calculation")
    T, Energies, delEnergies, Energies_squared, rel_times = calculation(eq_steps, mcSteps, group, cutoff, cutoff_type='m', size=1, error=10**(-8))
    print("Plotting")

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()

    ax = axs[0, 0]
    ax.errorbar(x=T, y=Energies, yerr=delEnergies)
    ax.set_title('Energies')
    ax = axs[0, 1]
    ax.errorbar(x=T, y=Energies_squared)
    ax.set_title('Energies Squared')
    ax = axs[1, 0]
    ax.errorbar(x=T, y = delEnergies)
    ax.set_title('Error at various T')
    ax = axs[1, 1]
    ax.errorbar(x=T, y = rel_times)
    ax.set_title('Relaxation times at each temperature')

    plt.show()

    print(rel_times)


if __name__ == "__main__":
    eqSteps = 1000
    group = 'h'
    cutoff = 24
    size = 0.2
    mcSteps = 2**12
    plots(eqSteps, mcSteps, group, cutoff=cutoff, size=size, error=10**(-8))


