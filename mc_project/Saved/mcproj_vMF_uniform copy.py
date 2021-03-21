import numpy as np
import math
import matplotlib.pyplot as plt
from mc_project.lattice import LatticeStructure
from mc_project import sample_vMF


def init(n_points):
    # create config
    config = np.random.normal(0, 1, size=(n_points, 4, 2))

    # Fix phase
    for i in range(n_points):
        norm = np.sqrt(sum(sum(config[i]*config[i])))
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


def avg_energy(config, graph):
    return total_E(config, graph)/len(graph)

def squared_E(config, graph):
    return avg_energy(config, graph)**2


def energy_diff(cand, curr, neighbors, config):
    energy = 0
    for j, _ in neighbors:
        psi_neigh = config[j]
        cand_diff = cand - psi_neigh
        curr_diff = curr - psi_neigh
        energy += sum(sum(cand_diff*cand_diff)) - sum(sum(curr_diff*curr_diff))
    return energy


def MC_step(n_points, config, graph, kappa, beta):
    '''Monte Carlo move using Metropolis algorithm '''
    for _ in range(n_points):
        rand_pos = np.random.randint(0, n_points)
        curr = config[rand_pos]

        # Choose and normalize candidate
        cand = np.reshape(sample_vMF(np.concatenate(curr, axis=None), kappa, num_samples=1), (4, 2))

        # Accept or deny candidate
        upd = curr
        neighbors = graph[rand_pos][1]
        del_E = energy_diff(cand, curr, neighbors, config)
        if del_E < 0:
            upd = cand
        elif np.random.rand() < np.exp(-del_E*beta):
            upd = cand
        config[rand_pos] = upd
    return config

'''
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
'''

def estimate_correlation(quantities, mc_steps, batch_size, num_batches, ddof=0):
    ''' Calculates mean, relaxation time, and error, of a data set quantities

    input: quantities - list of data points for computation
           batch_size - size of batches
           num_batches - number of batches

    returns: mean, relaxation time, error
    '''
    mean  = np.mean(quantities)
    variance = np.var(quantities, ddof=ddof)

    batch_means = list()
    for i in range(num_batches):
        batch_mean = (1/batch_size)*np.sum([quantities[j] for j in range(batch_size*i, batch_size*(i+1))])
        batch_means.append(batch_mean)

    batch_var = np.var(batch_means, ddof=ddof)

    rel_t = batch_size*batch_var/variance
    error = np.sqrt(variance*(1+2*rel_t)/(mc_steps))

    return mean, rel_t, error


def calculation(nt, eq_steps, mc_steps, kappa, group, cutoff, func_list, ddof=0, cutoff_type='m', size=1, error=10**(-8)):
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

    # Batch size
    M = int(np.rint(mc_steps**(1/3)))
    num_batches = int(mc_steps/M)

    # Create lattice and graph (graph provides neighboring points)
    lattice = LatticeStructure(group=group, cutoff=cutoff, cutoff_type=cutoff_type, size=size, error=error)
    graph = lattice.periodic_graph()
    n_points = len(graph)

    # Initialize configuration
    config = init(n_points)

    # Get temperature points
    T = np.linspace(1., 7., nt)

    quantity_dict = dict()

    for t in range(nt):
        # initialize total energy and mag
        beta = 1./T[t]

        print("Beginning temp step: ", t+1)
        # evolve the system to equilibrium
        for _ in range(eq_steps):
            MC_step(n_points, config, graph, kappa, beta)

        # Perform sweeps
        quantities = list()
        for _ in range(mc_steps):
            MC_step(n_points, config, graph, kappa, beta)

            # Calculate each quantity
            count = 0
            for func in func_list:
                quantity = func(config, graph)
                try:
                    quantities[count].append(quantity)
                except IndexError:
                    quantities.append([quantity])
                finally:
                    count += 1
        
    
        # Add quantities to dictionary with errors and relaxation times
        count = 0
        for q in quantities:
            # Estimate autocorrelation time and error
            mean_q, rel_t, del_quant = estimate_correlation(q, mc_steps, M, num_batches, ddof=ddof)

            # Get dict key
            dict_label = func_list[count].__name__

            # Add quantity itself
            try:
                quantity_dict[dict_label]["Values"].append(mean_q)
            except KeyError:
                quantity_dict[dict_label] = {"Values" : [mean_q]}

            # Add relaxation time
            try:
                quantity_dict[dict_label]["Relaxation times"].append(rel_t)
            except KeyError:
                quantity_dict[dict_label]["Relaxation times"] = [rel_t]
            
            # Add error
            try:
                quantity_dict[dict_label]["Errors"].append(del_quant)
            except KeyError:
                quantity_dict[dict_label]["Errors"] = [del_quant]
        
            count += 1
        

    results = [T, quantity_dict]

    return results


def plots(nt, eq_steps, mc_steps, kappa_list, group, cutoff, func_list, ddof=0, cutoff_type='m', size=1, error=10**(-8)):

    print("Performing calculation 1")
    T, quantity_dict1 = calculation(nt, eq_steps, mc_steps, kappa_list[0], group, cutoff, func_list, ddof=ddof,cutoff_type='m', size=1, error=10**(-8))
    
    print("Performing calculation 2")
    T, quantity_dict2 = calculation(nt, eq_steps, mc_steps, kappa_list[1], group, cutoff, func_list, ddof=ddof,cutoff_type='m', size=1, error=10**(-8))
    
    print("Performing calculation 3")
    T, quantity_dict3 = calculation(nt, eq_steps, mc_steps, kappa_list[2], group, cutoff, func_list, ddof=ddof,cutoff_type='m', size=1, error=10**(-8))
    
    print("Plotting")

    Energies1 = quantity_dict1["avg_energy"]["Values"]
    delEnergies1 = quantity_dict1["avg_energy"]["Errors"]
    Energies_squared1 = quantity_dict1["squared_E"]["Values"]
    rel_times1 = quantity_dict1["avg_energy"]["Relaxation times"]

    Energies2 = quantity_dict2["avg_energy"]["Values"]
    delEnergies2 = quantity_dict2["avg_energy"]["Errors"]
    Energies_squared2 = quantity_dict2["squared_E"]["Values"]
    rel_times2 = quantity_dict2["avg_energy"]["Relaxation times"]

    Energies3 = quantity_dict3["avg_energy"]["Values"]
    delEnergies3 = quantity_dict3["avg_energy"]["Errors"]
    Energies_squared3 = quantity_dict3["squared_E"]["Values"]
    rel_times3 = quantity_dict3["avg_energy"]["Relaxation times"]

    print(Energies1[0], Energies2[0], Energies3[0])

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Plots for kappa = {},{},{}". format(kappa_list[0], kappa_list[1], kappa_list[2]))
    fig.tight_layout()

    ax1 = axs[0, 0]
    ax1.errorbar(x=T, y=Energies1, yerr=delEnergies1, label="k=1")
    ax1.errorbar(x=T, y=Energies2, yerr=delEnergies2, label='k=10')
    ax1.errorbar(x=T, y=Energies3, yerr=delEnergies3, label='k=20')
    ax1.set_title('Energies')
    ax1.legend(loc = "lower right")

    ax2 = axs[0, 1]
    ax2.errorbar(x=T, y=Energies_squared1, label="k=1")
    ax2.errorbar(x=T, y=Energies_squared2, label='k=10')
    ax2.errorbar(x=T, y=Energies_squared3, label='k=20')
    ax2.set_title('Energies Squared')
    ax1.legend(loc = "lower right")

    ax3 = axs[1, 0]
    ax3.errorbar(x=T, y = delEnergies1, label="k=1")
    ax3.errorbar(x=T, y = delEnergies2, label='k=10')
    ax3.errorbar(x=T, y = delEnergies3, label='k=20')
    ax3.set_title('Error at various T')
    ax1.legend(loc = "upper right")

    ax4 = axs[1, 1]
    ax4.errorbar(x=T, y = rel_times1, label="k=1")
    ax4.errorbar(x=T, y = rel_times2, label='k=10')
    ax4.errorbar(x=T, y = rel_times3, label='k=20')
    ax4.set_title('Relaxation times at each temperature')
    ax1.legend(loc = "upper right")

    plt.show()


if __name__ == "__main__":
    eqSteps = 1000
    group = 'h'
    cutoff = 9
    size = 1
    mc_steps = 2**15
    mc_steps = 1000
    nt = 5
    ddof = 1
    func_list = [avg_energy, squared_E]

    kappa_list = [1, 10, 20]
    plots(nt, eqSteps, mc_steps, kappa_list, group, cutoff=cutoff, func_list=func_list, ddof=ddof, size=size, error=10**(-8))