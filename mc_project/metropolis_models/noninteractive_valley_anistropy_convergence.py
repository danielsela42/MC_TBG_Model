import numpy as np
import math
import multiprocessing as mp
import matplotlib.pyplot as plt
from mc_project.utilities import LatticeStructure
from mc_project.utilities import sample_vMF


def one_config_avg(config):
    field_average = np.mean(abs(config), axis=0)
    return [np.linalg.norm(field_average[i])**2 for i in range(4)]

'''
def average_comp_mag(configs):
    avgs = list()
    for config in configs:
        config_avg = one_config_avg(config)
        for i in range(4):
            try:
                avgs[i].append(config_avg[i])
            except IndexError:
                avgs.append([config_avg[i]])
    print(np.sum([np.mean(avgs[i]) for i in range(4)]))
    return [np.mean(avgs[i]) for i in range(4)]
'''
def average_comp_mag(configs):
    avg_Ap = list()
    avg_Am = list()
    avg_Bp = list()
    avg_Bm = list()
    for config in configs:
        Ap = list()
        Am = list()
        Bp = list()
        Bm = list()
        for sample in config:
            Ap.append(sample[0][0]**2 + sample[0][1]**2)
            Am.append(sample[1][0]**2 + sample[1][1]**2)
            Bp.append(sample[2][0]**2 + sample[2][1]**2)
            Bm.append(sample[3][0]**2 + sample[3][1]**2)
        avg_Ap.extend(Ap)
        avg_Am.extend(Am)
        avg_Bp.extend(Bp)
        avg_Bm.extend(Bm)
    print(np.sum([np.mean(avg_Ap), np.mean(avg_Am), np.mean(avg_Bp), np.mean(avg_Bm)]))
    return [np.mean(avg_Ap), np.mean(avg_Am), np.mean(avg_Bp), np.mean(avg_Bm)]



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
        total_energy += np.sum(psi[0]*psi[0]) - np.sum(psi[1]*psi[1]) + np.sum(psi[2]*psi[2]) - np.sum(psi[3]*psi[3])
    return total_energy


def avg_energy(config, graph):
    return total_E(config, graph)/len(graph)

def squared_E(config, graph):
    return avg_energy(config, graph)**2


def energy_diff(cand, curr, config):
    curr_sq = curr*curr
    cand_sq = cand*cand
    energy_curr = np.sum(curr_sq[0]) - np.sum(curr_sq[1]) + np.sum(curr_sq[2]) - np.sum(curr_sq[3])
    energy_cand = np.sum(cand_sq[0]) - np.sum(cand_sq[1]) + np.sum(cand_sq[2]) - np.sum(cand_sq[3])
    energy = energy_cand - energy_curr
    return energy


def MC_step(n_points, config, graph, beta, n_accepted_list):
    '''Monte Carlo move using Metropolis algorithm '''
    for i in range(n_points):
        # rand_pos = np.random.randint(0, n_points)
        rand_pos = i
        curr = config[rand_pos]

        # Choose and normalize candidate
        # cand = np.reshape(sample_vMF(np.concatenate(curr, axis=None), kappa, num_samples=1), (4, 2))
        cand = np.random.normal(0, 1, size=(4, 2))
        norm = np.sqrt(np.sum(np.sum(cand*cand)))
        inv_norm = 1/norm
        cand *= inv_norm

        # Accept or deny candidate
        upd = curr
        del_E = energy_diff(cand, curr, config)
        if del_E < 0:
            upd = cand
            n_accepted_list[0] += 1
        elif np.random.rand() < np.exp(-del_E*beta):
            upd = cand
            n_accepted_list[0] += 1
        config[rand_pos] = upd
    return config


def MC_step_vMF(n_points, config, graph, beta, kappa, n_accepted_list):
    '''Monte Carlo move using Metropolis algorithm '''
    for i in range(n_points):
        # rand_pos = np.random.randint(0, n_points)
        rand_pos = i
        curr = config[rand_pos]

        # Choose and normalize candidate
        cand = np.reshape(sample_vMF(np.concatenate(curr, axis=None), kappa, num_samples=1), (4, 2))

        # Accept or deny candidate
        upd = curr
        del_E = energy_diff(cand, curr, config)
        if del_E < 0:
            upd = cand
            n_accepted_list[0] += 1
        elif np.random.rand() < np.exp(-del_E*beta):
            upd = cand
            n_accepted_list[0] += 1
        config[rand_pos] = upd
    return config


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
    # print(rel_t, batch_size, batch_var, variance)
    error = np.sqrt(variance*(1+2*rel_t)/(mc_steps))

    return mean, rel_t, error


def mcmc_sampling(eq_steps, mc_steps, beta, n_points, graph):

    # Initialize configuration
    config = init(n_points)

    configs = list()

    # evolve the system to equilibrium
    n_accepted_list = [0]
    for _ in range(eq_steps):
        MC_step(n_points, config, graph, beta, n_accepted_list)

    # Perform sweeps
    n_accepted_list = [0]
    for _ in range(mc_steps):
        MC_step(n_points, config, graph, beta, n_accepted_list)
        configs.append(config.copy())
    n_accepted = n_accepted_list[0]

    return configs, n_accepted/(mc_steps*n_points)


def mcmc_sampling_vMF(eq_steps, mc_steps, beta, kappa, n_points, graph):

    # Initialize configuration
    config = init(n_points)

    configs = list()

    # evolve the system to equilibrium
    n_accepted_list = [0]
    for _ in range(eq_steps):
        MC_step_vMF(n_points, config, graph, beta, kappa, n_accepted_list)

    # Perform sweeps
    n_accepted_list = [0]
    for _ in range(mc_steps):
        MC_step_vMF(n_points, config, graph, beta, kappa, n_accepted_list)
        configs.append(config.copy())
    n_accepted = n_accepted_list[0]

    return configs, n_accepted/(mc_steps*n_points)


def burn_in(arg):
    eq_steps, beta, n_points, graph = arg

    # Initialize configuration
    config = init(n_points)

    chain = list()

    # evolve the system to equilibrium
    n_accepted_list = [0]
    for _ in range(eq_steps):
        config = MC_step(n_points, config, graph, beta, n_accepted_list)
        chain.append(config.copy())
    return chain


def multi_chains(arg_list):
    chains = list()
    for arg in arg_list:
        n_points = arg[2]
        eq_steps = arg[0]
        chain = burn_in(arg)
        reshaped_config = np.array(chain).reshape((eq_steps, n_points*8))
        chains.append(reshaped_config)
    return chains


def verify_convergence(n_processes, n_chains, eq_steps, beta, n_points, graph):
    arg = (eq_steps, beta, n_points, graph)

    args_list = [list() for i in range(n_processes)]

    for i in range(n_chains):
        args_list[i % n_processes].append(arg)
    
    print("Creating process pool")

    # Create pool
    pool = mp.Pool(processes=n_processes)
    results = pool.map_async(multi_chains, args_list)
    pool.close() # Close pool
    print("Process pool closed")
    pool.join() # Join pools
    print("Pools joined")

    # Get results
    chains = list()
    for res in results.get():
        chains.extend(res)

    print("got results")
    
    # Get colleciton of chains
    parallel_chains = np.array(chains)
    # Get collection of coordinates in each chain
    coords_chains = np.array([np.transpose(chain) for chain in parallel_chains])
    # Get the average of each coordinate per chain
    chain_coord_means = np.array([[np.mean(coord_set[i]) for i in range(n_points*8)] for coord_set in coords_chains])
    # Get the overall mean of each coordinate
    overall_mean = np.array([np.mean(np.transpose(chain_coord_means)[k]) for k in range(n_points*8)])

    print("Getting covariance matrices")
    #Initialize 
    between_chain_covar = np.zeros((n_points*8, n_points*8))
    within_chain_covar = np.zeros((n_points*8, n_points*8))
    for i in range(n_points*8):
        for j in range(n_points*8):
            if i > j: continue
            between_chain_covar[i, j] = (1/(n_chains-1)) * np.sum([(means[i] - overall_mean[i])*(means[j] - overall_mean[j]) for means in chain_coord_means])
            within_arr = [np.sum((parallel_chains[k, :, i] - chain_coord_means[k, i])*(parallel_chains[k, :, j] - chain_coord_means[k, j])/(n_chains*(mc_steps - 1))) for k in range(n_chains)]
            within_chain_covar[i, j] = np.sum(within_arr)
            if i < j:
                between_chain_covar[j, i] = between_chain_covar[i, j]
                within_chain_covar[j, i] = within_chain_covar[i, j]

    print("Calculating R matrix")
    covar = ((mc_steps - 1)/mc_steps)*within_chain_covar + between_chain_covar
    w_inv = np.linalg.inv(within_chain_covar)
    eigen_matr = np.dot(w_inv, covar)/mc_steps
    
    print("Getting eigenvalue")
    lambda1 = np.amax(np.linalg.eigvals(eigen_matr))

    R = (mc_steps - 1)/mc_steps + (1 + 1/n_chains)*lambda1

    print("R value", R)
    if R.real - 1 < 0.1 and abs(R.imag) < 0.01:
        return True
    else:
        return False 

def calculation(nt, eq_steps, mc_steps, group, cutoff, func_list, kappa_list=list(), n_processes=12, ddof=0, cutoff_type='m', size=1, error=10**(-8)):
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
    # Number of chains to test convergence
    # n_chains = 12
    
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
    T = np.linspace(0.05, 0.1, nt)

    quantity_dict = dict()

    acceptance_rates =list()

    beta_T = [1/T[nt - t - 1] for t in range(nt)]

    configs_avgs = list()

    for t in range(nt):
        # initialize temperature
        beta = beta_T[t]
        kappa = kappa_list[t]

        # Check convergence
        # convergence = verify_convergence(n_processes, n_chains, eq_steps, beta, n_points, graph)
        # print("\t Convergence: ", convergence)

        print("Beginning temp step: ", t+1)
        # evolve the system to equilibrium
        if kappa_list:
            configs, acceptance_rate = mcmc_sampling_vMF(eq_steps, mc_steps, beta, kappa, n_points, graph)
        else:
            configs, acceptance_rate = mcmc_sampling(eq_steps, mc_steps, beta, n_points, graph)
        print("\t The acceptanced rate is", acceptance_rate)
        acceptance_rates.append(acceptance_rate)

        # Collect quantities
        quantities = list()
        for config in configs:
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

        config_avg = average_comp_mag(configs)
        for i in range(4):
            try:
                configs_avgs[i].append(config_avg[i])
            except IndexError:
                configs_avgs.append([config_avg[i]])
    
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
        

    results = [beta_T, quantity_dict, configs_avgs, acceptance_rates]

    return results


def plots(nt, eq_steps, mc_steps, group, cutoff, func_list, kappa_list=list(), ddof=0, cutoff_type='m', size=1, error=10**(-8)):
    values = dict()
    errors = dict()
    rel_times = dict()

    beta, quantity_dict, configs_avgs, acceptance_rates = calculation(nt, eq_steps, mc_steps, group, cutoff, func_list, kappa_list=kappa_list, ddof=ddof,cutoff_type='m', size=1, error=10**(-8))
    
    for k, v in quantity_dict.items():
        try:
            values[k].extend(v["Values"])
        except KeyError:
            values[k] = v["Values"]

        try:
            errors[k].extend(v["Errors"])
        except KeyError:
            errors[k] = v["Errors"]

        try:
            rel_times[k].extend(v["Relaxation times"])
        except KeyError:
            rel_times[k] = v["Relaxation times"]

    
    print("Plotting")

    # n_plots = 2*len(func_list) + 2

    ncols = 2
    nrows = len(func_list) + 1

    fig, axs = plt.subplots(nrows, ncols)
    fig.suptitle("Plot of results")
    fig.tight_layout()
    fig.set_figheight(1)
    fig.set_figwidth(9)

    for row in range(nrows):
        if row < nrows - 1:
            ax = axs[row, 0]
            dict_label = func_list[row].__name__
            ax.set_title(dict_label)
            ax.errorbar(x=beta, y=values[dict_label], yerr= errors[dict_label])

            ax = axs[row, 1]
            ax.set_title(dict_label + " Relaxation Times")
            ax.errorbar(x=beta, y=rel_times[dict_label])
        elif row == nrows - 1:
            ax = axs[row, 1]
            ax.set_title("Acceptance Rates")
            ax.errorbar(x=beta, y=acceptance_rates)

            ax = axs[row, 0]
            ax.set_title("Field Magnitudes")
            ax.errorbar(x=beta, y=configs_avgs[0], label="+A", color="b")
            ax.errorbar(x=beta, y=configs_avgs[1], label="-A", color="k")
            ax.errorbar(x=beta, y=configs_avgs[2], label="+B", color='y')
            ax.errorbar(x=beta, y=configs_avgs[3], label="-B", color='r')
            ax.legend()

    plt.show()


if __name__ == "__main__":
    eqSteps = 1000
    group = 'h'
    cutoff = 9
    size = 1
    # mc_steps = 2**15
    mc_steps = 8**4
    nt = 10
    ddof = 1
    func_list = [avg_energy, squared_E]

    kappa_list = [14.5, 15.87, 17.2, 18.8, 20.5, 22.65, 25.05, 27.7, 30.8, 34.8]
    # kappa_list = [14.5, 15.1, 15.8, 16.4, 17.1, 17.8, 18.7, 19.4, 20.1, 21.1, 22.2, 23.3, 24.4, 25.8, 26.9, 28.2, 30, 31.5, 33.6, 35.8]
    plots(nt, eqSteps, mc_steps, group, kappa_list=kappa_list, cutoff=cutoff, func_list=func_list, ddof=ddof, size=size, error=10**(-8))