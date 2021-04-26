import numpy as np
import multiprocessing as mp
from mc_project.utilities import LatticeStructure
from mc_project.utilities import sample_vMF

def init(n_points):
    # create config
    config = np.random.normal(0, 1, size=(n_points, 4, 2))

    # Fix phase
    for i in range(n_points):
        norm = np.sqrt(sum(sum(config[i]*config[i])))
        config[i] *= 1/norm
    return config

def MC_step(energy_diff, n_points, config, graph, beta, n_accepted_list):
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
        neighbors = graph[rand_pos][1]
        del_E = energy_diff(cand, curr, neighbors, config)
        if del_E < 0:
            upd = cand
            n_accepted_list[0] += 1
        elif np.random.rand() < np.exp(-del_E*beta):
            upd = cand
            n_accepted_list[0] += 1
        config[rand_pos] = upd
    return config


def MC_step_vMF(energy_diff, n_points, config, graph, beta, kappa, n_accepted_list):
    '''Monte Carlo move using Metropolis algorithm '''
    for i in range(n_points):
        # rand_pos = np.random.randint(0, n_points)
        rand_pos = i
        curr = config[rand_pos]

        # Choose and normalize candidate
        cand = np.reshape(sample_vMF(np.concatenate(curr, axis=None), kappa, num_samples=1), (4, 2))

        # Accept or deny candidate
        upd = curr
        neighbors = graph[rand_pos][1]
        del_E = energy_diff(cand, curr, neighbors, config)
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


def mcmc_sampling_uniform(energy_diff, eq_steps, mc_steps, beta, n_points, graph):

    # Initialize configuration
    config = init(n_points)

    configs = list()

    # evolve the system to equilibrium
    n_accepted_list = [0]
    for _ in range(eq_steps):
        MC_step(energy_diff, n_points, config, graph, beta, n_accepted_list)

    # Perform sweeps
    n_accepted_list = [0]
    for _ in range(mc_steps):
        MC_step(energy_diff, n_points, config, graph, beta, n_accepted_list)
        configs.append(config.copy())
    n_accepted = n_accepted_list[0]

    return configs, n_accepted/(mc_steps*n_points)


def mcmc_sampling_vMF(energy_diff, eq_steps, mc_steps, beta, kappa, n_points, graph):

    # Initialize configuration
    config = init(n_points)

    configs = list()

    # evolve the system to equilibrium
    n_accepted_list = [0]
    for _ in range(eq_steps):
        MC_step_vMF(energy_diff, n_points, config, graph, beta, kappa, n_accepted_list)

    # Perform sweeps
    n_accepted_list = [0]
    for _ in range(mc_steps):
        MC_step_vMF(energy_diff, n_points, config, graph, beta, kappa, n_accepted_list)
        configs.append(config.copy())
    n_accepted = n_accepted_list[0]

    return configs, n_accepted/(mc_steps*n_points)


def mcmc_sampling_vMF_mp(args):
    energy_diff, eq_steps, mc_steps, beta, kappa, n_points, graph = args

    # Initialize configuration
    config = init(n_points)

    configs = list()

    # evolve the system to equilibrium
    n_accepted_list = [0]
    for _ in range(eq_steps):
        MC_step_vMF(energy_diff, n_points, config, graph, beta, kappa, n_accepted_list)

    # Perform sweeps
    n_accepted_list = [0]
    for _ in range(mc_steps):
        MC_step_vMF(energy_diff, n_points, config, graph, beta, kappa, n_accepted_list)
        configs.append(config.copy())
    n_accepted = n_accepted_list[0]

    return configs, n_accepted/(mc_steps*n_points), kappa


def burn_in(arg):
    energy_diff, eq_steps, beta, n_points, graph = arg

    # Initialize configuration
    config = init(n_points)

    chain = list()

    # evolve the system to equilibrium
    n_accepted_list = [0]
    for _ in range(eq_steps):
        config = MC_step(energy_diff, n_points, config, graph, beta, n_accepted_list)
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


def verify_convergence(energy_diff, n_processes, n_chains, eq_steps, beta, n_points, graph):
    arg = (energy_diff, eq_steps, beta, n_points, graph)

    args_list = [list() for i in range(n_processes)]

    for i in range(n_chains):
        args_list[i % n_processes].append(arg)
    
    # print("Creating process pool")

    # Create pool
    pool = mp.Pool(processes=n_processes)
    results = pool.map_async(multi_chains, args_list)
    pool.close() # Close pool
    # print("\t Process pool closed")
    pool.join() # Join pools
    # print("\t Pools joined")

    # Get results
    chains = list()
    for res in results.get():
        chains.extend(res)
    
    # Get colleciton of chains
    parallel_chains = np.array(chains)
    # Get collection of coordinates in each chain
    coords_chains = np.array([np.transpose(chain) for chain in parallel_chains])
    # Get the average of each coordinate per chain
    chain_coord_means = np.array([[np.mean(coord_set[i]) for i in range(n_points*8)] for coord_set in coords_chains])
    # Get the overall mean of each coordinate
    overall_mean = np.array([np.mean(np.transpose(chain_coord_means)[k]) for k in range(n_points*8)])

    # print("\t Getting covariance matrices")
    #Initialize 
    between_chain_covar = np.zeros((n_points*8, n_points*8))
    within_chain_covar = np.zeros((n_points*8, n_points*8))
    for i in range(n_points*8):
        for j in range(n_points*8):
            if i > j: continue
            between_chain_covar[i, j] = (1/(n_chains-1)) * np.sum([(means[i] - overall_mean[i])*(means[j] - overall_mean[j]) for means in chain_coord_means])
            within_arr = [np.sum((parallel_chains[k, :, i] - chain_coord_means[k, i])*(parallel_chains[k, :, j] - chain_coord_means[k, j])/(n_chains*(eq_steps - 1))) for k in range(n_chains)]
            within_chain_covar[i, j] = np.sum(within_arr)
            if i < j:
                between_chain_covar[j, i] = between_chain_covar[i, j]
                within_chain_covar[j, i] = within_chain_covar[i, j]

    # print("\t Calculating R matrix")
    covar = ((eq_steps - 1)/eq_steps)*within_chain_covar + between_chain_covar
    w_inv = np.linalg.inv(within_chain_covar)
    eigen_matr = np.dot(w_inv, covar)/eq_steps
    
    # print("\t Getting eigenvalue")
    lambda1 = np.amax(np.linalg.eigvals(eigen_matr))

    R = (eq_steps - 1)/eq_steps + (1 + 1/n_chains)*lambda1

    print("\t R value:", R)
    if abs(R.real - 1) < 0.1 and abs(R.imag) < 0.01:
        return True
    else:
        return False


def adaptive_sampling(energy_diff, eq_steps, mc_steps, beta, n_points, graph, n_processes=12, max_iters=3):
    kappa_list_init = [1] + list(np.linspace(5, 50, n_processes-1))

    args_list = [(energy_diff, eq_steps, 2*eq_steps, beta, kappa, n_points, graph) for kappa in kappa_list_init]

    # Create pool
    pool = mp.Pool(processes=n_processes)
    results = pool.map_async(mcmc_sampling_vMF_mp, args_list)
    pool.close() # Close pool
    # print("\t Process pool closed")
    pool.join() # Join pools
    # print("\t Pools joined")

    kappa_acceptance_rate = list()
    for result in results.get():
        kappa_acceptance_rate.append(result)
    
    kappa_acceptance_rate.sort(key=lambda x:x[2])
    # chains = [tup[0] for tup in kappa_acceptance_rate]
    acceptance_rates = [tup[1] for tup in kappa_acceptance_rate]

    for i in range(n_processes):
        if 0.22 < acceptance_rates[i] < 0.25:
            return mcmc_sampling_vMF(energy_diff, eq_steps, mc_steps, beta, kappa_list_init[i], n_points, graph)
    
    if np.all([acceptance_rates[i] > 0.6 for i in range(n_processes)]):
        configs, acceptance_rate = mcmc_sampling_uniform(energy_diff, eq_steps, mc_steps, beta, n_points, graph)
        if acceptance_rate >= 0.6:
            return configs, acceptance_rate
        else:
            print(acceptance_rate)
            raise Exception("Failed to find good concentration parameter: problem near uniformity")

    kappa_guess = np.interp(0.234, acceptance_rates, kappa_list_init)

    if kappa_guess < 1.5:
        configs, acceptance_rate = mcmc_sampling_uniform(energy_diff, eq_steps, mc_steps, beta, n_points, graph)
        if acceptance_rate >= 0.6:
            return configs, acceptance_rate
        else:
            print(acceptance_rate)
            raise Exception("Failed to find good concentration parameter: problem near uniformity")
    else:
        kappa_list = kappa_list_init
        count = 0
        configs, acceptance_rate = mcmc_sampling_vMF(energy_diff, eq_steps, eq_steps, beta, kappa_guess, n_points, graph)
        acceptance_rates.append(acceptance_rate)
        kappa_list.append(kappa_guess)
        kappa_guess = np.interp(0.234, acceptance_rates, kappa_list)
        while acceptance_rate > 0.25 or acceptance_rate < 0.20:
            configs, acceptance_rate = mcmc_sampling_vMF(energy_diff, eq_steps, eq_steps, beta, kappa_guess, n_points, graph)
            acceptance_rates.append(acceptance_rate)
            kappa_list.append(kappa_guess)
            kappa_guess = np.interp(0.234, acceptance_rates, kappa_list)
            count += 1
            if count > max_iters: raise Exception("Reached max_iters in adaptive sampling: Last acceptance rate was {}".format(acceptance_rate))
        else:
            return mcmc_sampling_vMF(energy_diff, eq_steps, mc_steps, beta, kappa_guess, n_points, graph)


def calculation(nt, beta_range, eq_steps, mc_steps, energy_diff, group, cutoff, func_list, convergence=False, n_chains=12, kappa_list=list(), n_processes=12, ddof=0, cutoff_type='m', size=1, error=10**(-8), log_to_consol=False):
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

    # Get coupling constant (beta)
    beta_T = np.linspace(beta_range[0], beta_range[1], nt)

    quantity_dict = dict()

    acceptance_rates =list()

    chains = list()

    for t in range(nt):
        # initialize temperature
        beta = beta_T[t]

        # Verify convergence
        if convergence:
            R_test = verify_convergence(energy_diff, n_processes, n_chains, eq_steps, beta, n_points, graph)
            if log_to_consol: print("\t Convergence: ", R_test)

        if log_to_consol: print("Beginning temp step: ", t+1)
        # evolve the system to equilibrium
        if not kappa_list:
            configs, acceptance_rate = adaptive_sampling(energy_diff, eq_steps, mc_steps, beta, n_points, graph, n_processes=n_processes)
        elif kappa_list[t] == 0:
            configs, acceptance_rate = mcmc_sampling_uniform(energy_diff, eq_steps, mc_steps, beta, n_points, graph)
        else:
            configs, acceptance_rate = mcmc_sampling_vMF(energy_diff, eq_steps, mc_steps, beta, kappa_list[t], n_points, graph)
        if log_to_consol: print("\t The acceptance rate is", acceptance_rate)
        acceptance_rates.append(acceptance_rate)

        chains.append(configs)

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
        

    results = [beta_T, chains, quantity_dict, acceptance_rates]

    return results