''' Constrained Hybrid/Hamiltonian Monte Carlo step
'''

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from mc_project.utilities import LatticeStructure
import time


def energy(q, n_points, graph):
    total_energy = 0
    for i in range(n_points):
        psi = q[8*i:8*i+9]
        for ind, _ in graph[i][1]:
            diff = psi - q[8*ind:8*ind+9]
            total_energy += sum(diff*diff)
    return total_energy


def mass_matr(q):
    return np.identity(len(q))

def mass_inv(q_len, mass):
    inv = np.zeros((q_len, q_len))
    for i in range(q_len):
        inv[i, i] = 1/mass[i, i]
    return inv

def constraints(q, n_points):
    return np.array([np.dot(q[8*i:8*i+9], q[8*i:8*i+9]) for i in range(n_points)])

def C(q, n_points):
    F = np.zeros((n_points, 8*n_points))
    for i in range(n_points):
        F[i][8*i:8*i+9] = 2*q[8*i:8*i+9]
    return F

def C_transposed(q, Cq):
    transposed = np.transpose(Cq)
    return transposed

def acceptance_H(beta, p, q, q_len, n_points, graph):
    mass = mass_matr(q)
    return 0.5*np.dot(p, np.dot(mass_inv(q_len, mass), p)) + 0.5*np.log(np.linalg.norm(mass)) + beta*energy(q, n_points, graph)

def guidance_H(beta, p, q, q_len, n_points, graph):
    return acceptance_H(beta, p, q, q_len, n_points, graph)

def gH_p(m_inv, p):
    return np.dot(m_inv, p)

def gH_q(p, q):
    return 0


def p0_sampler(n_points, q0, q_len):
    # Pick initial unconstrained momentum
    mass = mass_matr(q0)
    p0_initial = np.random.multivariate_normal(np.zeros(q_len), mass)
    const = C(q0, n_points)
    M_inv_q0 = mass_inv(q_len, mass)
    prod = np.dot(const, M_inv_q0)

    p0_proj = np.zeros(q_len)

    for i in range(n_points):
        p0_i = p0_initial[8*i:8*i+9]
        s = prod[i][8*i:8*i+9]
        s_sq_mag = sum(s*s)
        s_inv_sq = 1/s_sq_mag
        proj = p0_i - s_inv_sq * np.dot(s, p0_i)
        p0_proj[8*i:8*i+9] = proj

    return p0_proj


def rattle_eqs_1(n_points, q_len, q0, p0, h):
    def F(x):
        q1 = x[0:q_len]
        phalf = x[(q_len):(2*q_len)]
        lamb = x[(2*q_len):]
        Cq0 = C(q0, n_points)
        F1 = phalf  - p0 +(h/2)*(gH_q(phalf, q0) + np.dot(C_transposed(q0, Cq0), lamb))
        F2 = q1 - q0 - (h/2)*(gH_p(mass_inv(q_len, mass_matr(phalf)), phalf) + gH_p(phalf, q1))
        F3 = constraints(q1, n_points)
        result = np.concatenate((F1, F2, F3))
        return result
    return F

def rattle_eqs_2(n_points, q_len, q1, phalf, q0, h, m_inv):
    def F(x):
        p1 = x[0:q_len]
        mu = x[q_len:]
        Cq0 = C(q0, n_points)
        F1 = p1  - phalf +(h/2)*(gH_q(phalf, q1) + C_transposed(q0, Cq0)*mu)
        F2 = np.dot(C(q1, n_points), gH_p(m_inv, p1))
        result = np.concatenate((F1, F2))
        return result
    return F


def hmc_step(lattice_size, beta, q0, q_len, h, L, graph, error):
    ''' CMHMC step

    input: beta - 1/T unitless
           q0 - initial parameters as vector
           q_len - # of parameters
           M - mass matrix
           h - step size
           L - integration length
           acceptance_H - acceptance Hamiltonian
           guidance_H - guidance Hamiltonian
           C - jacobian of constraint (function with input parameters q)
    '''
    # Sample momentmum given constraints
    p0 = p0_sampler(lattice_size, q0, q_len)

    # Evolve via Hamilton's equation using RATTLE integrator
    p_list = [p0]
    q_list = [q0]
    for i in range(L):
        p = p_list[-1]
        q = q_list[-1]

        # Solve first set of equations
        guess1 = np.zeros(2*q_len+lattice_size)
        guess1[0:q_len] = q
        guess1[q_len:2*q_len] = p
        print("beginning first eq")
        t0 = time.time()
        F1 = rattle_eqs_1(lattice_size, q_len, q, p, h)
        x1 = optimize.newton_krylov(F1, guess1)
        t1 = time.time()
        print("Sovled first in ", t1 - t0)

        # Get results
        q1 = x1[0:q_len]
        phalf = x1[(q_len):(2*q_len)]

        # Solver second set
        guess2 = np.zeros(q_len+lattice_size)
        guess2[0:q_len] = phalf
        m_inv = mass_inv(q_len, mass_matr(phalf))
        t0 = time.time()
        F2 = rattle_eqs_2(lattice_size, q_len, q1, phalf, h, q0, m_inv)
        x2 = optimize.newton_krylov(F2, guess2)
        t1 = time.time()

        print("Sovled second in ", t1 - t0)

        # Verify results
        if abs(F1(x1)) < error or abs(F2(x2)) < error:
            raise Exception("ERROR: Non-convergent solutions to Hamilton's equation at step {}.".format(i))

        # Get desired results
        p1 = x2[0:q_len]
        p_list.append(p1)
        q_list.append(q1)

    pL = p_list[-1]
    qL = q_list[-1]

    # Randomly pick number in [0, 1)
    u = np.random.rand()

    # Acceptance of proposal
    q_upd = q0
    if u <= np.min(1, np.exp(acceptance_H(beta, p0, q0, q_len, lattice_size, graph) - acceptance_H(beta, pL, qL, q_len, lattice_size, graph))):
        q_upd = qL

    return q_upd


def init(n_points):
    # create config
    config = np.random.uniform(low=-1.0, high=1.0, size=(n_points, 4, 2))

    # Fix phase
    for i in range(n_points):
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


def calculation(nt, eq_steps, mc_steps, step_size, L, group, cutoff, func_list, ddof=0, cutoff_type='m', size=1, error=10**(-3)):
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
    q0 = np.reshape(config, 8*n_points)
    q_len = 8*n_points

    # Get temperature points
    T = np.linspace(1., 2., nt)

    quantity_dict = dict()

    acceptance_rates =list()
    mc_total = n_points*mc_steps

    beta_T = [1/T[nt - t - 1] for t in range(nt)]

    for t in range(nt):
        # initialize temperature
        beta = beta_T[t]

        print("Beginning temp step: ", t+1)
        # evolve the system to equilibrium
        n_accepted_list = [0]
        for _ in range(eq_steps):
            q = hmc_step(n_points, beta, q0, q_len, step_size, L, graph, error)

        n_accepted = n_accepted_list[0]
        print(n_accepted/(eq_steps*n_points))


        # Perform sweeps
        quantities = list()
        n_accepted_list = [0]
        for _ in range(mc_steps):
            q = hmc_step(n_points, beta, q, q_len, step_size, L, graph, error)

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

        n_accepted = n_accepted_list[0]
        acceptance_rate = n_accepted/mc_total
        acceptance_rates.append(acceptance_rate)
        #print(acceptance_rate)
        
    
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
        

    results = [beta_T, quantity_dict, acceptance_rates]

    return results


def plots(nt, eq_steps, mc_steps, step_size, L, group, cutoff, func_list, ddof=0, cutoff_type='m', size=1, error=10**(-8)):
    values = dict()
    errors = dict()
    rel_times = dict()
    rates = list()

    beta, quantity_dict, acceptance_rates = calculation(nt, eq_steps, mc_steps, step_size, L, group, cutoff, func_list, ddof=ddof,cutoff_type='m', size=1, error=10**(-8))
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
    
    rates = acceptance_rates

    
    print("Plotting")

    ncols = 2
    nrows = int(len(func_list)/2)

    fig, axs = plt.subplots(nrows, ncols)
    fig.suptitle("Plots for kappa=" + ','.join([str(elem) for elem in kappa_list]))
    fig.tight_layout()

    count = 0
    for row in range(nrows):
        for col in range(ncols):
            if count + 1 > ncols*len(func_list)/2:
                break
            try:
                ax = axs[row, col]
            except IndexError:
                ax = axs[col]
    
            dict_label = func_list[count].__name__
            ax.set_title(dict_label)
            ax.errorbar(x=beta, y=values[dict_label], yerr= errors[dict_label], label=dict_label)
            ax.legend()
            count += 1

    plt.figure(2)
    plt.plot(beta, rates, label="Acceptance Rates")

    plt.show()


if __name__ == "__main__":
    step_size = 0.5
    L = 10

    eqSteps = 1000
    group = 'h'
    cutoff = 9
    size = 1
    # mc_steps = 2**15
    mc_steps = 8**3
    nt = 5
    ddof = 1
    func_list = [avg_energy, squared_E]

    kappa_list = [0, -1, -0.1, -0.1, -0.3]
    plots(nt, eqSteps, mc_steps, step_size, L, group, cutoff=cutoff, func_list=func_list, ddof=ddof, size=size, error=10**(-8))

