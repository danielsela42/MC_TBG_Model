import numpy as np
import math
import multiprocessing as mp
import matplotlib.pyplot as plt
from mc_project.utilities import LatticeStructure
from mc_project.utilities import sample_vMF
from mc_project.utilities import calculation

'''
def one_config_avg(config):
    field_average = np.mean(abs(config), axis=0)
    return [np.linalg.norm(field_average[i])**2 for i in range(4)]

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
def total_E(config, graph):
    total_energy = 0
    for i in range(len(config)):
        psi = config[i]
        total_energy += np.sum(psi[0]*psi[0]) + np.sum(psi[1]*psi[1]) + np.sum(psi[2]*psi[2]) + np.sum(psi[3]*psi[3])
    return total_energy


def avg_energy(config, graph):
    return total_E(config, graph)/len(graph)

def squared_E(config, graph):
    return avg_energy(config, graph)**2


def energy_diff(cand, curr, config, graph):
    curr_sq = curr*curr
    cand_sq = cand*cand
    energy_curr = np.sum(curr_sq[0]) + np.sum(curr_sq[1]) + np.sum(curr_sq[2]) + np.sum(curr_sq[3])
    energy_cand = np.sum(cand_sq[0]) + np.sum(cand_sq[1]) + np.sum(cand_sq[2]) + np.sum(cand_sq[3])
    energy = energy_cand - energy_curr
    return energy

def plots(nt, beta_range, eq_steps, mc_steps, group, cutoff, func_list, kappa_list=list(), ddof=0, cutoff_type='m', size=1, error=10**(-8)):
    values = dict()
    errors = dict()
    rel_times = dict()

    beta, chains, quantity_dict, acceptance_rates, all_quantities = calculation(nt, beta_range, eq_steps, mc_steps, energy_diff, group, cutoff, func_list, kappa_list=kappa_list, ddof=ddof, cutoff_type='m', size=1, error=10**(-8), log_to_consol=True)
    energies = [q[2] for q in all_quantities]

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
    fig.set_figheight(9)
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
            ax = axs[row, 0]
            ax.set_title("Acceptance Rates")
            ax.errorbar(x=beta, y=acceptance_rates)

    for i in range(len(chains)):
        fig = plt.figure(i+2)
        energy_list = energies[i]
        plt.hist(energy_list, bins=20, density=True)
        # plt.scatter(energy_list, np.exp(-1*beta[i]*np.array(energy_list)))
    plt.show()

    plt.show()


if __name__ == "__main__":
    eqSteps = 1000
    group = 'h'
    cutoff = 9
    size = 1
    # mc_steps = 2**15
    mc_steps = 8**4
    nt = 5
    ddof = 1
    func_list = [avg_energy, squared_E, total_E]
    beta_range = [5, 10]

    kappa_list = list()
    plots(nt, beta_range, eqSteps, mc_steps, group, kappa_list=kappa_list, cutoff=cutoff, func_list=func_list, ddof=ddof, size=size, error=10**(-8))