import numpy as np
import math
import multiprocessing as mp
import matplotlib.pyplot as plt
from mc_project.utilities import LatticeStructure
from mc_project.utilities import sample_vMF
from mc_project.utilities import calculation
import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs


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
    return [np.mean(avg_Ap), np.mean(avg_Am), np.mean(avg_Bp), np.mean(avg_Bm)]


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


def energy_diff(cand, curr, config, graph):
    curr_sq = curr*curr
    cand_sq = cand*cand
    energy_curr = np.sum(curr_sq[0]) - np.sum(curr_sq[1]) + np.sum(curr_sq[2]) - np.sum(curr_sq[3])
    energy_cand = np.sum(cand_sq[0]) - np.sum(cand_sq[1]) + np.sum(cand_sq[2]) - np.sum(cand_sq[3])
    energy = energy_cand - energy_curr
    return energy


def plots(nt, beta_range, eq_steps, mc_steps, group, cutoff, func_list, kappa_list=list(), ddof=0, cutoff_type='m', size=1, error=10**(-8)):
    values = dict()
    errors = dict()
    rel_times = dict()

    beta, chains, quantity_dict, acceptance_rates, all_quantities = calculation(nt, beta_range, eq_steps, mc_steps, energy_diff, group, cutoff, func_list, kappa_list=kappa_list, ddof=ddof,cutoff_type='m', size=1, error=10**(-8), log_to_consol=True)
    energies = [q[2] for q in all_quantities]

    configs_avgs = [list(), list(), list(), list()]
    for chain in chains:
        for i in range(4):
            configs_avgs[i].append(average_comp_mag(chain)[i])

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

    # dens_u = sm.nonparametric.KDEMultivariate(data=energies, var_type='cc', bw='normal_reference')

    
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

    for i in range(len(chains)):
        fig = plt.figure(i+2)
        energy_list = energies[i]
        plt.hist(energy_list, bins=20, density=True)
        # plt.scatter(energy_list, np.exp(-1*beta[i]*np.array(energy_list)))
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

    # kappa_list = [14.5, 15.87, 17.2, 18.8, 20.5, 22.65, 25.05, 27.7, 30.8, 34.8]
    # kappa_list = [14.5, 15.1, 15.8, 16.4, 17.1, 17.8, 18.7, 19.4, 20.1, 21.1, 22.2, 23.3, 24.4, 25.8, 26.9, 28.2, 30, 31.5, 33.6, 35.8]
    
    #kappa_list = [3, 4, 5, 6, 7, 8, 11, 12, 13, 14.5]
    #kappa_list = [0.3, 0.6, 1, 2, 3]
    kappa_list = list()
    plots(nt, beta_range, eqSteps, mc_steps, group, kappa_list=kappa_list, cutoff=cutoff, func_list=func_list, ddof=ddof, size=size, error=10**(-8))