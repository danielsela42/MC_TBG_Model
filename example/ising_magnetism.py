import numpy as np
import math
import matplotlib.pyplot as plt

def init(L):
        state = 2 * np.random.randint(2, size=(L,L)) - 1
        return state
        
def E_dimensionless(config, L):
    total_energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = config[(i+1)%L, j] + config[i, (j+1)%L] + config[(i-1)%L, j] + config[i, (j-1)%L]
            total_energy += -nb * S
    return (total_energy/4)


def magnetization(config):
    Mag = np.sum(config)
    return Mag


def MC_step(config, beta, sequential=False):
    '''Monte Carlo move using Metropolis algorithm '''
    L = len(config)
    for i in range(L):
        for j in range(L):
            if sequential:
                a = i
                b = j
            else:
                a = np.random.randint(0, L) # looping over i & j therefore use a & b
                b = np.random.randint(0, L)
            sigma =  config[a, b]
            neighbors = config[(a+1)%L, b] + config[a, (b+1)%L] + config[(a-1)%L, b] + config[a, (b-1)%L]
            del_E = 2*sigma*neighbors
            # if del_E < 0:
            #    sigma *= -1
            if np.random.rand() < 1/(1+np.exp(del_E*beta)):
                sigma *= -1
            config[a, b] = sigma
    return config


def correlation(meas, initial):
    x_list = [i for i in range(initial, len(meas))]
    corr = list()
    denom = np.mean([val**2 for val in meas])-np.mean(meas)**2
    for t in range(len(meas)):
        corr.append((np.mean(meas[0]*meas[t])-np.mean(meas)**2)/denom)
    
    x = np.array(x_list)
    y= np.array([corr[i] for i in range(initial, len(meas))])
    idx = np.isfinite(x) & np.isfinite(np.log(y))
    res = np.polyfit(x[idx], np.log(y[idx]), 1)
    #res = np.polyfit(np.exp(x), y, 1)
    rel_time = -1/res[0]
    return corr, rel_time, np.exp(res[1])


def calcul_energy_mag_C_X(config, L, eqSteps, err_runs, sequential, initial=10):
        
    # L is the length of the lattice
        
    nt      = 100         #  number of temperature points
    mcSteps = 1000
    
    T_c = 2/math.log(1 + math.sqrt(2))
        
    # the number of MC sweeps for equilibrium should be at least equal to the number of MC sweeps for equilibrium

    # initialization of all variables
    T = np.linspace(1., 7., nt)
    E, M = np.zeros(nt), np.zeros(nt)
    # C, X = np.zeros(nt), np.zeros(nt)
    C_theoric, M_theoric = np.zeros(nt), np.zeros(nt)
    # delta_E,delta_M, delta_C, delta_X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
    # n1 = 1.0/(mcSteps*L*L)
    # n2 = 1.0/(mcSteps*mcSteps*L*L)    # n1 and n2 will be use to compute the mean value and the # by sites
        # of E and E^2
        
    Energies = []
    Magnetizations = []
    SpecificHeats = []
    Susceptibilities = []
    delEnergies = []
    delMagnetizations = []
    delSpecificHeats = []
    delSusceptibilities = []
    
    for t in range(nt):
        # initialize total energy and mag
        beta = 1./T[t]

        print("Beginning temp step: ", t+1)
        # evolve the system to equilibrium
        for _ in range(eqSteps):
            MC_step(config, beta, sequential)
        # list of ten macroscopic properties
        Ez = []; Cz = []; Mz = []; Xz = [] 
        for _ in range(err_runs):
            E = E_squared = M = M_squared = 0
            for _ in range(mcSteps):
                MC_step(config, beta, sequential)           
                energy = E_dimensionless(config,L) # calculate the energy at time stamp
                mag = abs(magnetization(config)) # calculate the abs total mag. at time stamp

                # sum up total energy and mag after each time steps

                E += energy
                E_squared += energy**2
                M += mag
                M_squared += mag**2


            # mean (divide by total time steps)

            E_mean = E/mcSteps
            E_squared_mean = E_squared/mcSteps
            M_mean = M/mcSteps
            M_squared_mean = M_squared/mcSteps

            # calculate macroscopic properties (divide by # sites) and append

            Energy = E_mean/(L**2)
            SpecificHeat = beta**2 * (E_squared_mean - E_mean**2)/L**2
            Magnetization = M_mean/L**2
            Susceptibility = beta * (M_squared_mean - M_mean**2)/(L**2)

            Ez.append(Energy); Cz.append(SpecificHeat); Mz.append(Magnetization); Xz.append(Susceptibility)

        Energy = np.mean(Ez)
        Energies.append(Energy)
        delEnergy = np.std(Ez)
        delEnergies.append(float(delEnergy))
        
        Magnetization = np.mean(Mz)
        Magnetizations.append(Magnetization)
        delMagnetization = np.std(Mz)
        delMagnetizations.append(delMagnetization)

        
        SpecificHeat = np.mean(Cz)
        SpecificHeats.append(SpecificHeat)
        delSpecificHeat = np.std(Cz)
        delSpecificHeats.append(delSpecificHeat)

        Susceptibility = np.mean(Xz)
        delSusceptibility = np.std(Xz)        
        Susceptibilities.append(Susceptibility)
        delSusceptibilities.append(delSusceptibility)
        

        if T[t] - T_c >= 0:
            C_theoric[t] = 0
        else:
            M_theoric[t] = pow(1 - pow(np.sinh(2*beta), -4),1/8)
        
        coeff = math.log(1 + math.sqrt(2))
        if T[t] - T_c >= 0:
            C_theoric[t] = 0
        else: 
            C_theoric[t] = (2.0/np.pi) * (coeff**2) * (-math.log(1-T[t]/T_c) + math.log(1.0/coeff) - (1 + np.pi/4)) 
    
    energy_corr, ener_rel, A_ener = correlation(Energies, initial)
    mag_corr, mag_rel, A_mag = correlation(Magnetizations, initial)

    results = [T,Energies,Magnetizations,SpecificHeats,Susceptibilities, delEnergies, delMagnetizations,
                M_theoric, C_theoric, delSpecificHeats, delSusceptibilities, energy_corr, mag_corr, ener_rel, mag_rel, A_ener, A_mag]

    return results


def plots(L, eqSteps, err_runs, sequential=False, initial=10):

    config = init(L)

    print("Performing calculation")
    T,Energies,Magnetizations,SpecificHeats,Susceptibilities, delEnergies, delMagnetizations,_, _, delSpecificHeats, delSusceptibilities, energy_corr, mag_corr, ener_rel, mag_rel, A_ener, A_mag = calcul_energy_mag_C_X(config, L, eqSteps, err_runs, sequential, initial)
    print("Plotting")

    shift_T = [i for i in range(len(T))]

    fig, axs = plt.subplots(3, 2)
    fig.tight_layout()

    ax = axs[0, 0]
    ax.errorbar(x=T, y=Energies, yerr=delEnergies)
    ax.set_title('Energies')
    ax = axs[0, 1]
    ax.errorbar(x=T, y=Magnetizations, yerr=delMagnetizations)
    ax.set_title('Magnetizations')
    ax = axs[1, 0]
    ax.errorbar(x=T, y=SpecificHeats, yerr=delSpecificHeats)
    ax.set_title('Specific Heats')
    ax = axs[1, 1]
    ax.errorbar(x=T, y=Susceptibilities, yerr=delSusceptibilities)
    ax.set_title('Susceptibilities')

    
    ax = axs[2, 0]
    ye = A_ener*np.exp(np.array(shift_T)*(-1/ener_rel))
    ide = [(val < 10) for val in ye]
    ax.errorbar(x=shift_T, y=energy_corr)
    ax.errorbar(x=np.array(shift_T)[ide], y=ye[ide])
    ax.set_title('Energy Correlation')


    ym = A_mag*np.exp(np.array(shift_T)*(-1/mag_rel))
    idm = [(val < 10) for val in ym]
    ax = axs[2, 1]
    ax.errorbar(x=shift_T, y=mag_corr)
    ax.errorbar(x=np.array(shift_T)[idm], y=ym[idm])
    ax.set_title('Magnetization Correlation')

    # fig.suptitle('MC Simulation Plots')

    plt.show()
    print("Energy relaxation: ", ener_rel, ", Magnetization relaxation: ", mag_rel)



if __name__ == "__main__":
    L = 4
    eqSteps = 50
    err_runs = 50
    sequential = False
    initial = 20
    plots(L, eqSteps, err_runs, sequential, initial)