''' Constrained Hybrid/Hamiltonian Monte Carlo step
'''

import numpy as np
import scipy as sp

def p0_sampler(q0, C, M, q_len):
    # Pick initial unconstrained momentum
    p0_i = np.random.multivariate_normal(np.zeros(q_len), M(q0))

    # TODO: Project momentum onto constrained space

    return p0_i


def rattle_int(q0, p0, h, guidance_H, C):
    #TODO: RATTLE integrator
    return p0, q0


def hmc_step(beta, q0, q_len, M, h, L, acceptance_H, guidance_H, C):
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
    p0 = p0_sampler(q0, C, M, q_len)

    # Evolve via Hamilton's equation using RATTLE integrator
    p_list = [p0]
    q_list = [q0]
    for _ in range(L):
        p = p_list[-1]
        q = q_list[-1]
        p1, q1 = rattle_int(p, q, h, guidance_H, C)
        p_list.append(p1)
        q_list.append(q1)

    pL = p_list[-1]
    qL = q_list[-1]

    # Randomly pick number in [0, 1)
    u = np.random.rand()

    # Acceptance of proposal
    q_upd = q0
    if u <= np.min(1, np.exp(acceptance_H(p0, q0) - acceptance_H(pL, qL))):
        q_upd = qL

    return q_upd

