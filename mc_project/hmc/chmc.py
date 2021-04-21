''' Constrained Hybrid/Hamiltonian Monte Carlo step
'''

import numpy as np
from scipy import optimize

def p0_sampler(size, q0, C, M, M_inv, q_len):
    # Pick initial unconstrained momentum
    p0_initial = np.random.multivariate_normal(np.zeros(q_len), M(q0))
    const = C(q0)
    M_inv_q0 = M_inv(q0)
    prod = np.dot(const, M_inv_q0)

    p0_proj = np.zeros(q_len)

    for i in range(size):
        p0_i = p0_initial[i:i+8]
        s = prod[i][i:i+8]
        s_sq_mag = sum(s*s)
        s_inv_sq = 1/s_sq_mag
        proj = p0_i - s_inv_sq * np.dot(s, p0_i)
        p0_proj[i:i+1] = proj

    return p0_proj


def rattle_eqs_1(size, q0, p0, h, gH_q, gH_p, C_transposed, constraints):
    def F(x):
        lamb = x[0]
        q1 = x[1:(size+1)]
        phalf = x[(size+1):]
        F1 = phalf  - p0 +(h/2)*(gH_q(phalf, q0) + C_transposed(q0)*lamb)
        F2 = q1 - q0 - (h/2)*(gH_p(phalf, q0) + gH_p(phalf, q1))
        F3 = constraints(q1)
        result = np.concatenate(F1, F2, F3)
        return result
    return F

def rattle_eqs_2(size, q1, phalf, q0, h, gH_q, gH_p, C, C_transposed, constraints):
    def F(x):
        mu = x[0]
        p1 = x[1:(size+1)]
        F1 = p1  - phalf +(h/2)*(gH_q(phalf, q0) + C_transposed(q0)*mu)
        F2 = C(q1)*gH_p(p1, q1)
        result = np.concatenate(F1, F2)
        return result
    return F


def hmc_step(lattice_size, beta, q0, q_len, M, M_inv, h, L, acceptance_H, guidance_H, gH_p, gH_q, constraints, C, C_transposed, error):
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
    p0 = p0_sampler(lattice_size, q0, C, M, M_inv, q_len)

    # Evolve via Hamilton's equation using RATTLE integrator
    p_list = [p0]
    q_list = [q0]
    for i in range(L):
        p = p_list[-1]
        q = q_list[-1]

        # Solve first set of equations
        guess1 = np.zeros(2*q_len + 1)
        F1 = rattle_eqs_1(q_len, q, p, h, gH_q, gH_p, C_transposed, constraints)
        x1 = optimize.newton_krylov(F1, guess1)

        # Solver second set
        guess2 = guess1 = np.zeros(2*q_len + 1)
        F2 = rattle_eqs_1(q_len, q, p, h, gH_q, gH_p, C_transposed, constraints)
        x2 = optimize.newton_krylov(F2, guess2)

        # Verify results
        if abs(F1(x1)) < error or abs(F2(x2)) < error:
            raise Exception("ERROR: Non-convergent solutions to Hamilton's equation at step {}.".format(i))

        # Get desired results
        q1 = x1[1:q_len+1]
        p1 = x2[1:q_len+1]
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

