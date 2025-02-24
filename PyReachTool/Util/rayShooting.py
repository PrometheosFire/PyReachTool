import numpy as np
import cvxpy as cp
from itertools import product

global_cube_vertices = {}

def generate_cube_vertices(m):
    """ Generate all possible vertices of an m-dimensional hypercube. """
    return np.array(list(product([-1, 1], repeat=m)))

def rayShooting(V, constraints, p, m, center):
    #global global_cube_vertices
    
    first_set = V.shape[1] == 0 if V is not None else True
    
    if m not in global_cube_vertices and first_set:
        global_cube_vertices = np.array(list(product([-1, 1], repeat=m)))
    
    nb_samples = max(400, 200 + m + 2**m)
    
    n = p.shape[0]
    VM = np.zeros((m, nb_samples))
    Vm = np.zeros((m, nb_samples))
    
    d = cp.Variable(m)
    
    if first_set:
        getV = cp.Problem(cp.Maximize(cp.matmul(cp.hstack([d, np.zeros(n-m)]), p)), constraints)
    
    k = cp.Variable()
    getRay = cp.Problem(cp.Maximize(cp.matmul(cp.hstack([d, np.zeros(n-m)]), p)), constraints + [p[:m] == k * d + center])
    
    for i in range(nb_samples):
        if first_set and i < m:
            direction = np.eye(m)[i]
            VM[:, i] = getV.solve() if getV.solve() is not None else np.nan
            Vm[:, i] = getV.solve() if getV.solve() is not None else np.nan
        elif first_set and i >= m and i < m + 2**m:
            direction = global_cube_vertices[m][i - m]
            VM[:, i] = getV.solve() if getV.solve() is not None else np.nan
            Vm[:, i] = getV.solve() if getV.solve() is not None else np.nan
        else:
            direction = -1 + 2 * np.random.rand(m)
            try:
                VM[:, i] = getRay.solve() if getRay.solve() is not None else np.nan
                Vm[:, i] = getRay.solve() if getRay.solve() is not None else np.nan
            except Exception as e:
                print("Solver exception:", e, "Direction:", direction)
    
    V = np.hstack([V, VM, Vm]) if V is not None else np.hstack([VM, Vm])
    return V
