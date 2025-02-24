import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag
from VolumeRayShootting import VolumeRayShooting

class BasicSet:
    def __init__(self, G, c, A, b):
        self.G = G
        self.c = c
        self.A = A
        self.b = b
    

### set operations ###

def CZcvxHull(X, Y):
    """
    Computes the convex hull of two constrained zonotopes.
    
    Parameters:
        X (Constrained Zonotope): Constrained Zonotope object.
        Y (Constrained Zonotope): Constrained Zonotope object.
    
    Returns:
        Z (Constrained Zonotope): Constrained Zonotope that Z = {z = lambda*x+(1-lambda)*y: x \in X, y \in Y}
    """
    n1 = X.G.shape[1]
    n2 = Y.G.shape[1]
    
    A31 = np.vstack([np.eye(n1), -np.eye(n1), np.zeros((2 * n2, n1))])
    A32 = np.vstack([np.zeros((2 * n1, n2)), np.eye(n2), -np.eye(n2)])
    A30 = np.concatenate([-0.5 * np.ones(2 * n1), 0.5 * np.ones(2 * n2)])
    
    G = np.hstack([
        X.G, Y.G, (X.c - Y.c) / 2, np.zeros((X.G.shape[0], 2 * (n1 + n2)))
    ])
    c = (X.c + Y.c) / 2
    
    A_block = block_diag(X.A, Y.A)
    A_zeros = np.zeros((A_block.shape[0], 2 * (n1 + n2)))
    
    A = np.vstack([
        np.hstack([A_block, np.vstack([-X.b / 2, Y.b / 2]), A_zeros]),
        np.hstack([A31, A32, A30[:, None], np.eye(2 * (n1 + n2))])
    ])
    
    b = np.concatenate([
        0.5 * X.b, 0.5 * Y.b, -0.5 * np.ones(2 * (n1 + n2))
    ])
    
    return G, c, A, b


def CZIntersect(C, X, Y):
    """
    CZIntersect - Function that calculates the points in X such that if multiplied by matrix C would be in Y.
    
    Parameters:
        X (Constrained Zonotope): Constrained Zonotope object.
        Y (Constrained Zonotope): Constrained Zonotope object.
        C (numpy.ndarray): Matrix used in the intersection computation.
    
    Returns:
        nextX (Constrained Zonotope): Constrained Zonotope that Z = {z: z \in X,  Cz \in Y}
    """

    G = np.hstack([X['G'], np.zeros((X.G.shape[0], Y.G.shape[1]))])
    c = X.c
    
    A = np.vstack([
        block_diag(X.A, Y.A),
        np.hstack([C @ X.G, -Y.G])
    ])
    
    b = np.concatenate([X.b, Y.b, Y.c - C @ X.c])
    
    return G, c, A, b

def CZLinMap(A, X, t):
    """
    CZLinMap - Function that calculates the linear mapping of a constrained zonotope.
    
    Parameters:
        A (numpy.ndarray): Matrix used in the linear mapping.
        X (Constrained Zonotope): Constrained Zonotope object.
        t (numpy.ndarray): Translation vector.
    
    Returns:
        Y (Constrained Zonotope): Constrained Zonotope that Z = {z: z = A*x+t, x \in X}
    """
    
    G = A @ X.G
    c = A @ X.c + t
    A = X.A
    b = X.b
    
    return G, c, A, b

def CZMinkowskiSum(X, Y):
    """
    CZMinkowskiSum - Function that calculates the Minkowski sum of two Constrained Zonotopes.
    
    Parameters:
        X (Constrained Zonotope): Constrained Zonotope object.
        Y (Constrained Zonotope): Constrained Zonotope object.
    
    Returns:
        Z (Constrained Zonotope): Constrained Zonotope that Z = {z = x+y: x \in X, y \in Y}
    """
    
    G = np.hstack([X.G, Y.G])
    c = X.c + Y.c
    A = block_diag(X.A, Y.A)
    b = np.concatenate([X.b, Y.b])
    
    return G, c, A, b

def CZsubsetCvxHull(Y):
    raise NotImplementedError("CZsubsetCvxHull function is not implemented yet.")


def ZtoHrep(G,c):
    raise NotImplementedError("ZtoHrep function is not implemented yet.")

### set operations ###


def compileCZ(Z):
    """
    compileCZ - Function returning a Yalmip constraint set representing the Constrained Zonotope.
    
    Z = {z = G*xi+c: A*xi = b, ||xi||_inf <= 1}
    
    Parameters:
        Z (Constrained Zonotope): Constrained Zonotope object.
    
    Returns:
        constraints (list): CVXPY constraint set
        p (cp.Variable): CVXPY variable representing a point in Z.
    """

    n, g = Z.G.shape

    p = cp.Variable(n)
    xi = cp.Variable(g)

    # Check whether there are constraints that need to be added.
    if np.all(Z.A == 0):
        constraints = [cp.norm_inf(xi) <= 1, p == Z.G @ xi + Z.c]
    else:
        constraints = [cp.norm_inf(xi) <= 1, Z.A @ xi == Z.b, p == Z.G @ xi + Z.c]

    return constraints, p

def convertCZ2AH(X):
    raise NotImplementedError("convertCZ2AH function is not implemented yet.")

def CZCenter(X):
    """
    CZCenter - Function that calculates the center of the Constrained Zonotope.
    
    Parameters:
        X (Constrained Zonotope): Constrained Zonotope object.
    
    Returns:
        c (numpy.ndarray): Point representing the center of the resulting zonotope.
    """
    
    if np.all(Z.A == 0):
        c = X.c
    else:
        try:
            solution = np.linalg.solve(X.A @ X.A.T, X.b)
        except np.linalg.LinAlgError:
            # Fallback to least-squares if matrix is singular
            solution = np.linalg.lstsq(X.A @ X.A.T, X.b, rcond=None)[0]

        c = X.G @ (X.A.T @ solution ) + X.c
    return c

def CZNbDoubles(X):
    """
    CZNbDoubles - Function that calculates the number of doubles of the Constrained Zonotope.
    
    Parameters:
        X (Constrained Zonotope): Constrained Zonotope object.
    
    Returns:
        total (int): Number of doubles of the Constrained Zonotope.
    """
    
    n, ng = X.G.shape
    nc = X.A.shape[0]

    total = (n + nc) * (ng + 1)
    return total

def CZOverbound(nDim, type, radius):
    """
    CZOverbound - Function returning trivial Zonotope overbounding simple sets like norm balls.
    
    Parameters:
        nDim (int): Dimension of the set.
        type (str): Type of the simple set.
        radius (float): Length of the set.
    
    Returns:
        X (Constrained Zonotope): Constrained Zonotope object representing the overbounding set.
    """
    
    if type.lower() == 'ball':
        G = np.diag(radius)
        c = np.zeros((nDim, 1))
        A = np.zeros((0, len(radius)))
        b = np.zeros((0, 1))

    else:
        raise ValueError('Unknown type of set')
    
    return G, c, A, b

def CZPropagate(system, prevX, u, d):
    """
    CZPropagate - Function encapsulating the propagation phase of a
    set-valued state estimation for a linear system with no uncertainties.

    The function returns the set X(k+1) that satisfies:
         X(k+1) = A X(k) + B u(k) + L D(k)
    
    Parameters:
        system (dict): Structure with all the matrices from the dynamical model.
        Z (Constrained Zonotope): Zonotope object.
        u (numpy.ndarray): Value of the actuation.
        d (Set): Set for the unknown disturbance signal.
    
    Returns:
        X (Constrained Zonotope): Zonotope that X = 
            {Ax + Bu + Ld: x \in prevX, d \in D} given u value.
    """
    A = system['A']
    B = system['B']
    L = system['L']
    
    G1, c1, A1, b1 = CZLinMap(A, prevX, B @ u)  # A*x + B*u
    G2, c2, A2, b2 = CZLinMap(L, d, np.zeros((L.shape[0], 1)))  # L*d

    return CZMinkowskiSum(BasicSet(G1, c1, A1, b1), BasicSet(G2, c2, A2, b2))  # A*x + B*u + L*d

def CZUpdate(system, X, u, y, noise):
    """
    CZUpdate - Function encapsulating the update phase of a
    set-valued state estimation for a linear system with no uncertainties.

    The function returns the set X(k) that satisfies the measurement set Y(k):
        X(k) = Xp(k) intersect_C Y(k)
    
    Parameters:
        system (dict): Structure with all the matrices from the dynamical model.
        X (Constrained Zonotope): Zonotope object, propagated set-valued estimate X(k)
        u (numpy.ndarray): Value of the actuation.
        d (Set): Set for the unknown disturbance signal.
    
    Returns:
        nextX (Constrained Zonotope): Zonotope with all values that can result in the
        measurement y given the noise set.
    """
    N = system['N']
    C = system['C']
    D = system['D']
    
    G, c, A, b = CZLinMap(-N, noise, y - D @ u)

    return CZIntersect(C, X, BasicSet(G, c, A, b))


def CZVolume(X):
    """
    CZVolume - Function that calculates the volume of the Constrained Zonotope.
    
    Parameters:
        X (Constrained Zonotope): Constrained Zonotope object.
    
    Returns:
        volume (float): Volume of the Constrained Zonotope.
    """
    
    n = X.G.shape[0]
    F, p = compileCZ(X)

    if n == 1:

        v = cp.Variable()

        problem = cp.Problem(cp.Minimize(v * p), F)
        extreme_neg = problem.solve()                   # Solve using mosek?

        problem = cp.Problem(cp.Maximize(v * p), F)
        extreme_pos = problem.solve()

        volume = extreme_pos - extreme_neg

    # elif n == 2:
    else:
        #Let us resort to an implementation of a ray shooting technique
        volume = VolumeRayShooting(F, p, n, CZCenter(X))

    return volume
