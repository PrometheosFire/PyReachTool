import cvxpy as cp
import numpy as np
from itertools import combinations
from zonotope import Zonotope

### set operations ###

def boxZonotope(Z):
    """
    boxZonotope - Function that calculates the Interval that overbounds the Zonotope.
    
    Parameters:
        Z (Zonotope): Zonotope object.
    
    Returns:
        Zbox (Zonotope): Zonotope Zonotope matching an Interval.
    """

    Zbox_p = Z.p
    Zbox_H = np.diag(np.sum(np.abs(Z.H), axis=1))
    return Zonotope(Zbox_H, Zbox_p)

def ZonotopeLinMap(A, Z, t):
    """
    ZonotopeLinMap - Function that calculates the linear map AX + t for Zonotope X.
    
    Parameters:
        A (numpy.ndarray): Matrix defining the linear transformation.
        H (numpy.ndarray): Generator Matrix defining the zonotope.
        c (numpy.ndarray): Point representing the center of the zonotope.
        t (numpy.ndarray): Translation vector
    
    Returns:
        new_H (numpy.ndarray): Generator Matrix defining the resulting zonotope.
        new_c (numpy.ndarray): Point representing the center of the resulting zonotope.
    """
    
    new_H = A @ Z.H
    new_c = A @ Z.p + t
    return new_H, new_c


def ZonotopeMinkowskiSum(X, Y):
    """
    ZonotopeMinkowskiSum - Function that calculates the Minkowski sum of two Zonotopes.
    
    Parameters:
        X (Zonotope): Zonotope object.
        Y (Zonotope): Zonotope object.
    
    Returns:
        Z (Zonotope): Zonotope that Z = {z = x+y: x \in X, y \in Y}.
    """
    
    H = np.hstack((X.H, Y.H))
    p = X.p + Y.p
    return Zonotope(H, p)

### set operations ###

def compileZonotope(Z): 
    """
    compileZonotope - Function returning a CVXPY constraint set representing the Zonotope.
    
    Z = {z = G*xi+c: A*xi <= b, ||xi||_inf <= 1}
    
    Parameters:
        Z (Zonotope): Zonotope object.
    
    Returns:
        constraints (list): CVXPY constraint set
        p (cp.Variable): CVXPY variable representing a point in Z.
    """

    n, m = Z.H.shape
    x = cp.Variable(m)
    p = cp.Variable(n)

    if Z.H is None or Z.H.size == 0:
        constraints = [p >=1, p <= -1]
    else:
        constraints = [cp.norm_inf(x) <= 1, p == Z.H @ x + Z.p]
    
    return constraints, p

def ZonotopeCenter(Z):
    """
    ZonotopeCenter - Function that calculates the center of the Zonotope.
    
    Parameters:
        Z (Zonotope): Zonotope object.
    
    Returns:
        c (numpy.ndarray): Point representing the center of the resulting zonotope.
    """
    
    return Z.p

def ZonotopeNbDoubles(Z):
    """
    ZonotopeNbDoubles - Function that calculates the number of doubles of the Zonotope.
    
    Parameters:
        Z (Zonotope): Zonotope object.
    
    Returns:
        total (int): Number of doubles of the zonotope.
    """
    n, m = Z.H.shape
    
    return n * (m + 1)


def ZonotopeOverbound(nDim, type, radius):
    """
    ZonotopeOverbound - Function returning trivial Zonotope overbounding simple sets like norm balls.
    
    Parameters:
        nDim (int): Dimension of the set.
        type (str): Type of simple set.
        radius (float): Length of the set.
    
    Returns:
        Z (Zonotope): Trivial Zonotope object.
    """
    
    if type.lower() == 'ball':
        H = np.diag(radius)
        p = np.zeros((nDim, 1))
    else:
        raise ValueError('Unknown type of zonotope')
    
    return Zonotope(H, p)

def ZonotopePropagate(system, prevX, u, d):
    """
    ZonotopePropagate - Function encapsulating the propagation phase of a
    set-valued state estimation for a linear system with no uncertainties.

    The function returns the set X(k+1) that satisfies:
         X(k+1) = A X(k) + B u(k) + L D(k)
    
    Parameters:
        system (dict): Structure with all the matrices from the dynamical model.
        Z (Zonotope): Zonotope object.
        u (numpy.ndarray): Value of the actuation.
        d (Set): Set for the unknown disturbance signal.
    
    Returns:
        X (Zonotope): Zonotope that X = 
            {Ax + Bu + Ld: x \in prevX, d \in D} given u value.
    """
    
    A = system['A']
    B = system['B']
    L = system['L']
    
    H1, c1 = ZonotopeLinMap(A, prevX, B @ u) #  A*x + B*u
    H2, c2 = ZonotopeLinMap(L, d, np.zeros((L.shape[0], 1)))
    
    return ZonotopeMinkowskiSum(Zonotope(H1, c1,), Zonotope(H2, c2)) # A*x + B*u + L*Disturbance

def ZonotopeUpdate(system, X, u, y, noise):
    """
    ZonotopeUpdate - Function encapsulating the update phase of a
    set-valued state estimation for a linear system with no uncertainties.

    The function returns the set X(k) that satisfies the measurement set Y(k):
        X(k) = Xp(k) intersect_C Y(k)
    
    Parameters:
        system (dict): Structure with all the matrices from the dynamical model.
        X (Zonotope): Zonotope object, propagated set-valued estimate X(k)
        u (numpy.ndarray): Value of the actuation.
        y (numpy.ndarray): Measurement.
        noise (set): Set for the  noise.
    Returns:
        nextX (Zonotope): Zonotope with all values that can result in the
%       measurement y given the noise set.
    """
    
    N = system['N']
    C = system['C']
    D = system['D']
    
    H1, c1 = ZonotopeLinMap(-N, noise, y - D @ u)
    Y = boxZonotope(Zonotope(H1, c1))

    n = X.H.shape[0]

    H = X.H
    p = X.p

    for jout in range(C.shape[0]):  # take each measurement as independent and intersect
        c =C[jout, :].reshape(-1, 1)
        d = Y.p[jout, 0]
        sigma = Y.H[jout, jout]

        # compute the intersection with the strip for the jout measurement
        # Test added to avoid numerical problems
        if abs((c.T @ (H @ H.T) @ c + sigma**2)) < 1E-8:
            if abs(c.T @ p - d) <= 1E-8:
                continue
            else:
                nextH = np.full_like(H, np.nan)
                nextc = np.full_like(p, np.nan)
                return nextH, nextc

        lstar = (H @ H.T @ c) / (c.T @ H @ H.T @ c + sigma**2)
        nextH = np.hstack(((np.eye(n) - lstar @ c.T) @ nextH, sigma * lstar))
        nextc = nextc + lstar * (d - c.T @ nextc)
    
    return Zonotope(nextH, nextc)

def ZonotopeVolume(Z):
    """
    ZonotopeVolumme - Function that calculates the hypervolume of the Zonotope.
    
    Parameters:
        Z (Zonotope): Zonotope object.
    
    Returns:
        volume (float): Volume of the zonotope.
    """
    
    n, m = Z.H.shape

    # Generate all combinations of m choose n
    comb_indices = list(combinations(range(m), n))

    vol = 0

    for indices in comb_indices:
        submatrix = Z.H[:, indices]
        vol += abs(np.linalg.det(submatrix))

    vol = 2**n * vol
    
    return vol
