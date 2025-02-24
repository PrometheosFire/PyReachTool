import numpy as np
from scipy.spatial import ConvexHull
from rayShooting import rayShooting

def VolumeRayShooting(F, p, n, center):
    """
    VolumeRayShooting -  Function that approximates the volume of any set by
        the volume of an inner polytope until the growth of collecting another
        set of points on the surface would result in less than 1% growth.
    Parameters:
        F CVXPY constraint set representing the reachable set.
        p (cp.Variable): CVXPY variable representing a point in the set.
        n (int): size of the state space.
        center (np.array): a guess for the center.
    
    Returns:
        vol (float): underapproximation of the real volume
    """
    
    prev_vol = 1E-10
    vol = 0
    all_points = []

    while abs(vol / prev_vol - 1) >= 1e-2:
        prev_vol = vol
        all_points = rayShooting(all_points, F, p, n, center)
        hull = ConvexHull(all_points.T)
        vol = hull.volume
        print(f'Volume: {vol}, increment: {(vol / prev_vol - 1) * 100}%, nb_points: {len(all_points)}')

    return vol

