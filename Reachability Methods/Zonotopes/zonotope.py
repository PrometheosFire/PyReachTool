import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from operations import *

class Zonotope:
    def __init__(self, H, p):
        self.H = H
        self.p = p.reshape(-1, 1)

    def box(self):
        H, p = boxZonotope(self)
        return Zonotope(H, p)

    def LinMap(self, A, t):
        H, p = ZonotopeLinMap(A, self, t)
        return Zonotope(H, p)

    def __add__(self, other):
        H, p = ZonotopeMinkowskiSum(self, other)
        return Zonotope(H, p)

    def compile(self):
        return compileZonotope(self)
    
    def center(self):
        return ZonotopeCenter(self)
    
    def nbDoubles(self):
        return ZonotopeNbDoubles(self)
    
    def overbound(self):
        H, p = ZonotopeOverbound(self)
        return Zonotope(H, p)
    
    def propagate(self, system, u, d):
        H, p = ZonotopePropagate(system, self, u, d)
        return Zonotope(H, p)

    def update(self, system, u, y, noise):
        H, p = ZonotopeUpdate(system, self, u, y, noise)
        return Zonotope(H, p)
    
    def volume(self):
        return ZonotopeVolume(self)
    
    def plot(self):
        if self.H.shape[0] == 2:
            return self.plot2D()
        elif self.H.shape[0] == 3:
            pass
        else:
            raise NotImplementedError("Plotting for dimensions higher than 3 is not supported.")
            

    def plot2D(self):
        _, m = self.H.shape

        combinations = np.array(np.meshgrid(*[[-1, 1]] * m)).T.reshape(-1, m)

        # Compute the points of the zonotope
        points = self.p.T + combinations @ self.H.T # Matrix multiplication: c * G^T
        # Compute the convex hull of the points
        hull = ConvexHull(points)

        # Plot the zonotope
        plt.figure()
        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], edgecolor='k', alpha=0.5, label="Zonotope")

        # Plot the generators (optional)
        for i in range(m):
            plt.arrow(self.p[0,0], self.p[1, 0], self.H[0, i], self.H[1, i], head_width=0.1, head_length=0.1, fc='r', ec='r', label=f'Generator {i+1}' if i == 0 else "")

        plt.title("2D Zonotope")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)
        plt.legend()
        plt.axis('equal')  # Ensure equal scaling
        plt.show()