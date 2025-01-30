import numpy as np
from operations import *
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

class Zonotope:
    def __init__(self, H, p):
        self.H = H
        self.p = p

    def box(self):
        return boxZonotope(self)

    def LinMap(self, A, t):
        return ZonotopeLinMap(A, self, t)

    def __add__(self, other):
        return ZonotopeMinkowskiSum(self, other)

    def compile(self):
        return compileZonotope(self)
    
    def center(self):
        return ZonotopeCenter(self)
    
    def nbDoubles(self):
        return ZonotopeNbDoubles(self)
    
    def overbound(self):
        return ZonotopeOverbound(self)
    
    def propagate(self, system, u, d):
        return ZonotopePropagate(system, self, u, d)

    def update(self, system, u, y, noise):
        return ZonotopeUpdate(system, self, u, y, noise)
    
    def volume(self):
        return ZonotopeVolume(self)
    
    def plot(self):
        if self.H.shape[0] == 2:
            return self.plot2D(self)
        elif self.H.shape[0] == 3:
            pass
        else:
            raise NotImplementedError("Plotting for dimensions higher than 3 is not supported.")
            

    def plot2D(self):
        m, n = self.H.shape

        m = self.H.shape[1]  # Number of generators
        combinations = np.array(np.meshgrid(*[[-1, 1]] * m)).T.reshape(-1, m)

        # Compute the points of the zonotope
        points = combinations @ self.H.T + self.p # Matrix multiplication: c * G^T

        # Compute the convex hull of the points
        hull = ConvexHull(points)

        # Plot the zonotope
        plt.figure()
        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], edgecolor='k', alpha=0.5, label="Zonotope")

        # Plot the generators (optional)
        for i in range(n):
            plt.arrow(0, 0, self.H[0, i], self.H[1, i], head_width=0.1, head_length=0.1, fc='r', ec='r', label=f'Generator {i+1}' if i == 0 else "")

        plt.title("2D Zonotope")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)
        plt.legend()
        plt.axis('equal')  # Ensure equal scaling
        plt.show()