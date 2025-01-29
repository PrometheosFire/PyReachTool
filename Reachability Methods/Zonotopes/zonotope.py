import numpy as np
from operations import *

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


