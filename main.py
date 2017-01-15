import numpy as np
from numpy import absolute, sign, sqrt, arctan
from scipy.special import ellipe, ellipk, ellipeinc, ellipkinc
from scipy.constants import pi
from scipy.integrate import dblquad

def ellipz(phi,m):
    return ellipeinc(phi,m) - ellipe(m)*ellipkinc(phi,m)/ellipk(m)

def lambda0(phi, m):
    return ellipkinc(phi,1-m)/ellipk(1-m) + 2/pi * ellipk(m)* ellipz(phi,1-m)

class SolenoidModel:
    def __init__(self, radius, length, centerz):
        self.z = centerz
        self.a = radius
        self.L = length
    
    def fieldBz(self, r, z):
        xi_pos = z - self.z + self.L / 2.0
        xi_neg = z - self.z - self.L / 2.0
        # print "xi = ", xi_pos, xi_neg
        phi_pos = arctan(absolute(xi_pos/(self.a - r)))
        phi_neg = arctan(absolute(xi_neg/(self.a - r)))
        # print "phi = ", phi_pos, phi_neg
        m_pos = 4.0 / (xi_pos ** 2 + (r + self.a) ** 2)
        m_neg = 4.0 / (xi_neg ** 2 + (r + self.a) ** 2)
        # print "m/ra = ", m_pos, m_neg
        term1_pos = xi_pos * sqrt(m_pos) * ellipk(m_pos * (r * self.a)) 
        term1_neg = xi_neg * sqrt(m_neg) * ellipk(m_neg * (r * self.a))
        # print "t1 = ", term1_pos, term1_neg        
        term2_pos = sign(xi_pos * (self.a - r)) * lambda0(phi_pos,m_pos * (r * self.a))
        term2_neg = sign(xi_neg * (self.a - r)) * lambda0(phi_neg,m_neg * (r * self.a))
        # print "t2 = ", term2_pos, term2_neg
        return (term1_pos - term1_neg) / pi + term2_pos - term2_neg
        
    def fieldBr(self, r, z):
        if r/self.a > 0.01:
            return self.fieldBrFull(r, z)
        else:
            xi_pos = z - self.z + self.L / 2.0
            xi_neg = z - self.z - self.L / 2.0
            term1_pos = 1.0/sqrt((xi_pos ** 2 + self.a ** 2) ** 3)
            term1_neg = 1.0/sqrt((xi_neg ** 2 + self.a ** 2) ** 3)
            return - self.a ** 2 * r * (term1_pos - term1_neg)

    def fieldBrFull(self, r, z):
        xi_pos = z - self.z + self.L / 2.0
        xi_neg = z - self.z - self.L / 2.0
        # print "xi = ", xi_pos, xi_neg
        m_pos = 4.0 * (r * self.a)/ (xi_pos ** 2 + (r + self.a) ** 2)
        m_neg = 4.0 * (r * self.a)/ (xi_neg ** 2 + (r + self.a) ** 2) 
        term1_pos = ((1.0 - m_pos/2.0) * ellipk(m_pos) - ellipe(m_pos))/sqrt(m_pos)
        term1_neg = ((1.0 - m_neg/2.0) * ellipk(m_neg) - ellipe(m_neg))/sqrt(m_neg)
        return - 4.0/pi * sqrt(self.a / r) * (term1_pos - term1_neg)    
        
    def inductance(self, s):
        rmap = lambda x: (1.0 - np.exp(-x))
        func = lambda x,z: np.exp(-x) * rmap(x) * self.fieldBz(s.a * rmap(x), z)
        (ind, err) = dblquad(func, s.z - 0.5 * s.L, s.z + 0.5 * s.L, lambda z: 0.0, lambda z: np.inf, epsrel = 1e-3)
        return ((2.0 * pi * s.a ** 2) * ind, (2.0 * pi * s.a ** 2) * err)

class Coil:
    # units = cm, mA, mH
    def __init__(self, radius, length, centerz, loops):
        self.s = SolenoidModel(radius, length, centerz)
        self.n = loops/length
        
    def fieldBz(self, r, z):
        return pi * 1e-4 * self.n * self.s.fieldBz(r,z)
        
    def fieldBr(self, r, z):
        return pi * 1e-4 * self.n * self.s.fieldBr(r,z)
        
    def inductance(self, s):
        (ind, err) = self.s.inductance(s)
        return (pi * 1e-6 * self.n ** 2 * ind, pi * 1e-6 * self.n ** 2 * err)
        