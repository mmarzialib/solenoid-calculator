#import matplotlib
#matplotlib.use('Qt4Agg')

import numpy as np
import matplotlib.pyplot as pl
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
        if self.a == s.a:
            return self.inductancemapped(s)
        elif self.a > s.a:
            func = lambda r,z: r * self.fieldBz(r, z)
            (ind, err) = dblquad(func, s.z - 0.5 * s.L, s.z + 0.5 * s.L, lambda z: 0.0, lambda z: s.a, epsrel = 1e-3)
            return ((2.0 * pi) * ind, (2.0 * pi) * err)
        else:
            return s.inductance(self)
        
    def inductancemapped(self, s):
        rmap = lambda x: (1.0 - np.exp(-x))
        func = lambda x,z: np.exp(-x) * rmap(x) * self.fieldBz(s.a * rmap(x), z)
        (ind, err) = dblquad(func, s.z - 0.5 * s.L, s.z + 0.5 * s.L, lambda z: 0.0, lambda z: np.inf, epsrel = 1e-3)
        return ((2.0 * pi * s.a ** 2) * ind, (2.0 * pi * s.a ** 2) * err)        
          
    def inductanceInf(self):
        return 4.0 * pi * self.a**2 * self.L

class Coil:
    # units = cm, mA, mH
    def __init__(self, radius, length, centerz, loops):
        self.solmod = SolenoidModel(radius, length, centerz)
        self.n = loops/length
        
    def fieldBz(self, r, z):
        return pi * 1e-4 * self.n * self.solmod.fieldBz(r,z)
        
    def fieldBr(self, r, z):
        return pi * 1e-4 * self.n * self.solmod.fieldBr(r,z)
        
    def inductance(self, c):
        (ind, err) = self.solmod.inductance(c.solmod)
        return (pi * 1e-6 * self.n * c.n * ind, pi * 1e-6 * self.n * c.n * err)
        
    def inductanceInf(self):
        return pi * 1e-6 * self.n ** 2 * self.solmod.inductanceInf()
       
    def fieldPlot(self, rMax, zMax, rRes = None, zRes = None):
        if rRes == None:
            rRes = rMax/100
        if zRes == None:
            zRes = zMax/100
        (rr, zz) = np.mgrid[0.0:rMax:rRes,0.0:zMax:zRes]
        Bz = np.zeros(np.shape(zz))
        Br = np.zeros(np.shape(rr))
        riMax, ziMax = rr.shape
        for ri in range(riMax):
            for zi in range(ziMax):
                Bz[ri,zi] = self.fieldBz(rr[ri,zi], zz[ri,zi])
                Br[ri,zi] = self.fieldBr(rr[ri,zi], zz[ri,zi])
        fig = pl.figure()
        ax = fig.gca()
        ax.set_xlim(0.0, rMax)
        ax.set_ylim(0.0, zMax)
        # Contourf plot
        cset_s = ax.streamplot(rr[:,1].reshape(riMax), zz[1,:].reshape(ziMax),
                               Br.T, Bz.T, linewidth=2.0/Bz[0,0]*np.sqrt(Br**2+Bz**2).T,
                               color='gray', arrowstyle='->')
        cset_z = ax.contour(rr, zz, Bz/Bz[0,0], colors='b')
        cset_r = ax.contour(rr, zz, Br/Bz[0,0], colors='r')
        # Label plot
        ax.clabel(cset_z, inline=1, fontsize=10)
        ax.clabel(cset_r, inline=1, fontsize=10)
        ax.set_xlabel('r')
        ax.set_ylabel('z')
        fig.show()

class MultiCoil(Coil):
    def __init__(self, coil_list = None):
        if coil_list == None:
            self.coils = []
        else:
            self.coils = coil_list
    
    def add_coil(self, coil):
        self.coils.append(coil)
    
    def fieldBz(self, r, z):
        Bz = 0.0
        for c in self.coils:
            Bz += c.fieldBz(r,z)
        return Bz
    
    def fieldBr(self, r, z):
        Br = 0.0
        for c in self.coils:
            Br += c.fieldBr(r,z)
        return Br
    
    def inductance(self):
        L = 0.0
        for c1 in self.coils:
            for c2 in self.coils:
                (L12, eL12) = c1.inductance(c2)
                L += L12
        return L
    
    def inductanceInf(self):
        L = 0.0
        for c1 in self.coils:
            L += c1.inductanceInf()
        return L
        
