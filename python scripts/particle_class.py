import numpy as np
import scipy
import re
import math

#rawr

class particle:
    """A way to track the mass, spin statistics, degrees of freedom, and temperature of particles in the model"""
    
    def __init__(self, mass, spin, dof, temp, mu=0, fast=False):
        self.mass = mass #in GeV
        self.spin = spin
        self.dof = dof
        self.temp = temp #in GeV
        self.mu = mu #chemical potential, only used in ODE integration (not in deriving thermal cross sections)
        self.fast = fast #this option uses analytically derived expressions for rho and n assuming MB statistics
        
    def DF(self, E, MB=False):
        if MB: #for maxwell-boltzmann stats
            f=math.exp(-(E-self.mu)/self.temp)
        elif self.spin%1 == 0: #BE stats
            try:
                f = 1/(math.exp((E-self.mu)/self.temp) -1)
            except OverflowError:
                f=0
        else: #FD stats
            f = 1/(math.exp((E-self.mu)/self.temp) +1)
        return f
    def DFpm1(self,E ,MB=False): #this is for the terms like f+/-1 appearing in collision term
        if MB:
            f=1
        elif self.spin%1 == 0:
            try:
                f = 1/(math.exp((E-self.mu)/self.temp) -1) +1
            except OverflowError:
                f=1
        else:
            try:
                f = 1/(math.exp((E-self.mu)/self.temp) +1) -1
            except OverflowError:
                f = -1

        return f

    def rhoDF(self, E, t, MB=False): #make a generic function of temperature instead of evaluating at particle temp
        if MB: 
            f=math.exp(-(E-self.mu)/t)
        elif self.spin%1 == 0:
            f = 1/(math.exp((E-self.mu)/t) -1)
        else:
            f = 1/(math.exp((E-self.mu)/t) +1)
        return f

    def neq(self, maxtemp=100): #equilibrium number density
        if self.fast==True: #set for True when using ODE integrator, speeds up code by 30x
            return self.dof /(2 * math.pi**2) * self.temp * self.mass**2 *scipy.special.kn(2, self.mass/self.temp)# * math.exp(self.mu/self.temp)
        else: #do full numerical integration, ignoring chemical potential which we will assume factors out
            return self.dof/(2*math.pi**2)*scipy.integrate.quad(lambda p: p**2 * self.DF(math.sqrt(p**2+self.mass**2)), 0, self.temp*maxtemp)[0]

    def rhoeq(self, maxtemp=100): #equilibrium energy density 
        if self.fast == True: #set for True in integrator
            return self.dof* self.mass**4/(2*math.pi**2) * (3 * (self.temp/self.mass)**2 * scipy.special.kn(2, self.mass/self.temp)\
            + (self.temp/self.mass)* scipy.special.kn(1, self.mass/self.temp))* math.exp(self.mu/self.temp)
        else: #full numerical integral of energy density
            return self.dof/(2*math.pi**2)*scipy.integrate.quad(lambda p: p**2 *\
            self.DF(math.sqrt(p**2+self.mass**2))*math.sqrt(p**2+self.mass**2), 0, self.temp*maxtemp)[0]

    def peq(self, maxtemp=100): #equilibrium pressure, only used in Boltzmann equations for EoS
        return self.dof /(2 * math.pi**2) * self.temp**2 * self.mass**2 *scipy.special.kn(2, self.mass/self.temp) * math.exp(self.mu/self.temp)
    

    def tnew(self, rhonew, maxtemp=100): #solve for new temperature given a rho and a mu
        #only applies in MB limit where things can run much faster in the code
        deltarho = lambda temp: self.dof* self.mass**4/(2*math.pi**2) * (3 * (temp/self.mass)**2 * scipy.special.kn(2, self.mass/temp)\
        + (temp/self.mass)* scipy.special.kn(1, self.mass/temp)) * math.exp(self.mu/temp) - rhonew
        return scipy.optimize.fsolve(deltarho, self.temp)[0]

    def update(self,temp, mu=0):
        #appropriately updates properties of an instance of the particle class
        self.temp = temp
        self.mu = mu #adding a chemical potential

    def mass_update(self, m):
        self.mass = m
