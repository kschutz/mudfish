import numpy as np 
import math 
from particle_class import particle 
from scipy import special
import scipy


class analytics:
	""" an option to input the analytic form of the cross sections to check for agreement w the numerics
	and also as an option to speed up the code"""

	def __init__(self, i1=None,i2=None,i3=None,f1=None,f2=None,f3=None):
		self.i1,self.i2,self.i3,self.f1,self.f2,self.f3 = i1,i2,i3,f1,f2,f3

	def cross_section(self):
		if self.i3 is not None:
			return 576*9*self.i1.mass**3 * self.i1.temp**2 /(96*math.sqrt(5) * math.pi**5 ) * self.i1.neq()**3
		elif self.i2 is None:
			if self.f1.spin%1 !=0:
				return self.i1.mass**2*self.f1.mass**2*math.sqrt(self.i1.mass**2/4 - self.f1.mass**2)*self.i1.temp * scipy.special.kn(1,self.i1.mass/self.i1.temp)/((2*math.pi)**3)
			else:
				return self.i1.mass**5 * self.i1.temp*special.kn(1, self.i1.mass/self.i1.temp) / ((2*np.pi)**3)
		elif self.f2 is None:
			print('rawr')
		else:
			rootlambda = lambda x,y,z: math.sqrt((1-(z+y)**2/x**2)*(1 - (z-y)**2/x**2))
			Ecut = 10*max(self.i1.temp, self.i1.mass)
			return self.i1.temp/(16*(2*np.pi)**5) * scipy.integrate.quad(lambda s: np.sqrt(s) *rootlambda(np.sqrt(s),self.i1.mass, self.i2.mass)\
			*rootlambda(np.sqrt(s), self.f1.mass, self.f2.mass)*scipy.special.kn(1, np.sqrt(s)/self.i1.temp),4*self.i1.mass**2, Ecut**2)[0]