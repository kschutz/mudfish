import numpy as np
import math
import vegas
from analytics import analytics


class process:
    """A way to keep track of collision terms for different processes with up to 3 particles
    in the initial or final state. When entering in info for the initial and final state particles,
    please enter as an instance of a particle class so all the relevant information is there!
    For now the matrix element should be entered as a function of momenta
    """
    
    def __init__(self,amplitude,i1=None,i2=None,i3=None,f1=None,f2=None,f3=None,nonequilibrium=False, nevals=1e4, analytic=False, Ecut = 5, forward = 'rawr'):
        self.amplitude = amplitude
        self.i1 = i1
        self.f1 = f1
        self.i2 = i2
        self.f2 = f2
        self.i3 = i3
        self.f3 = f3
        self.nonequilibrium = nonequilibrium
        self.Ecut = Ecut
        self.nevals=nevals
        self.analytic = analytic
        self.forward = forward
        if self.i2 is not None:
            if self.i3 is not None:
                self.ptype = '3 to 2'
            elif self.f3 is not None:
                self.ptype = '2 to 3'
            elif self.f2 is None:
                self.ptype = 'inverse decay'
            else:
                self.ptype = '2 to 2'
        else:
            self.ptype = 'decay'


        particles = [self.i1, self.i2, self.i3, self.f1, self.f2, self.f3]
        energies = []
        for i in particles:
            if i is not None:
                energies.append(i.temp)
                energies.append(i.mass)
        self.upperenergy= self.Ecut*max(energies)

    def update(self,i1=None,i2=None,i3=None,f1=None,f2=None,f3=None):
        self.i1,self.i2,self.i3,self.f1,self.f2,self.f3 = i1,i2,i3,f1,f2,f3
    
    def CrossSection(self):
        if self.analytic == True:
            xc = analytics(self.i1, self.i2, self.i3, self.f1, self.f2, self.f3)
            return xc.cross_section()

        if self.ptype == '2 to 2':
            self.BF = lambda e1,e2,e3: self.i1.DF(e1)*self.i2.DF(e2)*self.f1.DFpm1(e3)*self.f2.DFpm1(e1+e2-e3)
            if self.nonequilibrium and self.amplitude.type=='s-Channel':
                ###I am adding something weird that is temporary
                if self.forward is True:
                    self.integrand = lambda s,e1,e2,e3: self.BF(e1,e2,e3)*e1\
                    *self.amplitude.unweightedintegrand(s,e1,e2,e3,self.i1.mass,self.i2.mass,self.f1.mass,self.f2.mass)
                    sIntegrator = vegas.Integrator([[(self.i1.mass+self.i2.mass)**2,4*self.upperenergy**2],[self.i1.mass,self.upperenergy],[self.i2.mass,self.upperenergy],[self.f1.mass,self.upperenergy]])
                    sIntegrator(lambda x:self.integrand(x[0],x[1],x[2],x[3]),nitn=10, neval=self.nevals)
                    result =sIntegrator(lambda x:self.integrand(x[0],x[1],x[2],x[3]),nitn=10, neval=self.nevals)
                    return result 
                elif self.forward is False:
                    self.integrand = lambda s,e1,e2,e3: self.BF(e1,e2,e3)*e3\
                    *self.amplitude.unweightedintegrand(s,e1,e2,e3,self.i1.mass,self.i2.mass,self.f1.mass,self.f2.mass)
                    sIntegrator = vegas.Integrator([[(self.i1.mass+self.i2.mass)**2,4*self.upperenergy**2],[self.i1.mass,self.upperenergy],[self.i2.mass,self.upperenergy],[self.f1.mass,self.upperenergy]])
                    sIntegrator(lambda x:self.integrand(x[0],x[1],x[2],x[3]),nitn=10, neval=self.nevals)
                    result =sIntegrator(lambda x:self.integrand(x[0],x[1],x[2],x[3]),nitn=10, neval=self.nevals)
                    return result 
                else:
                    self.integrand = lambda s,e1,e2,e3: self.BF(e1,e2,e3)*(e1-e3)\
                    *self.amplitude.unweightedintegrand(s,e1,e2,e3,self.i1.mass,self.i2.mass,self.f1.mass,self.f2.mass)
                    sIntegrator = vegas.Integrator([[(self.i1.mass+self.i2.mass)**2,4*self.upperenergy**2],[self.i1.mass,self.upperenergy],[self.i2.mass,self.upperenergy],[self.f1.mass,self.upperenergy]])
                    sIntegrator(lambda x:self.integrand(x[0],x[1],x[2],x[3]),nitn=10, neval=self.nevals)
                    result =sIntegrator(lambda x:self.integrand(x[0],x[1],x[2],x[3]),nitn=10, neval=self.nevals) #converge at ~3e4
                    return result
            elif self.amplitude.type=='s-Channel':
                self.integrand = lambda s,e1,e2,e3: self.BF(e1,e2,e3)\
                *self.amplitude.unweightedintegrand(s,e1,e2,e3,self.i1.mass,self.i2.mass,self.f1.mass,self.f2.mass)
                sIntegrator = vegas.Integrator([[(self.i1.mass+self.i2.mass)**2,4*self.upperenergy**2],[self.i1.mass,self.upperenergy],[self.i2.mass,self.upperenergy],[self.f1.mass,self.upperenergy]])
                sIntegrator(lambda x:self.integrand(x[0],x[1],x[2],x[3]),nitn=10, neval=self.nevals) #convergence at ~1e4
                result = sIntegrator(lambda x:self.integrand(x[0],x[1],x[2],x[3]),nitn=10, neval=self.nevals)
                return result

            elif self.amplitude.type == 't-Channel':
                self.integrand = lambda t,e1,e2,e3: self.BF(e1,e2,e3)\
                *self.amplitude.unweightedintegrand(t,e1,e2,e3,self.i1.mass,self.i2.mass,self.f1.mass,self.f2.mass)
                sIntegrator = vegas.Integrator([[-self.upperenergy, self.i1.mass**2. +self.f1.mass**2.],[self.i1.mass,self.upperenergy],[self.i2.mass,self.upperenergy],[self.f1.mass,self.upperenergy]])
                sIntegrator(lambda x:self.integrand(x[0],x[1],x[2],x[3]),nitn=10, neval=self.nevals) #convergence at ~1e4
                result = sIntegrator(lambda x:self.integrand(x[0],x[1],x[2],x[3]),nitn=10, neval=self.nevals)
                return result
                
                
        elif self.ptype == 'decay':
            self.BF = lambda ea,e1:self.i1.DF(ea)*self.f1.DFpm1(e1)*self.f2.DFpm1(ea-e1)
            if self.nonequilibrium:
                self.integrand = lambda e1,ea:self.BF(ea,e1) * ea\
                *self.amplitude.unweightedintegrand(0.0,ea,0.0,e1,self.i1.mass,self.f1.mass,self.f2.mass,0.0)
                decayintegrator = vegas.Integrator([[self.f1.mass+1e-10,self.upperenergy],[self.i1.mass+1e-10,self.upperenergy]])
                decayintegrator(lambda x:self.integrand(x[0],x[1]),nitn=10, neval=self.nevals)
                result = decayintegrator(lambda x:self.integrand(x[0],x[1]),nitn=10, neval=self.nevals)
                return result
            else:
                self.integrand = lambda e1,ea:self.BF(ea,e1)\
                *self.amplitude.unweightedintegrand(0.0,ea,0.0,e1,self.i1.mass,self.f1.mass,self.f2.mass,0.0)
                decayintegrator = vegas.Integrator([[self.f1.mass+1e-10,self.upperenergy],[self.i1.mass+1e-10,self.upperenergy]])
                decayintegrator(lambda x:self.integrand(x[0],x[1]),nitn=10, neval=self.nevals)
                result = decayintegrator(lambda x:self.integrand(x[0],x[1]),nitn=10, neval=self.nevals) #convergence after ~1e3 evals
                return result 
        
        
        elif self.ptype == 'inverse decay':
            self.BF = lambda ea,e1:self.i1.DF(e1)*self.i2.DF(ea-e1)*self.f1.DFpm1(ea)
            if self.nonequilibrium:
                self.integrand = lambda e1,ea:self.BF(ea,e1)*ea\
                *self.amplitude.unweightedintegrand(0.0,ea,0.0,e1,self.f1.mass,self.i1.mass,self.i2.mass,0.0)
                decayintegrator = vegas.Integrator([[self.i1.mass,self.upperenergy],[self.f1.mass,self.upperenergy]])
                decayintegrator(lambda x:self.integrand(x[0],x[1]),nitn=10, neval=self.nevals)
                result = decayintegrator(lambda x:self.integrand(x[0],x[1]),nitn=10, neval=self.nevals)
                return result 
            else:
                self.integrand = lambda e1,ea:self.BF(ea,e1)\
                *self.amplitude.unweightedintegrand(0.0,ea,0.0,e1,self.f1.mass,self.i1.mass,self.i2.mass,0.0)
                decayintegrator = vegas.Integrator([[self.i1.mass,self.upperenergy],[self.f1.mass,self.upperenergy]])
                decayintegrator(lambda x:self.integrand(x[0],x[1]),nitn=10, neval=self.nevals)
                result = decayintegrator(lambda x:self.integrand(x[0],x[1]),nitn=10, neval=self.nevals)
                return result
            
            
        elif self.ptype == '3 to 2':
            m = self.i1.mass
            self.BF = lambda e1,p2,p3,p4,e5: math.exp(-e1/self.i1.temp -math.sqrt(p2**2+m**2)/self.i2.temp)#self.i1.DF(math.sqrt(p3**2+m**2))*self.i2.DF(math.sqrt(p4**2+m**2))*self.i3.DF(e5)*self.f1.DFpm1(e1)*self.f2.DFpm1(math.sqrt(p2**2+m**2))
            self.integrand = lambda p2,p3,p4,x2,x3,x4,phi3,phi4: self.amplitude.WZWintegral(x2,p2,x3,phi3,p3,x4,phi4,p4,m)[-1]\
            *self.BF(self.amplitude.WZWintegral(x2,p2,x3,phi3,p3,x4,phi4,p4,m)[0],p2,p3,p4,self.amplitude.WZWintegral(x2,p2,x3,phi3,p3,x4,phi4,p4,m)[1])
            wzwintegrator = vegas.Integrator([[m,self.upperenergy],[m,self.upperenergy],[m,self.upperenergy],[-1,1],[-1,1],[-1,1],[0,2*np.pi],[0,2*np.pi]])
            wzwintegrator(lambda x: self.integrand(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]),nitn=10,neval=self.nevals)
            result = wzwintegrator(lambda x: self.integrand(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]),nitn=10,neval=self.nevals)
            return result
        elif self.ptype == '2 to 3':
            m = self.i1.mass
            self.BF = lambda e1,e2,e3,e4,e5: self.i1.DF(e1)*self.i2.DF(e2)*self.f1.DFpm1(e3)*self.f2.DFpm1(e4)*self.f3.DFpm1(e5)
        else:
            print('something silly is happening')
                