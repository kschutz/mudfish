import math
import numpy as np
from scipy.optimize import fsolve
#import warnings

def p(e, m):
    return np.sqrt(e*e - m*m)

#warnings.filterwarnings("error")

class amplitude:
    """An object that represents |M|^2 in terms of the momenta of particles involved,
    or the Mandelstam variables (either is viable). """
    
    def __init__(self,form,s=False,t=False,complicatedMandelstam=False,threept=False,WZW=False,upperenergy=5):
        self.form=form
        self.upperenergy = upperenergy
        if s:
            self.type = 's-Channel'
        elif t:
            self.type = 't-Channel'
        elif complicatedMandelstam:
            self.type = 'more complicated'
        elif WZW:
            self.type = 'Wess Zumino Witten'
        else:
            self.type = 'three-point interaction'
            
            
    def unweightedintegrand(self,var,e1,e2,e3,m1,m2,m3,m4):
        if self.type == 's-Channel':
            st = var - (e1 + e2)**2
            if st<0:   
                p1 = p(e1,m1)
                p2 = p(e2,m2)
                p3 = p(e3,m3)
                if e1+e2-e3>m4:
                    pf = np.sqrt((e1+e2-e3)**2 -m4**2)
                    if st+(p1+p2)**2>0 and st+(p1-p2)**2<0:
                        if st+(p3+pf)**2>0 and st+(p3-pf)**2<0:
                            return np.pi/(8*(2*np.pi)**6)*self.form(var)*1/math.sqrt((e1+e2)**2-var)
                        else:
                            return 0.0
                    else:
                        return 0.0
                else:
                    return 0.0   
            else:
                return 0.0
        elif self.type == 't-Channel':
            tt = var - (e1 - e3)**2
            if tt<0:   
                p1 = p(e1,m1)
                p2 = p(e2,m2)
                p3 = p(e3,m3)
                if e1+e2-e3>m4:
                    pf = np.sqrt((e1+e2-e3)**2 -m4**2)
                    if tt+(p1+p3)**2>0 and tt+(p1-p3)**2<0:
                        if tt+(p2+pf)**2>0 and tt+(p2-pf)**2<0:
                            return np.pi/(8*(2*np.pi)**6)*self.form(var)*1/math.sqrt((e1-e3)**2-var)
                        else:
                            return 0.0
                    else:
                        return 0.0
                else:
                    return 0.0   
            else:
                return 0.0
            print('work in progress')
        elif self.type == 'more complicated':
            print('use the more complicated version...')
        elif self.type == 'three-point interaction':
            #for now e1 corresponds to the axion, e3 corresponds to one of the decay products
            if e3<e1-m2:
                pa = p(e1,m1)
                p1 = p(e3,m3)
                x0 = (-(e3-e1)**2 + m2**2 + p1**2 +pa**2)/(2*p1*pa)
                if x0>-1 and x0<1:
                    return 1/(4*(2*np.pi)**3)*self.form(s)
                else:
                    return 0
            else:
                return 0
     
    
    def WZWintegral(self,x2,p2,x3,phi3,p3,x4,phi4,p4,m):
        q2=p2*np.array([math.sqrt(1-x2**2),0,x2])
        q3=p3*np.array([math.sqrt(1-x3**2)*math.cos(phi3),math.sqrt(1-x3**2)*math.sin(phi3),x3])
        q4=p4*np.array([math.sqrt(1-x4**2)*math.cos(phi4),math.sqrt(1-x4**2)*math.sin(phi4),x4])
        
        E2 = math.sqrt(p2**2+m**2)
        E3 =math.sqrt(p3**2+m**2)
        E4=math.sqrt(p4**2+m**2)
        
        dfctsol = lambda p1:math.sqrt(p1**2+m**2)+E2-E3-E4\
        -math.sqrt(np.dot(np.array([0,0,p1])+q2-q3-q4,np.array([0,0,p1])+q2-q3-q4)+m**2)
        
        try:
            p1 = fsolve(dfctsol,0)[0]
            E1 = math.sqrt(p1**2+m**2)
            E5 = math.sqrt(np.dot(np.array([0,0,p1])+q2-q3-q4,np.array([0,0,p1])+q2-q3-q4)+m**2)
            if p1>0 and p1<math.sqrt(self.upperenergy**2- m**2):
                if E5>0 and E5< self.upperenergy:
                    return [E1,E5,self.form(p1,p2,p3,p4,x2,x3,x4,phi3,phi4,m)\
                    *1/(p1/E1+(p1+p2*x2-p3*x3-p4*x4)/E5)*p1**2*p2**2*p3**2*p4**2\
                    /(32*E1*E2*E3*E4*E5)]
                else:
                    return [1,1,0]
            else:
                return [1,1,0]
        except RuntimeWarning:
            return [1,1,0]