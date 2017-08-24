import numpy as np
import scipy
import re
import math
import numdifftools as nd
import math

def neq(T, m, dof):
	return dof /(2 * math.pi**2) * T * m**2 *scipy.special.kn(2, m/T)

def rhoeq(T, m, mu, dof):
	return dof* m**4/(2*math.pi**2) * (3 * (T/m)**2 * scipy.special.kn(2, m/T)\
            + (T/m)* scipy.special.kn(1, m/T))* math.exp(mu/T)

def peq(T, m , mu, dof):
	return dof /(2 * math.pi**2) * T**2 * m**2 *scipy.special.kn(2, m/T) * math.exp(mu/T)

def rhoplusp(T, m, mu, dof):
	return rhoeq(T, m, mu, dof)+peq(T, m , mu, dof)

def dndT(T, m, dof):
	return dof /(2 * math.pi**2) *m**2 *(m*scipy.special.kn(1, m/T)/T + 3* scipy.special.kn(2, m/T))

def drhodT(T, m, mu, dof):
	return dof /(2 * math.pi**2) *math.exp(mu/T)*m/T* (m*(m**2 + 3*T*(-mu + 4*T))*scipy.special.kn(0, m/T) \
		+ (-m**2*(mu - 5*T) + 6*T**2*(-mu + 4*T)) *scipy.special.kn(1, m/T))

def replace_element(l, index, newval):
	l[index]=newval
	return l

filelist = ['jacobian0'+str(j)+'.txt' for j in range(4)]+['jacobian1'+str(j)+'.txt' for j in range(4)]+['jacobian2'+str(j)+'.txt' for j in range(4)]+['jacobian3'+str(j)+'.txt' for j in range(4)]
jacmatrix=[ [i for i in range(4)] for j in range(4)]
for i in filelist:
	f = open(i, 'r')
	jac11 = f.read()
	f.close()
	jac11 = jac11.replace('BesselK', 'scipy.special.kn')
	jac11 = jac11.replace('Pi', 'math.pi')
	jac11 = jac11.replace('Log', 'math.log')
	index1 = int(i[8])
	index2 = int(i[9])
	#print(index1, index2)
	jacmatrix[index1][index2]=jac11

def derivatives(a, particles, k, sigma_wzw=0, sigma_pipiaa=0, sigma_agg=0, dE_pipiaa=0, dE_agg=0, Cprime=0, dialup_factor=1, dialup_photons=1, Hconversion=2e-19):
	mpi = particles[0].mass
	ma = particles[1].mass
	dofpi =particles[0].dof
	dofa = particles[1].dof


	upi, vpi, ua, va = k[0],k[1],k[2],k[3]

	Tpi = (vpi+1)/a
	Ta = (va+1)/a

	dTpi_val = vpi
	dTa_val = va
	ua_val = ua
	upi_val = upi

	
	sigma_aapipi =sigma_pipiaa
	sigma_gga = sigma_agg
	dE_aapipi = dE_pipiaa
	dE_gga = dE_agg


	constagg = sigma_agg/(Ta * scipy.special.kn(1,  ma/Ta))
	constgga = sigma_gga
	constaggE = dE_agg/(Ta**2* scipy.special.kn(1,  ma/Ta))
	constggaE = dE_gga#dE_agg/(Ta/a* scipy.special.kn(1,  ma/Ta))
	constaapipi = sigma_aapipi/neq(Ta, ma, dofa)**2
	constpipiaa = sigma_pipiaa/neq(Tpi, mpi, dofpi)**2
	constaapipiE = dE_aapipi/neq(Tpi, mpi, dofpi)**2
	constpipiaaE = dE_pipiaa/neq(Ta, ma, dofa)**2
	constpiapiaE = Cprime/(neq(Tpi, mpi, dofpi)*neq(Ta, ma, dofa))
	constWZW = sigma_wzw/neq(Tpi, mpi, dofpi)**3/Tpi**2
	H0=Hconversion

	jout=[0, 0, 0, 0]
	for i in range(4):
		jin=[0, 0, 0, 0]
		for j in range(4):
			#print(eval(jacmatrix[i][j]))
			jin[j] = float(eval(jacmatrix[i][j]))
		jout[i]=jin


	d=[0]*4
            
	d[1] = (-3*(rhoplusp((dTpi_val+1)/a,mpi,(dTpi_val+1)/a*math.log(upi_val+1), dofpi))\
     + a**2/Hconversion*(Cprime*(upi_val+1)*(ua_val+1)-dE_pipiaa*(upi_val+1)**2+dE_aapipi*(ua_val+1)**2))\
     /drhodT((dTpi_val+1)/a,mpi,(dTpi_val+1)/a*math.log(upi_val+1), dofpi) + (dTpi_val+1)/a
            
	d[3] = (dTa_val+1)/a+1/drhodT((dTa_val+1)/a,ma,(dTa_val+1)/a*math.log(ua_val+1), dofa)\
    *(-3*(rhoplusp((dTa_val+1)/a,ma,(dTa_val+1)/a*math.log(ua_val+1), dofa))+a**2/Hconversion\
	*(-Cprime*(upi_val+1)*(ua_val+1)+dE_pipiaa*(upi_val+1)**2-dE_aapipi*(ua_val+1)**2 -dE_agg*(ua_val+1) + dE_gga) )
        
	dneqpi = dndT((dTpi_val+1)/a,mpi,dofpi)*1/a*(d[1]-(dTpi_val+1)/a)
	dneqa = dndT((dTa_val+1)/a,ma,dofa)*1/a*(d[3]-(dTa_val+1)/a)

	d[0] = -3*(upi_val+1)/a- (upi_val+1)*dneqpi/neq((dTpi_val+1)/a,mpi,dofpi)\
     +a/(Hconversion*neq((dTpi_val+1)/a,mpi,dofpi))\
     *(-sigma_pipiaa * (upi_val+1)**2 +sigma_aapipi*(ua_val+1)**2 - sigma_wzw*upi_val*(upi_val+1)**2)    
	d[2] = -3*(ua_val+1)/a -(ua_val+1)*dneqa/neq((dTa_val+1)/a,ma,dofa)\
     +a/(Hconversion*neq((dTa_val+1)/a,ma,dofa))\
   		*(sigma_pipiaa*(upi_val+1)**2 - sigma_aapipi*(ua_val+1)**2 -sigma_agg*(ua_val+1) +sigma_gga)

	wzwterm = a/(Hconversion*neq((dTpi_val+1)/a,mpi,dofpi))*(-sigma_wzw*upi_val*(upi_val+1)**2)
	annihilationterm=a/(Hconversion*neq((dTpi_val+1)/a,mpi,dofpi))*(-sigma_pipiaa * (upi_val+1)**2 +sigma_aapipi*(ua_val+1)**2)
	decayterm = a/(Hconversion*neq((dTa_val+1)/a,ma,dofa))*(-sigma_agg*(ua_val+1) +sigma_gga)
	return [np.array(d), jout, wzwterm, annihilationterm, decayterm]

# def jacobian(particles, derivatives, current_values):
# 		#particles should be a list of the particles w boltzmann equations
# 		#derivatives should be a function that takes a list as its argument
# 		#current values is list with the coordinates of the point at which the Jacobian is evaluated
# 		#the list of current values should have the same ordering as the arguments of the derivatives function
# 	dim = 2*len(particles) #two variables per particle (n and rho)
# 	jmatrix = [0]*dim
# 	for i in range(dim):
# 		jin = [0]*dim
# 		for j in range(dim):
# 				#compute the derivative by replacing the jth current value with a variable and take the derivative of the
# 				#ith component of the collision derivatives wrt the jth variable holding other variables fixed.
# 				#Use the forward method because in this parameterization
# 				#(deviations from equilibrium/adiabaticity) the starting point is at zero, with values less than
# 				#zero making no physical sense, hence the derivatives initially shouldn't be two-sided
# 				#(this causes the code to break)
# 			rawr = nd.Derivative(lambda x: derivatives(replace_element(current_values,j,x))[i], method='forward')
# 			gulu = float(rawr(current_values[j]))
# 			jin[j] = gulu
# 		jmatrix[i] = jin
				
# 	return jmatrix


