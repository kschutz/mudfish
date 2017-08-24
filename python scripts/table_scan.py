import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from lookups import lookup
from particle_class import particle
from amplitude_class import amplitude
from process_class import process

try:
    import cPickle as pickle
except ImportError:
    import pickle

def lookup_scan(mpi, ma, temprange):
	file_end = 'mpi_'+str(mpi)+'_ma_'+str(ma)

	pion = particle(mpi, 0, 5, 0.1)
	pion.mass_update(mpi)
	alp = particle(ma,0, 1, 0.1)
	alp.mass_update(ma)
	photon = particle(0, 1, 2, 0.1)

	particles = [pion, alp, photon]

	fourpt = amplitude(lambda s:1, s=True)
	alpPhotons = amplitude(lambda s: 1,threept=True)

	annihilation=process(fourpt, i1=pion,i2=pion,f1=alp,f2=alp, nevals=3e4)
	antiannihilation = process(fourpt, i1=alp, i2=alp, f1=pion, f2=pion, nevals=3e4)
	transfer = process(fourpt, i1=pion,i2=alp,f1=pion,f2=alp,nonequilibrium=True, nevals=1e5)
	decaytophotons = process(alpPhotons, i1=alp,f1=photon,f2=photon, nevals=3e3)
	photoninversedecay = process(alpPhotons, i1=photon,i2=photon,f1=alp, nevals=3e3)

	annihilation_forward = process(fourpt, i1=pion, i2=pion, f1=alp, f2=alp, nonequilibrium=True, forward=True, nevals=3e4)
	alptophotons_energy = process(alpPhotons, i1=alp,f1=photon,f2=photon, nonequilibrium=True, nevals=3e3)


	ann_lookup = lookup(annihilation, [pion, alp], [temprange]*2)
	with open('annihilation_' + file_end + '.pkl', 'wb') as f:
		pickle.dump(ann_lookup, f)
	
	atogg = lookup(decaytophotons, [alp, photon], [temprange]*2)
	with open('decaytophotons_' + file_end + '.pkl', 'wb') as f:
		pickle.dump(atogg, f)

	ann_forward_lookup = lookup(annihilation_forward, [pion,alp], [temprange]*2)
	with open('ann_transfer_' + file_end + '.pkl', 'wb') as f:
		pickle.dump(ann_forward_lookup, f)	

	elastic_lookup = lookup(transfer, [pion,alp], [temprange]*2)
	with open('elastic_' + file_end + '.pkl', 'wb') as f:
		pickle.dump(elastic_lookup, f)	
	
	decay_lookup = lookup(alptophotons_energy, [photon,alp],[temprange]*2)
	with open('decay_transfer_' + file_end + '.pkl', 'wb') as f:
		pickle.dump(decay_lookup, f)	
