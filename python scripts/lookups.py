import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline

def lookup(process, particle_list, temp_list):
    """A way to generate lookup tables of the various cross sections to speed up time evolution"""
    if len(temp_list) == 1:
        C=[1]*len(temp_list[0])
        for i in temp_list[0]:
            [j.update(i) for j in particle_list]
            particles = particle_list
            Ecut = 5*max([i.temp for i in particles]+[i.mass for i in particles])
            result = process.CrossSection().val
            C[list(temp_list[0]).index(i)] = result
        return  interp1d(temp_list[0], np.log(C), kind="linear")
    elif len(temp_list) == 2:
        C = []
        xcounter = -1
        ycounter = -1
        for i in temp_list[0]:
            particle_list[0].update(i)
            xcounter = xcounter+1
            Cin = []
            for j in temp_list[1]:
                ycounter = ycounter+1
                particle_list[1].update(j)
                particles = particle_list
                Ecut = 5*max([i.temp for i in particles]+[i.mass for i in particles])
                result = process.CrossSection().val
                Cin.append(abs(result))
            C.append(Cin)
        print(C)
        return RectBivariateSpline(temp_list[0], temp_list[1], np.log(C))#[C, interp2d(temp_list[0], temp_list[1], np.log(C))]

	
