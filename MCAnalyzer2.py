import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 13})
import re


data = pd.read_csv("data/MC_sep270mm.csv")
I = data['areaI']
theta = data['theta']
phi = data["phi"]
A = (data["areaA"] + data["areaB"])*0.5
I = data["areaI"]

print(np.average(I/A))

def get_rotated_t(x, y, z, amount):
    return np.cos(amount)*z + np.sin(amount)*y

z = np.cos(theta)
x = np.sin(theta)*np.cos(phi) # rotates about this axis
y = np.sin(theta)*np.sin(phi)

t = np.cos(theta)


r = np.sqrt(x*x + y*y)
theta = np.arctan2(y, x)

DO_POLAR_PLOT = False

if DO_POLAR_PLOT:
    dtheta = np.pi*2.0 / 20.0
    dr =           1.0 / 20.0
    rg = np.arange(0, 1, dr)
    tg = np.arange(-np.pi, np.pi, dtheta)
    zig = np.meshgrid(rg, tg)
    WEE = np.zeros_like(zig[0])
    for RR in range(len(rg)):
        for TT in range(len(tg)):
            I_counter = 0.0
            for (this_r, this_t, this_I) in zip(r, theta, I):
                if this_r >= rg[RR] and this_r < rg[RR] + dr and this_t >= tg[TT] and this_t < tg[TT] + dtheta:
                    WEE[TT][RR] += this_I
                    I_counter += 1
            WEE[TT][RR] /= I_counter

    ex, wye = np.meshgrid(rg, tg)
    print(WEE)


    polar_ax = plt.figure().add_subplot(projection='polar')
    #polar_ax.set_rgrids([])
    #polar_ax.grid(linestyle='dashed')
    #polar_ax.grid(linestyle='none')
    #zoopt = polar_ax.contour(wye + dtheta*0.5, ex, WEE, levels=4)
    #PLOOPT = polar_ax.scatter(phi, np.sqrt(x*x+y*y), c=I, s=2.0)
    #print(zoopt)
    #plt.colorbar(PLOOPT, ax=polar_ax)
    plt.show()

def func(x):
    return 66.0*np.cos(x)**2 + 3

# doubles per single
obs = np.array([0.005577401266, 0.01150356448, 0.01832201938, 0.02439585731, 0.02275155553])
ang = np.array([1.5, 36, 56, 67.5, 77.5])


ps = ""
for a, o in zip(ang, obs):
    val = np.arccos(get_rotated_t(x, y, z, (90-a)*np.pi/180.0))

    tt = func(val)

    #plt.scatter(x, y, c=(I/A*tt))
    #plt.show()
    expected = (np.sum(I/A*tt)/(np.sum(tt)))
    print(expected)
    ps += str(o / expected) + ", "
print("[" + ps[:-2] + "]")



# okay, so we expect that 18.0% of events are coincidences for angle 77.5 deg
# we actually get that 0.648% of them are
# The ratio between these two is 0.0359
# So our detector efficiency is about equal to this number