import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre
from numpy.linalg import inv
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 13})

data = pd.read_csv("data/MC_sep270mm.csv")
#data = pd.read_csv("data/MC_sep0mm.csv")

eta = 0.05810
eta = 0.052449
eta = 1.0
POSITIVE_UNFOLDING = True
RIDGE_ALPHA = 0.00016

theta = data["theta"]
phi = data["phi"]
A = data["areaA"]
B = data["areaB"]
I = data["areaI"]

theta = np.concatenate((theta, np.ones_like(theta)*np.pi - theta))
phi = np.concatenate((phi, -phi))
A = np.concatenate((A, A))
B = np.concatenate((B, B))
I = np.concatenate((I, I))

z = np.cos(theta)
x = np.sin(theta)*np.cos(phi) # rotates about this axis
y = np.sin(theta)*np.sin(phi)

t = np.cos(theta)

angles = np.array([1.5, 36, 56, 67.5, 77.5])
# rates (particles per second)
#measurements = np.array([0.2215815046, 0.3577005722, 0.7591865608, 1.957417836, 2.078396893])

#measurements = np.array([0.009008574123, 0.01598316206, 0.01265196654, 0.01216913205, 0.01419746993])
measurements = np.array([0.0179747, 0.03438801533, 0.04627956, 0.05863309079, 0.05139484])
eta = np.array([0.14972613829088624, 0.0792944934317219, 0.07271698864171938, 0.0802619004715257, 0.06789471623606193])
measurements /= eta**2
eta = 1.0
#eta = np.array([0.06214893741390633, 0.06734993385340832, 0.06603632200908513, 0.08373064389217853, 0.14213551413590492])
#measurements /= eta
#eta = 1.0

#measurements /= np.array([0.025905386912923052,0.023432786010404227,0.020845705571223422,0.018993961004704,0.011038722941779842])**2
#measurements /= np.array([0.07927346155730389, 0.03500080239923907, 0.02211696107486322, 0.020609218703385772, 0.020666060166717543])**2
#measurements /= np.array([0.03915759731063324, 0.03417509540352162, 0.02520302324451351, 0.024432512987958514, 0.024953470237109684])**2

measurements = np.flip(measurements)
#print(measurements)
angles = np.flip(angles)
angles0 = 90.0 - angles
#measurements = np.flip(measurements)
#measurements = np.array([6.44553747, 5.97849459, 5.05196354, 4.80791344, 4.51791888])
angles = np.pi*(90.0 - angles)/180.0

def get_rotated_t(x, y, z, amount):
    return np.cos(amount)*z + np.sin(amount)*y

def angular_average(data, thetas, binny):
    bins, edges = np.histogram(theta2, bins=binny, weights=I)
    bins0, edges0 = np.histogram(theta2, bins=binny)
    return (edges[1:] + edges[:-1])*0.5, bins/bins0

histo_bins = np.arange(0.0, np.pi, np.pi/64.0)
bc = []
ai = []
for rate, dtheta, a0 in zip(measurements, angles, angles0):
    zp = np.cos(dtheta)*z + np.sin(dtheta)*y
    yp = -np.sin(dtheta)*z + np.cos(dtheta)*y
    r = np.sqrt(x**2 + yp**2)
    theta2 = np.arctan2(zp, r) + np.pi/2.0
    bin_centers, avg_areas = angular_average(I, theta2, histo_bins)
    plt.plot(bin_centers*180.0/np.pi, avg_areas, label="Detectors at " + str(a0) + "°")
    bc.append(bin_centers)
    ai.append(avg_areas)
plt.xlim(0, 90)
plt.xlabel("Polar Viewing Angle [°]")
plt.ylabel(r"Average Projected Area [$m^2$]")
plt.legend()
plt.grid(alpha=0.12, color="black")
plt.savefig("area_basis.png",dpi=300)
plt.show()
plt.clf()

def combine_with_weights(w):
    combined_sum = np.zeros_like(ai[0])
    for (area, center, weight) in zip(ai, bc, w):
        combined_sum += area*weight
    return combined_sum

def inner_product(A, B):
    total = 0.0
    dSolidAngle = np.sin(bin_centers)*(histo_bins[1]-histo_bins[0])*2.0*np.pi
    return np.sum(dSolidAngle*A*B)*eta**2 / 4.0

print(inner_product(np.ones_like(bin_centers), np.ones_like(bin_centers)))



NOMINAL_AREA = 0.125
all_yvals = []
yhigh = []
ylow = []
err_vals = []
deta = 0.007
for i in range(0, len(bin_centers)):
    this_yval = 0.0
    this_weight_sum = 0.0
    this_err = 0.0
    this_ylow = 0.0
    this_yhigh = 0.0
    for j in range(0, len(ai)):
        this_yval += measurements[j]*ai[j][i]
        this_weight_sum += ai[j][i]
    this_yval /= this_weight_sum * (eta**2) * NOMINAL_AREA
    this_ylow /= this_weight_sum
    this_yhigh /= this_weight_sum
    all_yvals.append(this_yval)
    yhigh.append(this_yhigh)
    ylow.append(this_ylow)

from scipy.optimize import curve_fit
def func(x, A, B):
    return A*np.cos(x)**2 + B
popt, pcov = curve_fit(func, bin_centers, all_yvals, p0=[84, 13])
print(popt)
print(pcov[0][0]**0.5)
print(pcov[1][1]**0.5)
x = np.arange(0, 90.0, 1.0)
all_yvals = np.array(all_yvals)
ylow = np.array(ylow)
yhigh = np.array(yhigh)
#for j in range(0, len(ai)):
#        plt.plot(bin_centers, measurements[j]*ai[j]/(eta**2 / NOMINAL_AREA))
plt.xlabel("Polar Angle (°)")
plt.grid(alpha=0.12, color="black")
plt.fill_between(bin_centers*180.0/np.pi, ylow, yhigh, alpha=0.3)
plt.plot(bin_centers*180.0/np.pi, all_yvals, color="black", label="Weighted Average Reconstruction")
plt.plot(bin_centers*180.0/np.pi, func(bin_centers, *popt), color="red", linestyle="dashed", label=r"Curve Fit: $" + str(popt[0]) + r"\cos^2(\theta) + " + str(popt[1]) + r"12$")
plt.legend()

plt.ylabel(r"Muon Flux [$s^{-1}m^{-2}sr^{-2}$]")
plt.xlim(0, 90.0)
#plt.ylim(0, 120)
plt.savefig("teleplot.png",dpi=300)
plt.show()
#exit()

import scipy


#plt.plot(bin_centers, combined)
#plt.show()
def get_total_badness(weights, alpha):
    combined = combine_with_weights(weights)
    my_sum = 0.0
    for (meas, area_i) in zip(measurements, ai):
        P = inner_product(combined, area_i)/eta**2
        my_sum += (P - meas)**2
    return (my_sum + np.sum(np.array(weights)**2)*alpha)

alpha = 6.5e-5
alpha_low = 4e-5
alpha_high = 9e-5
consts = scipy.optimize.minimize(get_total_badness, np.array([1, 0, 0, 0, 0]), args=(alpha), method='Powell')
consts_low = scipy.optimize.minimize(get_total_badness, np.array([1, 0, 0, 0, 0]), args=(alpha_low), method='Powell')
consts_high = scipy.optimize.minimize(get_total_badness, np.array([1, 0, 0, 0, 0]), args=(alpha_high), method='Powell')
plt.grid(alpha=0.12, color="black")
plt.fill_between(bin_centers*180.0/np.pi, combine_with_weights(consts_low.x) + 5, combine_with_weights(consts_high.x) - 5, alpha=0.25)

popt, pcov = curve_fit(func, bin_centers, combine_with_weights(consts.x), p0=[84, 13])
print(popt)
print(pcov[0][0]**0.5)
print(pcov[1][1]**0.5)

plt.plot(bin_centers*180.0/np.pi, combine_with_weights(consts.x), label="Deconvolved Solution", color="black")
plt.plot(bin_centers*180.0/np.pi, func(bin_centers, *popt), label=r"Fit: $(107\pm 32) \cos^2(\theta) + (0\pm 1)$", linestyle="dashed", color="red")

iii = 0
for (a, ww) in zip(ai, consts.x):
    #if iii==0:
        #plt.plot(bin_centers, a*ww, label = "Basis Functions", color="green", linestyle="dotted")
    #else:
        #plt.plot(bin_centers, a*ww, color="green", linestyle="dotted")
    iii += 1
plt.xlabel(r"Polar Angle ($\degree$)")
plt.xlim(0, 90.0)
plt.ylim(0, 150.0)
plt.ylabel(r"Muon Flux $(sr^{-1}m^{-2}s^{-1})$")
plt.legend()
plt.savefig("good_unfold.png", dpi=300)
plt.show()

print(inner_product(ai[0], ai[0]))
print(consts.x)


L_CURVE_SCAN = False
if L_CURVE_SCAN:
    L_curve_y = []
    L_curve_x = []
    for x in np.arange(-10, 1, 0.05):
        expval = 10**x
        L_curve_x.append(expval)
        consts = scipy.optimize.minimize(get_total_badness, np.array([0, 0, 0, 0, 0]), args=(expval), method='Powell')
        new_weights = consts.x
        L_curve_y.append(np.log(get_total_badness(new_weights, 0.0)))
    plt.plot(L_curve_x[1:-1], np.diff(np.diff(L_curve_y)))
    plt.xscale('log')
    
    plt.show()

plt.savefig("maybe_not_failed_unfolding.png", dpi=300)
exit()

def get_h_func(i, t):
    k = np.arccos(np.abs(t)) # even in theta
    k = t
    #k = np.cos(k)*np.pi*0.5 # even in cos()

    return np.interp(k, bc[i], ai[i])

    ###################################
    # Legendre Polynomials
    #return legendre(2*i)(t)

    ###################################
    # Flat window functions
    left_bound = i/len(angles) * np.pi * 0.5
    right_bound = (i+1)/len(angles) * np.pi * 0.5
    
    A = (k >= left_bound)*(k < right_bound)
    #return A

    ###################################
    # Gaussian Window Functions
    center = i/(len(angles)-1) * np.pi * 0.5
    gauss = np.exp(-10.0*(k - center)**2)
    #return gauss

    ###################################
    # Triangle Window Functions
    center = i/(len(angles)-1) * np.pi * 0.5
    dx = 1/(len(angles)-1)*np.pi*0.5
    #return  (k < center)*(k > center - dx) * (dx - (center - k)) + (k > center)*(k < center + dx) * (dx - (k - center))

for i in range(5):
    plt.plot(np.arange(0, np.pi, 0.01), get_h_func(i, np.arange(0, np.pi, 0.01)))
plt.show()

G_func = I

def integrate_over_sphere(Y):
    return 4.0 * np.pi * np.sum(Y) / len(Y)

# rows of M are the amount of each Gi meas. in SH[row #]
M = []
for a in angles:
    N = len(angles)
    basis_h = []
    for i in range(N):
        _h = get_h_func(i, get_rotated_t(x, y, z, a))
        basis_h.append(_h/integrate_over_sphere(_h))
    #basis_h = basis_h[:2]
    c = []
    for h in basis_h:
        c.append(integrate_over_sphere(h * G_func))
        #c.append(np.sum(H*I))
    M.append(c)
M = np.array(M).T
Minv = np.linalg.inv(M)
plt.imshow(M)
plt.show()

basis_h = []
for i in range(N):
    basis_h.append(get_h_func(i, t))
c = np.matmul(Minv, measurements.T)


XX = M
yy = measurements.T

L_curve_x = []
L_curve_y = []
L_pos = []
for exponent in np.arange(-15, 10, 0.025):
    alpha = np.power(10, exponent)
    L_curve_x.append(alpha)
    clf = Ridge(alpha=alpha, positive=POSITIVE_UNFOLDING)
    clf.fit(XX, yy)
    L_curve_y.append(np.log(np.sum((yy - clf.predict(XX))**2)))
    L_pos.append(np.min(clf.coef_))
    c = clf.coef_
plt.plot(L_curve_x[1:-1], -np.diff(np.diff(L_curve_y)))
plt.xscale("log")
plt.show()
plt.plot(L_curve_x, L_curve_y)
plt.xscale("log")
plt.show()
plt.plot(L_curve_x, L_pos)
plt.xscale("log")
plt.show()

# alpha value identified from kink in L-curve generated in Tikhonov analysis above
# 0.0015
clf = BayesianRidge()
clf = Ridge(RIDGE_ALPHA, positive=POSITIVE_UNFOLDING)
clf.fit(XX, yy)
L_curve_y.append(np.log(np.sum((yy - clf.predict(XX))**2)))
plt.scatter(angles, yy)
plt.scatter(angles, clf.predict(XX))
plt.show()
c = clf.coef_
#c = np.matmul(M, yy)


IW = np.zeros_like(theta)
n=0
for (h, C) in zip(basis_h, c):
    IW += h*C
    n += 1

plt.rcParams["figure.figsize"] = (7,7)
my_ax = plt.figure().add_subplot(projection='3d') # "and my axe" hahahahahaha lol

polar_ax = plt.figure().add_subplot(projection='polar')
polar_ax.set_rgrids([])
polar_ax.grid(linestyle='dashed')
PLOOPT = polar_ax.scatter(phi, np.sqrt(x*x+y*y), c=IW, s=4.0)

sc = my_ax.scatter3D(x, y, z, c=IW, s=4.0, alpha=1.0)
plt.colorbar(sc, ax=my_ax)
my_ax.set_box_aspect([ub - lb for lb, ub in (getattr(my_ax, f'get_{a}lim')() for a in 'xyz')])
my_ax.set_xlabel("x")
my_ax.set_ylabel("y")
my_ax.set_zlabel("z")
plt.show()


theta = np.arange(0, np.pi/2.0, 0.001)
t = np.cos(theta)
basis_h = []
for i in range(N):
    basis_h.append(get_h_func(i, t))

x = []
y = []
for i in range(0, len(theta)):
    x.append(theta[i])
    this_sum = 0
    for (C, H) in zip(c, basis_h):
        this_sum += C*H[i]
    y.append(this_sum)

plt.plot(np.array(x)*180.0/np.pi, y)
for i in range(N):
    plt.plot(np.array(x)*180.0/np.pi, get_h_func(i, t)*c[i], label=r"$c_" + str(i) + r"P_" + str(i*2) + r"(\cos(\theta))$")
plt.legend()
plt.show()


final_sum = 0.0
for i in range(N):
    final_sum += integrate_over_sphere(get_h_func(i, t)*c[i])
print(final_sum)

print(measurements[-1]/(integrate_over_sphere(I)/(2.0)))
print(integrate_over_sphere(I)/(4.0*np.pi))

print(c)