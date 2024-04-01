import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 13})

def func(x, A, B):
    return A*np.cos(x)**2 + B**2

Y = [122.22338913, 93.57563216, 50.94173493, 15.30366931, 5.33114536]
Y = [12.84115407 , 9.17918161,  4.11239366,  0.     ,     0.        ]
errs = np.sqrt(Y)*2.0

centers = np.arange(0, np.pi/2.0 + 0.001, np.pi/10.0)
centers = (centers[1:] + centers[:-1])*0.5
print(centers)

popt, pcov = curve_fit(func, centers, Y)
x = np.arange(0, 90.0, 1.0)

plt.plot(x, func(x*np.pi/180.0, *popt), color = 'red', label= r"$\cos^2$ Fit")

E = func(centers, *popt)
O = Y
d = 5 - 2

print("Normalized Chi-Squared: " + str(np.sum((O-E)**2/E)/d))

plt.bar(centers*180.0/np.pi, Y, width=16, yerr = errs, label="Tikhonov-Unfolded Measurements")

plt.xlabel("Muon Polar Angle [°]")
plt.ylabel(r"Muon Flux [$s^{-1}m^{-2}sr^{-2}$]")

plt.legend()
plt.show()

