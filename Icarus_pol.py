#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 22:12:02 2017

@author: aaa
"""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import custom_model

#%%

# Load data
prefix= 'testings/Icarus_pol/'
Rdata = np.loadtxt(prefix+'R.dat', dtype=bytes).astype(str)
Vdata = np.loadtxt(prefix+'V.dat', dtype=bytes).astype(str)
Rdata = Rdata[:, [0,1,2]].astype(float)
Vdata = Vdata[:, [0,1,2]].astype(float)

x_R   = Rdata[:,0]
P_R   = Rdata[:,1]
Perr_R= Rdata[:,2]

x_V   = Vdata[:,0]
P_V   = Vdata[:,1]
Perr_V= Vdata[:,2]

# Define the function and its Jacobians
def sin(x):
    return np.sin(x)

def cos(x):
    return np.cos(x)

def log(x):
    return np.log(x)

def pfunc(x, amplitude=1., spower=1., cpower=1., a_0=20.):
    ''' The empirical phase angle-polarization function.

    Parameters
    ----------
    x : float
        The phase angle in degree.
    amplitude : float, optional
        The amplitude, i.e., the ``b`` factor of the empirical function.
    spower : float, optional
        The power of sine term, i.e., the ``c1`` factor.
    cpower : float, optional
        The power of cosine term, i.e., the ``c2`` factor.

    a_0 : float, optional
        The inversion angle in degree, is fixed to 20 degrees here.

    '''

    x   = np.deg2rad(x)
    a_0 = np.deg2rad(a_0)
    return 100*amplitude*(sin(x))**spower * (cos(0.5*x))**cpower * sin(x-a_0)
#          ^^^ 100 multiplied to give % unit

#%%


param_bounds = ([0,0,0,18],[1, 2, 2, 22])
#                ^ ^ ^ ^
#                amplitude, spower, cpower, and a_0, resp.

# Bounded
popt_R, pcov_R = opt.curve_fit(pfunc, x_R, P_R, sigma=Perr_R,
                               bounds=param_bounds)
popt_V, pcov_V = opt.curve_fit(pfunc, x_V, P_V, sigma=Perr_V,
                               bounds=param_bounds)

# Unbounded: In this case opt.curve_fit uses Lev-Mar LSQ fitter by default.
popt_uR, pcov_uR = opt.curve_fit(pfunc, x_R, P_R, sigma=Perr_R)
popt_uV, pcov_uV = opt.curve_fit(pfunc, x_V, P_V, sigma=Perr_V)

print(popt_R, popt_V)
print(popt_uR, popt_uV)

#%%
plt.cla()
plt.plot(x_R, pfunc(x_R, *popt_R),  'r', label='fit R' , ls='', marker='x', ms=10)
plt.plot(x_V, pfunc(x_V, *popt_V),  'b', label='fit V' , ls='', marker='x', ms=10)
plt.plot(x_R, pfunc(x_R, *popt_uR), 'r', label='fit uR', ls='', marker='o', ms=10, mfc='w', alpha=0.5)
plt.plot(x_V, pfunc(x_V, *popt_uV), 'b', label='fit uV', ls='', marker='o', ms=10, mfc='w', alpha=0.5)
plt.errorbar(x_R, P_R, yerr=Perr_R, color='r', ls='', capsize=3)
plt.errorbar(x_V, P_V, yerr=Perr_V, color='b', ls='', capsize=3)
plt.xlabel('Phase angle ($^{\circ}$)')
plt.ylabel("P (%)")
plt.legend()



#%%

x       = np.linspace(0.01,179,180)
Pmod_R  = pfunc(x, *popt_R)
Pmod_uR = pfunc(x, *popt_uR)
Pmod_V  = pfunc(x, *popt_V)
Pmod_uV = pfunc(x, *popt_uV)

Pmax_R  = Pmod_R.max()
Pmax_uR = Pmod_uR.max()
amax_R  = np.where()
Pmax_V  = pfunc(x, *popt_V).max()
Pmax_uV = pfunc(x, *popt_uV).max()








#%%
#==============================================================================
# From here: For testing only
#==============================================================================

na, ns, nc = 20, 20, 20
amplitude  = np.linspace(5. , 15. , na)
spower     = np.linspace(0.01, 2  , ns)
cpower     = np.linspace(0.01, 2  , nc)
#aa, ss, cc = np.meshgrid(amplitude, spower, cpower)
chisq_R    = np.zeros((na, ns, nc)) # Reduced chi-square
chisq_V    = np.zeros((na, ns, nc)) # Reduced chi-square
dof_R      = len(P_R) - 3
dof_V      = len(P_V) - 3

#%%

for iaa in range(na):
    for iss in range(ns):
        for icc in range(nc):
            model_R = pfunc(x_R,
                            amplitude=amplitude[iaa],
                            spower=spower[iss],
                            cpower=cpower[icc],
                            a_0=20)
            chisq_R[iaa, iss, icc] = np.sum( ((model_R - P_R)/Perr_R)**2 ) / dof_R

            model_V = pfunc(x_V, amplitude=amplitude[iaa], spower=spower[iss], cpower=cpower[icc], a_0=20)
            chisq_V[iaa, iss, icc] = np.sum( ((model_V - P_V)/Perr_V)**2 ) / dof_V

print(chisq_R.min(), chisq_V.min())
print(np.where(chisq_R==chisq_R.min()), np.where(chisq_V==chisq_V.min()))

plt.clf()
plt.errorbar(x_R, P_R, yerr=Perr_R)
plt.errorbar(x_V, P_V, yerr=Perr_V, ls='--')

iaa0, iss0, icc0 = np.where(chisq_R==chisq_R.min())
plt.plot(x_R, pfunc(x_R, amplitude[iaa0], spower[iss0], cpower[icc0]))

for i in range(0,ns):
    plt.plot(x_R, pfunc(x_R, amplitude[iaa0], spower[i], cpower[icc0]))

iaa0, iss0, icc0 = np.where(chisq_V==chisq_V.min())
plt.plot(x_V, pfunc(x_V, amplitude[iaa0], spower[iss0], cpower[icc0]), ls='--')

plt.xlim(0,160)
plt.ylim(-2,10)

#%%
icc=0
plt.contourf(amplitude, spower, np.log10(chisq_R[:,:,icc]))
plt.colorbar()

#%%


def pfunc_deriv(x, amplitude=1., spower=1., cpower=1., a_0=20.):
    '''The empirical phase angle-polarization function.

    Parameters
    ----------
    x : float
        The phase angle in degree.
    amplitude : float, optional
        The amplitude, i.e., the ``b`` factor of the empirical function.
    spower : float, optional
        The power of sine term, i.e., the ``c1`` factor.
    cpower : float, optional
        The power of cosine term, i.e., the ``c2`` factor.
    a_0 : float, optional
        The inversion angle in degree.
    '''
    x   = np.deg2rad(x)
    a_0 = np.deg2rad(a_0)
    d_amplitude = (sin(x))**spower * (cos(0.5*x))**cpower * sin(x-a_0)
    d_spower    = amplitude * log(sin(x)) * (sin(x))**spower * (cos(0.5*x))**cpower * sin(x-a_0)
    d_cpower    = amplitude * log(cos(x)) * (sin(x))**spower * (cos(0.5*x))**cpower * sin(x-a_0)
    d_a_0       = - amplitude * (sin(x))**spower * (cos(0.5*x))**(cpower-1) * cos(x-a_0)
    return [d_amplitude, d_spower, d_cpower, d_a_0]


# Define the fitting method and call the custom model, and set initial guess.
fitter       = LevMarLSQFitter()
pfunc_model  = custom_model(pfunc, fit_deriv=pfunc_deriv)
pfunc_init_R = pfunc_model(amplitude=1, spower=1, cpower=1, a_0=20)
pfunc_init_R.a_0.fixed = True
pfunc_init_V = pfunc_model(amplitude=1, spower=1, cpower=1, a_0=20)

# Fit the data with Lev-Mar LSQ Fitter, for future use.
lsq_fit_R    = fitter(pfunc_init_R, x_R, P_R)


#%%
plt.cla()

fig, axarr = plt.subplots(1,3)
ax1, ax2, ax3 = axarr[0], axarr[1], axarr[2]

x     = np.linspace(0.01,180,180)
space = np.arange(0.1, 2, 0.2)

for i in range(len(space)):
    ax1.plot(x, pfunc(x, amplitude=space[i], spower=1, cpower=1, a_0=20), ls=":")
ax1.plot(x, pfunc(x, amplitude=space[0], spower=1, cpower=1, a_0=20),
         ls="-", lw=5, label='amplitude=0.1')
ax1.plot(x, pfunc(x, amplitude=space[-1], spower=1, cpower=1, a_0=20),
         ls="-", lw=5, label='amplitude=2.0')
ax1.set_title('changing amplitude')
ax1.legend()

for i in range(len(space)):
    ax2.plot(x, pfunc(x, amplitude=1, spower=space[i], cpower=1, a_0=20), ls=":")
ax2.plot(x, pfunc(x, amplitude=1, spower=space[0], cpower=1, a_0=20),
         ls="-", lw=5, label='sine power=0.1')
ax2.plot(x, pfunc(x, amplitude=1, spower=space[-1], cpower=1, a_0=20),
         ls="-", lw=5, label='sine power=2.0')
ax2.set_title('changing sine power')
ax2.legend()

for i in range(len(space)):
    ax3.plot(x, pfunc(x, amplitude=1, spower=1, cpower=space[i], a_0=20), ls=":")
ax3.plot(x, pfunc(x, amplitude=1, spower=1, cpower=space[0], a_0=20),
         ls="-", lw=5, label='cosine power=0.1')
ax3.plot(x, pfunc(x, amplitude=1, spower=1, cpower=space[-1], a_0=20),
         ls="-", lw=5, label='cosine power=2.0')
ax3.set_title('changing cosine power')
ax3.legend()

ax1.grid(ls=':')
ax2.grid(ls=':')
ax3.grid(ls=':')


#%%




