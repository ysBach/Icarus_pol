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
    return 100 * amplitude * (sin(x))**spower * (cos(0.5*x))**cpower * sin(x-a_0)
#          ^^^ 100 multiplied to give % unit

#%%


param_bounds = ([0., 0.0, 0.0, 19],
                [10, 2  , 2  , 21])
# amplitude, spower, cpower, and a_0, resp.

# Bounded
popt_R, pcov_R = opt.curve_fit(pfunc, x_R, P_R, sigma=Perr_R,
                               bounds=param_bounds)
popt_V, pcov_V = opt.curve_fit(pfunc, x_V, P_V, sigma=Perr_V,
                               bounds=param_bounds)

# Unbounded: In this case opt.curve_fit uses Lev-Mar LSQ fitter by default.
popt_uR, pcov_uR = opt.curve_fit(pfunc, x_R, P_R, sigma=Perr_R)
popt_uV, pcov_uV = opt.curve_fit(pfunc, x_V, P_V, sigma=Perr_V)

plt.cla()
x = np.linspace(0.01,179,180)
plt.plot(x, pfunc(x, *popt_R),  'r', label='fit R' , ls='-')
plt.plot(x, pfunc(x, *popt_V),  'b', label='fit V' , ls='-')
plt.plot(x, pfunc(x, *popt_uR), 'r', label='fit uR', ls=':')
plt.plot(x, pfunc(x, *popt_uV), 'b', label='fit uV', ls=':')
plt.errorbar(x_R, P_R, yerr=Perr_R, color='r', ls='', marker='o', mfc='None', capsize=3, label='obs')
plt.errorbar(x_V, P_V, yerr=Perr_V, color='b', ls='', marker='o', mfc='None', capsize=3)
plt.xlabel('Phase angle ($^{\circ}$)')
plt.ylabel("P (%)")
plt.xlim(0,160)
plt.ylim(-1,8)
plt.grid(ls=':')
plt.legend()

print('bound  : ', popt_R, popt_V)
print('unbound: ', popt_uR, popt_uV)


#%%

Pmod_R  = pfunc(x, *popt_R)
Pmod_uR = pfunc(x, *popt_uR)
Pmod_V  = pfunc(x, *popt_V)
Pmod_uV = pfunc(x, *popt_uV)

Pmax_R  = Pmod_R.max()
Pmax_uR = Pmod_uR.max()
amax_R  = x[np.where(Pmod_R == Pmax_R)]
amax_uR = x[np.where(Pmod_uR == Pmax_uR)]

Pmax_V  = Pmod_V.max()
Pmax_uV = Pmod_uV.max()
amax_V  = x[np.where(Pmod_V == Pmax_V)]
amax_uV = x[np.where(Pmod_uV == Pmax_uV)]

print('bound R   : ', Pmax_R, amax_R)
print('bound V   : ', Pmax_V, amax_V)
print('unbound R : ', Pmax_uR, amax_uR)
print('unbound V : ', Pmax_uV, amax_uV)






#%%
#==============================================================================
# From here: For testing only
#==============================================================================

def dPda0(x, spower, cpower, a_0):
    '''
    The LHS of ``dP/da = 0``.
    The LHS is divided by ``amplitude*(sin(x))**(spower-1)*(cos(0.5*x))**(cpower-1).
    '''
    x   = np.deg2rad(x)
    a_0 = np.deg2rad(a_0)
    term1 = spower   * cos(x) * cos(x/2) * sin(x-a_0)
    term2 = cpower/2 * sin(x) * sin(x/2) * sin(x-a_0)
    term3 =            sin(x) * cos(x/2) * cos(x-a_0)
    return term1 - term2 + term3


for cpower in np.arange(0.01,2,0.1):
    for spower in np.arange(0.01,2,0.1):
        plt.plot(x, dPda0(x, spower, cpower, 20), 'r:', alpha=0.1)
        plt.grid(ls=':')

plt.plot(x, dPda0(x, 0.001, 0.001, 20), 'r', label='cpower=spower=e-3')
plt.plot(x, dPda0(x, 0.001, 2, 20)    , 'g', label='cpower=e-3, spower=2')
plt.plot(x, dPda0(x, 2, 0.001, 20)    , 'b', label='cpower=2, spower=e-3')
plt.plot(x, dPda0(x, 2, 2, 20)        , 'k', label='cpower=spower=2')
plt.axhline(y=0)
plt.axvline(x=110)
plt.xlabel('Phase angle ($(^{\circ}$))')
plt.ylabel('dP/da')
plt.legend()
plt.show()
#%%
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




