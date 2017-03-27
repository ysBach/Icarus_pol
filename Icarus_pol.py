#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 22:12:02 2017

@author: aaa
"""
#==============================================================================
# Import modules
#==============================================================================
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import custom_model


#%%
#==============================================================================
# Define the functions
#==============================================================================
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
#==============================================================================
# Load data
#==============================================================================
prefix= './'
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


#%%
#==============================================================================
# Get the best fit for future use
#==============================================================================

param_bounds = ([0., -3.0, -3.0, 20],
                [1 ,  2  ,  2  , 30])
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
x = np.linspace(0.01,179,4*180)
plt.plot(x, pfunc(x, *popt_R),  'r', label='fit R' , ls='-')
plt.plot(x, pfunc(x, *popt_V),  'b', label='fit V' , ls='-')
plt.plot(x, pfunc(x, *popt_uR), 'r', label='fit R (unbound)', ls=':')
plt.plot(x, pfunc(x, *popt_uV), 'b', label='fit V (unbound)', ls=':')
plt.errorbar(x_R, P_R, yerr=Perr_R, color='r', ls='', marker='o', mfc='None',
             capsize=3, label='obs')
plt.errorbar(x_V, P_V, yerr=Perr_V, color='b', ls='', marker='o', mfc='None',
             capsize=3)
plt.xlabel('Phase angle ($^{\circ}$)')
plt.ylabel("P (%)")
plt.xlim(0,180)
plt.ylim(-1,8)
plt.grid(ls=':')
plt.legend()


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

print('parameters-----amplitude    spower      cpower        a_0 ')
print('R bound  : ', popt_R)
print('V bound  : ', popt_V)
print('-'*80)
print('R unbound: ', popt_uR)
print('V unbound: ', popt_uV)
print('-'*80)
print('            P_max | alpha_max')
print('bound R   : {0:.4f}| {1:.1f}'.format(Pmax_R, amax_R[0]))
print('bound V   : {0:.4f}| {1:.1f}'.format(Pmax_V, amax_V[0]))
print('unbound R : {0:.4f}| {1:.1f}'.format(Pmax_uR, amax_uR[0]))
print('unbound V : {0:.4f}| {1:.1f}'.format(Pmax_uV, amax_uV[0]))

#%%
#==============================================================================
# Fix a_0 value to 20 deg. Get marginalized error of Pmax and amax.
#==============================================================================

x = np.linspace(0.01,160,4*160)
# I calculated upto ~160 deg, since negative cpower gives maximum at 180 deg.

b0_R, s0_R, c0_R, a0_R = popt_R
b0_V, s0_V, c0_V, a0_V = popt_V

b_R = np.arange(0.8*b0_R, 1.2*b0_R, 0.01*b0_R)
s_R = np.arange(0.4*s0_R, 1.6*s0_R, 0.01*s0_R)
c_R = np.arange(0.7*c0_R, 1.3*c0_R, 0.01*c0_R)

b_V = np.arange(0.8*b0_V, 1.2*b0_V, 0.01*b0_V)
s_V = np.arange(0.4*s0_V, 1.6*s0_V, 0.01*s0_V)
c_V = np.arange(0.7*c0_V, 1.3*c0_V, 0.01*c0_V)

#chisq_R  = np.zeros((len(b_R), len(s_R), len(c_R)))
#Pmax_R   = np.zeros((len(b_R), len(s_R), len(c_R)))
#amax_R   = np.zeros((len(b_R), len(s_R), len(c_R)))
#chisq_V  = np.zeros((len(b_V), len(s_V), len(c_V)))
dof_R    = len(P_R)-3
dof_V    = len(P_V)-3
chimax_R = 1 + np.sqrt(2/dof_R)
chimax_V = 1 + np.sqrt(2/dof_V)

for bb in range(0, len(b_R)):
    for ss in range(0, len(s_R)):
        for cc in range(0, len(c_R)):
            model = pfunc(x_R, b_R[bb], s_R[ss], c_R[cc], a0_R)
            chisq = np.sum( ((P_R-model)/Perr_R)**2 )/dof_R
#            chisq_R[bb, ss, cc] = chisq
            if chisq < chimax_R:
                model  = pfunc(x, b_R[bb], s_R[ss], c_R[cc], a0_R)
                with open('Pmax_R.txt', 'a') as f:
                    Pmax = model.max()
                    amax = x[np.where(model == model.max())]
                    f.write('{0:d} {1:d} {2:d} {3:.4f} {4:.4f}\n'.format(bb, ss, cc, Pmax, amax[0]))
                    if amax>150:
                        plt.plot(x, model)
#    print(bb)
#                Pmax_R[bb,ss,cc] = model.max()
#                amax_R[bb,ss,cc] = x[np.where(model == model.max())]



for bb in range(0, len(b_V)):
    for ss in range(0, len(s_V)):
        for cc in range(0, len(c_V)):
            model = pfunc(x_V, b_V[bb], s_V[ss], c_V[cc], a0_V)
            chisq = np.sum( ((P_V-model)/Perr_V)**2 )/dof_V
#            chisq_R[bb, ss, cc] = chisq
            if chisq < chimax_V:
                model  = pfunc(x, b_V[bb], s_V[ss], c_V[cc], a0_V)
                with open('Pmax_V.txt', 'a') as f:
                    Pmax = model.max()
                    amax = x[np.where(model == model.max())]
                    f.write('{0:d} {1:d} {2:d} {3:.4f} {4:.4f}\n'.format(bb, ss, cc, Pmax, amax[0]))



#%%

calc_R = np.loadtxt('Pmax_R.txt')
calc_V = np.loadtxt('Pmax_V.txt')

print('R amp bin: {0:f}--{1:f}'.format(calc_R[:,0].min(), calc_R[:,0].max()))
print('R sin bin: {0:f}--{1:f}'.format(calc_R[:,1].min(), calc_R[:,1].max()))
print('R cos bin: {0:f}--{1:f}'.format(calc_R[:,2].min(), calc_R[:,2].max()))
print('V amp bin: {0:f}--{1:f}'.format(calc_V[:,0].min(), calc_V[:,0].max()))
print('V sin bin: {0:f}--{1:f}'.format(calc_V[:,1].min(), calc_V[:,1].max()))
print('V cos bin: {0:f}--{1:f}'.format(calc_V[:,2].min(), calc_V[:,2].max()))


#plt.plot(calc_R[:,0], 'o')
#plt.plot(calc_R[:,1], 'o')
#plt.plot(calc_V[:,0], 'o')
#plt.plot(calc_V[:,1], 'o')



#%%
#==============================================================================
# Testing: Plotting dP/da
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

plt.clf()
x = np.linspace(0.01,180,180)


for cpower in np.arange(0.01,2,0.2):
    for spower in np.arange(0.01,2,0.2):
        plt.plot(x, dPda0(x, spower, cpower, 20), 'r:', alpha=0.3)
        plt.plot(x, dPda0(x, spower, cpower, 18), 'b:', alpha=0.3)
        plt.plot(x, dPda0(x, spower, cpower, 22), 'g:', alpha=0.3)
        plt.grid(ls=':')

plt.plot(x, dPda0(x, 0.001, 0.001, 20), 'r', lw=4, label='cpower~0, spower~0')
plt.plot(x, dPda0(x, 0.001, 2, 20)    , 'g', lw=4, label='cpower~0, spower=2')
plt.plot(x, dPda0(x, 2, 0.001, 20)    , 'b', lw=4, label='cpower=2, spower~0')
plt.plot(x, dPda0(x, 2, 2, 20)        , 'k', lw=4, label='cpower=2, spower=2')
plt.text(0, -0.8 , 'a_0=18$^{\circ}$ (blue dot)' , color='blue')
plt.text(0, -0.95, 'a_0=20$^{\circ}$ (red dot)'  , color='red')
plt.text(0, -1.1 , 'a_0=22$^{\circ}$ (green dot)', color='green')

plt.text(15, -0.5, r'$P(\alpha) \propto (\sin \alpha)^{\rm spower} \
         (\cos \frac{\alpha}{2})^{\rm cpower} \sin (\alpha - \alpha_0)$',
         bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
plt.axhline(y=0)
plt.axvline(x=110)
plt.title(r'Plot of $ \frac{dP( \alpha )}{d \alpha}$: $P_{min}$ and $P_{max}$',
          y=1.01)
plt.xlabel('Phase angle ($^{\circ}$)')
plt.ylabel('dP/da (arbitrary unit)')
plt.legend()

plt.show()


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
plt.clf()
pi    = np.pi
x_rad = np.arange(0.00001, pi, 0.01)
def test(x, amplitude, beta, gamma):
    a = 2*(pi)**(1-beta)
    return -amplitude * sin(a * x**beta) * np.exp(gamma*x)

beta=np.arange(0.5,0.6,0.01)
for b in beta:
    plt.plot(x_rad, test(x_rad, 1, b, 1. ))
    plt.plot(x_rad, 2*test(x_rad, 1, b, 0.5), ls=':')
plt.plot(x_rad, 2*test(x_rad, 1, 0.3, 1), ls='--')

#%%
cellino = np.loadtxt(prefix+'cellino.csv', dtype=bytes).astype(str)
