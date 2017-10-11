'''
expressions for the probability density of observing a planet with apparent
radius $r_a$, in model 2

also makes plot for paper
'''
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import matplotlib as mpl
mpl.use('pgf')
pgf_with_custom_preamble = {
    'pgf.texsystem': 'pdflatex', # xelatex is default; i don't have it
    'font.family': 'serif', # use serif/main font for text elements
    'text.usetex': True,    # use inline math for ticks
    'pgf.rcfonts': False,   # don't setup fonts from rc parameters
    'pgf.preamble': [
        '\\usepackage{amsmath}',
        '\\usepackage{amssymb}'
        ]
    }
mpl.rcParams.update(pgf_with_custom_preamble)

import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.integrate import trapz
from scipy.integrate import quad
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, \
    LSQUnivariateSpline, UnivariateSpline

global α, β
β = 0 # p(q)~q^β vol limited
α = 3.5 # M~L^α
r_p = 1 #Re, Rj - units don't matter

f,ax = plt.subplots(figsize=(4,4))

#PRIMARIES
#boxed eqn 17/10/10.1 bottom. Define a lambda function for p1_ra so that you
#can use quad to integrate over the inf.
p1_ra = lambda r_a: (r_p/r_a)**3 * r_p**2/(r_a**3) * \
                    ( (r_p/r_a)**2 -1 )**((β-α+1)/α)

r_a = np.linspace(r_p*2**(-1/2), r_p, num=int(1e5))

p1_ra_vals = p1_ra(r_a)/quad(p1_ra, r_p/np.sqrt(2), r_p)[0]

ax.plot(r_a/r_p, p1_ra_vals, label='$p_1(r_a)$')

#SECONDARIES
#case 2, eqn 17/10/10.2 middle.
#probability density if we ignore the secondary stellar radius correction
p2_ra_case2 = lambda r_a: \
            (( (r_p/r_a)**2 -1 )**(-1/α))**β * \
            (1+(( (r_p/r_a)**2 -1 )**(-1/α))**α)**(3/2) * \
            ( (r_p/r_a)**2 -1 )**((-1-α)/α) * \
            r_p**2/(r_a**3)

r_a = np.linspace(0, r_p*2**(-1/2), num=int(4e5))

p2_ra_vals_case2 = p2_ra_case2(r_a)/quad(p2_ra_case2, 0, r_p*2**(-1/2))[0]

ax.plot(r_a/r_p, p2_ra_vals_case2,
        label='$p_2(r_a)$ (ignoring $R_2$ correction)')

#SECONDARIES
#case 1, eqn 17/10/10.2 top.
#probability density if we include the secondary stellar radius correction.
#seminanalytically was no bueno; go full Monte Carlo.

#draw q
q_grid = np.arange(1e-7, 1+1e-7, 1e-7)
prob_ml_q = q_grid**β * (1+q_grid**α)**(3/2)
prob_ml_q /= trapz(prob_ml_q, q_grid)

# inverse transform sampling to get samples of q
cdf_ml_q = np.append(0, np.cumsum(prob_ml_q)/np.max(np.cumsum(prob_ml_q)))
func = interp1d(cdf_ml_q, np.append(0, q_grid))
q_samples = func(np.random.uniform(size=int(5e7)))

ra_samples = r_p * (1/q_samples) * (1 + q_samples**(-α))**(-1/2)

ra_bins = np.linspace(0,np.max(ra_samples),num=201)

hist, bin_edges = np.histogram(
        ra_samples,
        bins=np.append(ra_bins,42),
        normed=True)

hist /= trapz(hist, ra_bins)
assert np.isclose(trapz(hist, ra_bins), 1), 'hist is normalized pdf'

#ax.step(bin_edges[:-1], hist, where='post',
#    label='secondaries (including $R_2$ correction), histogram')

#interpolate over bin mid-points

#WORSE METHODS
#knots = np.linspace(0+1e-3, np.max(ra_samples)-1e-3, num=8)
#knots = [0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
#lsqus = LSQUnivariateSpline(
#        ra_bins[:-1] + np.diff(ra_bins)/2,
#        hist[:-1],
#        knots,
#        k=3,
#        ext=0
#        )
#ius = InterpolatedUnivariateSpline(
#        ra_bins[:-1] + np.diff(ra_bins)/2,
#        hist[:-1],
#        k=1,
#        ext=0
#        )
#func = interp1d(ra_bins[:-1] + np.diff(ra_bins)/2, hist[:-1], fill_value='extrapolate')

#CHOSEN METHOD
spl = UnivariateSpline(
        ra_bins[:-1] + np.diff(ra_bins)/2,
        hist[:-1]
        )

spl.set_smoothing_factor(5e-3)

ra_grid = np.linspace(0,np.max(ra_samples),num=int(4e5))

p2_ra_vals_case1 = spl(ra_grid)/trapz(spl(ra_grid), ra_grid)
p2_ra_vals_case1[0] = 0

ax.plot(ra_grid/r_p, p2_ra_vals_case1,
        label='$p_2(r_a)$ (including $R_2$ correction)')

df = pd.DataFrame({'r_a':ra_grid, 'p2_ra_case1':p2_ra_vals_case1})
df.to_csv('p2_ra_model2_apparent_radius_prob_density_secondaries.csv',
        index=False)


##########
##########

ax.legend(loc='best',fontsize='x-small')

ax.set_xlabel('apparent / true planet radius')
ax.set_ylabel('probability density')

#ax.set_title(
#'''each line normalized independently; lines above top diverge.'''
#)
#ax.set_ylim([0,20])
ax.set_yscale('log')
ax.set_ylim([1e-1,1e2])

ax.tick_params(which='both', direction='in', zorder=0)

f.tight_layout(h_pad=0)
f.savefig('prob_r_a.pdf', dpi=300, bbox_inches='tight')
