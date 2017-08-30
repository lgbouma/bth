'''
compare your "Galaxia input catalog" teff distribn w/ KIC's from Batalha et al
2010.
'''
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import pickle
from astropy.table import Table
from astropy import units as u, constants as c
from math import pi as π

# Presume bins are in 1000K, and the listed Teffs are the centers of the bins
df = pd.read_csv('../Batalha_2010_Table_2.txt')

mmj = pd.read_csv('../Mamajek_Teff_sptype.txt', delim_whitespace=True)
mmj = mmj[(mmj['Teff']<11000)&(mmj['Teff']>3000)]

gic = pickle.load(
        open('/home/luke/local/selected_GIC_systems_kepler_analog.p','rb'))

singles = gic['singles']
doubles = gic['binaries']

Teff_s = np.array(10**singles['teff'])

# Get Teff distribution for doubles (as if they were singles)
R_d = np.array(doubles['R_as_if_single'])*u.Rsun
M_d = np.array(doubles['M_as_if_single'])

import binary_distribution as bd
L_d = bd.get_L(M_d)
Teff_d = np.array((( L_d*u.Lsun / \
            (4 * π * R_d**2 * c.sigma_sb))**(1/4)).to(u.K))

gic_teff = np.concatenate((Teff_s, Teff_d))

gic_hist, gic_bin_edges = np.histogram( gic_teff,
        bins=np.append(np.arange(3e3,11e3,1e3),12e3),normed=False)

#############################
# PLOT 1: Teff distribution #
#############################
plt.close('all')
f, ax = plt.subplots(figsize=(4,4))

ax.step(df['teff'], df['val'], where='mid',
        label='KIC (B+2010 Table 2)', lw=1)

ax.step(gic_bin_edges[:-1], gic_hist, where='post',
        label='GIC', lw=1, zorder=1)

for teff, yval, SpT in list(zip(mmj['Teff'][::2],
                                np.ones_like(mmj['Teff'][::2])*1.8e5,
                                mmj['SpT'][::2])):

    t = ax.text(teff, yval, SpT, alpha=0.3, fontsize=5, rotation=270,
            verticalalignment='top')

ax.set(
    xlabel='$T_{\mathrm{eff}}$ [K]',
    ylabel='number',
    yscale='log',
    xlim=[3e3,11e3],
    ylim=[5e1, 2e5]
    )
ax.legend(
    loc=(0.6, 0.7),
    fontsize=6
    )

f.tight_layout()
f.savefig('KIC_vs_GIC_teff_distribn.pdf', dpi=250)
