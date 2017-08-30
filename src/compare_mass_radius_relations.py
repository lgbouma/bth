'''
Mon 28 Aug 2017 08:32:38 AM EDT

Current mass-radius relation for stars of binaries is different for singles.
Show how different.
'''

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import pickle

gic = pickle.load(
        open('/home/luke/local/selected_GIC_systems_kepler_analog.p','rb'))

singles = gic['singles']
doubles = gic['binaries']

R_s = singles['Rstar']
M_s = singles['mact']

R_d1 = doubles['R_1']
M_d1 = doubles['M_1']

R_d2 = doubles['R_2'][ (doubles['M_2'] > 0.7) & (doubles['M_2'] < 1.3) ]
M_d2 = doubles['M_2'][ (doubles['M_2'] > 0.7) & (doubles['M_2'] < 1.3) ]

R_d = np.concatenate((R_d1, R_d2))
M_d = np.concatenate((M_d1, M_d2))

plt.close('all')
f,ax = plt.subplots(figsize=(4,4))

ax.scatter(R_d, M_d, s=1, c='black', alpha=1, lw=0, rasterized=True,
        label='doubles', zorder=0)
ax.scatter(R_s, M_s, s=1, c='blue', alpha=0.7, lw=0, rasterized=True,
        label='singles', zorder=-2)

ax.set(xlabel='radius [$R_\odot$]', ylabel='mass [$M_\odot$]')

ax.legend(loc='best', fontsize='x-small')

f.tight_layout()

f.savefig('galaxia_compare_mass_radius.pdf',
        dpi=250, bbox_inches='tight')

