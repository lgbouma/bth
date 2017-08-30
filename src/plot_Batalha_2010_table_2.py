import numpy as np, matplotlib.pyplot as plt, pandas as pd
from astropy.table import Table

# Presume bins are in 1000K, and the listed Teffs are the centers of the bins
df = pd.read_csv('Batalha_2010_Table_2.txt')

mmj = pd.read_csv('Mamajek_Teff_sptype.txt', delim_whitespace=True)
mmj = mmj[(mmj['Teff']<11000)&(mmj['Teff']>3000)]

#############################
# PLOT 1: Teff distribution #
#############################
plt.close('all')
f, ax = plt.subplots()

ax.step(df['teff'], df['val'], where='mid')
for teff, yval, SpT in list(zip(mmj['Teff'][::2],
                                np.ones_like(mmj['Teff'][::2])*8e4,
                                mmj['SpT'][::2])):

    t = ax.text(teff, yval, SpT, alpha=0.3, fontsize='xx-small', rotation=270)

ax.set(
    xlabel='teff',
    ylabel='number',
    yscale='log',
    ylim=[1e2,1e5],
    xlim=[3e3,11e3]
    )
ax.set_title(
    'table 2 Batalha+ 2010. KIC Teff distribution for planet target stars. '
    'SpTypes Pecault & Mamajek 2013',
    fontsize='xx-small'
    )

f.savefig('Batalha_2010_Table_2_distribution.pdf', dpi=300)

###############################
# PLOT 2: Teff cumulative sum #
###############################
plt.close('all')
f, ax = plt.subplots()

ax.step(df['teff'],
        np.cumsum(df['val'])/max(np.cumsum(df['val'])),
        where='mid')

for teff, yval, SpT in list(zip(mmj['Teff'][::2],
                                np.ones_like(mmj['Teff'][::2])*0.95,
                                mmj['SpT'][::2])):

    t = ax.text(teff, yval, SpT, alpha=0.3, fontsize='xx-small', rotation=270)

ax.set(
    xlabel='teff',
    ylabel='percent of {:d} stars'.format(int(max(np.cumsum(df['val'])))),
    yscale='linear',
    ylim=[0,1],
    xlim=[3e3,11e3]
    )

ax.set_title(
    'table 2 Batalha+ 2010. KIC Teff cumulative sum for planet target stars. '
    'SpTypes Pecault & Mamajek 2013',
    fontsize='xx-small'
    )

f.savefig('Batalha_2010_Table_2_cumulative_sum.pdf', dpi=300)
