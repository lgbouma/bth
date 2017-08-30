'''
we want a good mass-luminosity relation for dwarfs with masses less than 2
Msun.

this script takes reported masses and luminosities from a few sources, and then
derives a broken power law fit.
'''
from astropy.table import Table
import astropy.units as u
import pandas as pd, numpy as np, matplotlib.pyplot as plt

##############
# Wrangling. #
##############
# Collect M_v and masses from Benedict+ (2016).
tab10 = pd.read_csv('../data/Benedict_et_al_2016_table10_proc.txt',
                    delimiter=',')
tab11 = pd.read_csv('../data/Benedict_et_al_2016_table11_proc.txt',
                    delimiter=',')

_0 = pd.DataFrame(np.array(tab10[['Mv_pri', 'M_pri']]), columns=['Mv', 'M'])
_1 = pd.DataFrame(np.array(tab10[['Mv_sec', 'M_sec']]), columns=['Mv', 'M'])
b16 = pd.concat([_0, _1])

# Load in M_v to logL reported by E. Mamajek in
# http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt,
# downloaded 2017.08.02
mmj = pd.read_csv('../data/Mamajek_Mv_logL.txt', delimiter=',')

from scipy.interpolate import interp1d
f_mmj = interp1d(mmj['Mv'], mmj['logL'])

b16['logL'] = f_mmj(b16['Mv'])

# Load in Torres+ 2009: highest quality binary data in existence.
tab = Table.read('../data/Torres_Andersen_Gimenez_2009_table1.vot',
                 format='votable')
df = tab.to_pandas()
# Select things that at least resemble dwarfs. Some of these will be "peculiar"
# or have extra spectroscopic letters attached to them. That's OK.
df = df[ [('V' in spt.decode('ascii')) for spt in df['SpType'] ] ]

############
# Fitting. #
############
from scipy.optimize import curve_fit

def line_func(x, m, c):
    return m*x + c

mass = np.concatenate(( np.array(df['Mass']),
                        np.array(b16['M']) ))
luminosity = np.concatenate(( 10**np.array(df['logL']),
                              10**np.array(b16['logL']) ))

lo_M = mass[(mass < 0.5) & (mass >= 0.1)]
lo_L = luminosity[(mass < 0.5) & (mass >= 0.1)]
hi_M = mass[(mass >= 0.5) & (mass <= 1.2)]
hi_L = luminosity[(mass >= 0.5) & (mass <= 1.2)]

popt_lo, pcov_lo = curve_fit(line_func, np.log10(lo_M), np.log10(lo_L))

popt_hi, pcov_hi = curve_fit(line_func, np.log10(hi_M), np.log10(hi_L))

Mgrid = np.logspace(-1, 1, num=100)

L_fit_lo = 10**line_func(np.log10(Mgrid), *popt_lo)
L_fit_hi = 10**line_func(np.log10(Mgrid), *popt_hi)

m_lo, c_lo = popt_lo[0], popt_lo[1]
m_hi, c_hi = popt_hi[0], popt_hi[1]

# Find overlap point
x_int = (c_hi - c_lo) / (m_lo - m_hi)
M_int = 10**x_int
L_int = 10**line_func(x_int, *popt_lo)

print('m_lo: {:.16f}'.format(m_lo))
print('c_lo: {:.16f}'.format(c_lo))
print('m_hi: {:.16f}'.format(m_hi))
print('c_hi: {:.16f}'.format(c_hi))
print('M at merge: {:.16f}'.format(M_int))
print('L at merge: {:.16f}'.format(L_int))

def get_L(M):

    m_lo = 1.8818719873988132
    c_lo = -0.9799647314108376
    m_hi = 5.1540712426599882
    c_hi = 0.0127626185389781
    M_merge = 0.4972991257826812
    L_merge = 0.0281260412126928

    L = np.ones_like(M)

    # Method 1: totally fine.
    #lo_mask = M < M_merge
    #L[lo_mask] = 10**(np.log10(M[lo_mask])*m_lo + c_lo)

    #hi_mask = M >= M_merge
    #L[hi_mask] = 10**(np.log10(M[hi_mask])*m_hi + c_hi)

    # Method 2: cuter.
    L_lo = 10**(np.log10(M)*m_lo + c_lo)
    L_hi = 10**(np.log10(M)*m_hi + c_hi)

    L = np.maximum(L_lo, L_hi)

    return L


##############
# Make plot. #
##############
f, ax = plt.subplots(figsize=(4,4))

ax.scatter(df['Mass'], 10**np.array(df['logL']), color='k', alpha=0.1,
        lw=1, marker='x', label='TAG09, M > 1.2 Msun')

ax.scatter(df['Mass'][df['Mass']<=1.2],
        10**np.array(df['logL'][df['Mass']<=1.2]),
        color='k', alpha=0.6, lw=1,
        marker='x', label='TAG09, M < 1.2 Msun')

ax.plot(Mgrid, L_fit_hi, 'k-', label='high fit (0.5-1.2 Msun)', alpha=0.3)

ax.scatter(b16['M'], 10**np.array(b16['logL']), color='blue', alpha=0.4,
        lw=1, marker='x', label='B+16, M > 0.1 Msun')

ax.plot(Mgrid, L_fit_lo, 'b-', label='low fit (0.1-0.5Msun)', alpha=0.3)

ax.plot(Mgrid, get_L(Mgrid), 'g-', label='L(M) fit', alpha=0.7)

ax.legend(loc='upper left', fontsize='xx-small', scatterpoints=1,
        frameon=False)

ax.set_xlabel('mass [Msun]')
ax.set_ylabel('luminosity [Lsun]')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim([0.1,1e1])
ax.set_ylim([1e-3,1e3])

f.tight_layout()

f.savefig('mass_luminosity.pdf', dpi=300, bbox_inches='tight')
