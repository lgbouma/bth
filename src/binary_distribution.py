'''
fixed Rp. fixed primary L1. varying light ratio (L2/L1).

Find the stellar system properties (ONLY!)
'''
from __future__ import division, print_function

import matplotlib as mpl
mpl.use('Agg')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy import units as u, constants as c
from math import pi as π

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


def _make_distribution_plots(vl, doubles):

    savedir = '../results/analytic_model_plots/'
    # Plot distribution of doubles. First do volume limited case. 
    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    hist, bin_edges = np.histogram(vl['q'], bins=np.append(np.linspace(0,1,101),42),
            normed=True)
    ax.step(bin_edges[:-1], hist, 'k-', where='post',)
    ax.set(xlabel='$q = M_2/M_1$', ylabel='prob')
    ax.set_title(txt_vl, fontsize='small')
    f.tight_layout()
    f.savefig(savedir+'q_distribn_vol_limited.pdf', dpi=250, bbox_inches='tight')

    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    hist, bin_edges = np.histogram(vl['γ_R'],
            bins=np.append(np.linspace(0,1,501),42), normed=True)
    ax.step(bin_edges[:-1], hist, 'k-', where='post')
    ax.set(xlabel='$\gamma_R = L_2/L_1$',
           ylabel='prob')
    ax.set_title(txt_vl, fontsize='small')
    f.tight_layout()
    f.savefig(savedir+'gammaR_distribn_vol_limited.pdf', dpi=250, bbox_inches='tight')

    # Now do magnitude limited case. Maglimited mass ratio.
    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    hist, bin_edges = np.histogram(doubles['q'],
            bins=np.append(np.linspace(0,1,101),42), normed=True)
    ax.step(bin_edges[:-1], hist, where='post',
            label='numerical, empirical $L(M)$')
    # Analytic distribution from 17/09/24 result
    _q = np.arange(0,1+1e-3,1e-3)
    I_1 = 0.5060577377392849
    I_2 = 1.275894325140383
    norm = 9/(I_1*I_2)
    pdf_q_analytic = norm*I_1/9*(1+_q**3.5)**(3/2)
    pdf_q_analytic[_q<0.1] = 0
    ax.plot(_q, pdf_q_analytic, label='analytic, $L=M^{3.5}$')
    ax.legend(loc='upper left', fontsize='small')

    ax.set(xlabel='$q = M_2/M_1$', ylabel='prob')
    ax.set_title(txt_ml, fontsize='small')
    f.tight_layout()
    f.savefig(savedir+'q_distribn_mag_limited.pdf', dpi=250, bbox_inches='tight')

    # Maglimited light ratio
    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    hist, bin_edges = np.histogram(doubles['γ_R'],
            bins=np.append(np.linspace(0,1.01,101),42), normed=True)
    ax.step(bin_edges[:-1], hist, where='post',
            label='numerical, empirical $L(M)$')
    # Analytic distribution from 17/09/24 result
    from scipy.integrate import trapz
    _gammaR = np.arange(0,1+2e-5,2e-5)
    pdf_gammaR_analytic = (1+_gammaR)**(3/2)*_gammaR**(-5/7)
    pdf_gammaR_analytic[_gammaR<(0.1**(3.5))] = 0
    norm = trapz(pdf_gammaR_analytic, _gammaR)
    pdf_gammaR_analytic = pdf_gammaR_analytic/norm
    ax.plot(_gammaR, pdf_gammaR_analytic, label='analytic, $L=M^{3.5}$')
    ax.legend(loc='upper right', fontsize='small')
    ax.set(xlabel='$\gamma_R = L_2/L_1$',
           ylabel='prob',
           xlim=[0,1.05],
           ylim=[0.1,90],
           yscale='log')
    ax.set_title(txt_ml, fontsize='small')
    f.tight_layout()
    f.savefig(savedir+'gammaR_distribn_mag_limited.pdf', dpi=250, bbox_inches='tight')

    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    hist, bin_edges = np.histogram(doubles['γ_R'],
            bins=np.append(np.logspace(-4,1,11),42), normed=True)
    ax.step(bin_edges[:-1], hist, 'k-', where='post')
    ymax = 1.25
    ax.vlines(1/40, 0, ymax,
          label='Furlan+17 fig 21,'
          r'$\theta<1$arcsec: '
          '{:.1f}% of doubles'.format(
              len(doubles['γ_R'][doubles['γ_R']<(1/40)])\
              /len(doubles['γ_R'])*100)+\
          ' have $\Delta m > 4$'
          )
    ax.set(xlabel='$\gamma_R = L_2/L_1$',
           ylabel='prob',
           xlim=[5e-4,1.05],
           xscale='log')
    ax.legend(loc='upper left', fontsize=5)
    ax.set_title(txt_ml, fontsize='small')
    f.tight_layout()
    f.savefig(savedir+'gammaR_distribn_mag_limited_logx.pdf', dpi=250, bbox_inches='tight')


    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    hist, bin_edges = np.histogram(doubles['γ_R'],
            bins=np.append(np.linspace(0,1,501),42), normed=True)
    ax.plot(np.append(0, bin_edges[:-1]),
            np.append(0, np.cumsum(hist)/np.max(np.cumsum(hist))),
            'k-')
    ax.set(xlabel='$\gamma_R = L_2/L_1$',
           ylabel='prob',
           xlim=[0,1.05],
           ylim=[0,1.05])
    ax.set_title(txt_ml, fontsize='small')
    f.tight_layout()
    f.savefig(savedir+'gammaR_cumdistribn_mag_limited.pdf', dpi=250, bbox_inches='tight')

    outdf = pd.DataFrame({
                 'gamma_R_bins':np.append(0, bin_edges[:-1]),
                 'cdf':np.append(0, np.cumsum(hist)/np.max(np.cumsum(hist)))
                })
    outdf.to_csv('../data/gamma_R_cumulative_distribution_function.csv',
            index=False)

    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    hist, bin_edges = np.histogram(
            doubles['q'],
            bins=np.append(np.linspace(0,1,501),42),
            normed=True)
    ax.plot(np.append(0, bin_edges[:-1]),
            np.append(0, np.cumsum(hist)/np.max(np.cumsum(hist))),
            'k-')
    ax.set(xlabel='$q = M_2/M_1$',
           ylabel='prob',
           xlim=[0,1.05],
           ylim=[0,1.05])
    ax.set_title(txt_ml, fontsize='small')
    f.tight_layout()
    f.savefig(savedir+'q_cumdistribn_mag_limited.pdf', dpi=250, bbox_inches='tight')

    outdf = pd.DataFrame({
                 'q_bins':np.append(0, bin_edges[:-1]),
                 'cdf':np.append(0, np.cumsum(hist)/np.max(np.cumsum(hist)))
                })
    outdf.to_csv('../data/q_cumulative_distribution_function.csv',
            index=False)

    outdf = pd.DataFrame(
            {'q':np.array(doubles['q']),
             'gamma_R':np.array(doubles['γ_R'])
            })
    outdf.to_csv('../data/q_and_gamma_R.csv',
            index=False)



def draw_star_positions(d_max, N_sys, sys_type=None):
    '''
    Args:

        d_max: maximum distance out to which stars (of either single or binary
               class) are selected by mag-limit

        N_sys (int): the number of total stellar systems to draw

        sys_type: 'single' or 'binary' (star systems)

    Returns:

        a pandas dataframe with the positions, and their distance from the
        origin.

    '''

    assert sys_type == 'single' or sys_type == 'binary'
    assert type(N_sys) == int

    ## Inefficient, non-vectorized draw:
    #x_l,y_l,z_l,r_l = [], [], [], []
    #while len(r_l) < N_sys:
    #    if len(r_l) % 10 == 0:
    #        print('{:d}/{:d}'.format(len(r_l), N_sys))

    #    x = d_max * np.random.rand()
    #    y = d_max * np.random.rand()
    #    z = d_max * np.random.rand()

    #    if x**2 + y**2 + z**2 <= d_max**2:
    #        x_l.append(x.value)
    #        y_l.append(y.value)
    #        z_l.append(z.value)
    #        r_l.append( ((x**2 + y**2 + z**2)**(1/2)).value )

    # Vectorized draw:
    # vol sphere / vol cube = π/6 ~= half. Draw 10* the number of systems you
    # really need. Then cut.
    _x = d_max * np.random.rand( (10*N_sys) )
    _y = d_max * np.random.rand( (10*N_sys) )
    _z = d_max * np.random.rand( (10*N_sys) )
    _r = np.sqrt(_x**2 + _y**2 + _z**2)

    assert len(_x[_r < d_max]) > N_sys

    x = _x[_r < d_max][:N_sys]*u.pc
    y = _y[_r < d_max][:N_sys]*u.pc
    z = _z[_r < d_max][:N_sys]*u.pc
    r = _r[_r < d_max][:N_sys]*u.pc

    return pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'r': r,
        'sys_type': [sys_type for _ in range(len(r))]
        })


def draw_vol_limited_light_ratios(df, M_1, L_1):
    '''
    given a volume limited sample of double star systems returned by
    `draw_star_positions`, assign each system a mass ratio and light ratio.
    '''

    N_d_vl = len(df) # number of double star systems in volume limited sample

    # Rhagavan+ 2010, Fig 16. Close enough.
    q = np.random.uniform(low=0.1, high=1, size=N_d_vl)

    M_2 = q * M_1

    L_2 = get_L(M_2)

    γ_R = L_2/L_1
    try:
        assert np.max(γ_R) < 1
    except:
        import IPython; IPython.embed()

    ## OUTDATED METHOD
    ## Salaris & Cassisi 2005, Fig 5.11 & text. α is the mass-luminosity power
    ## exponent.
    #α = np.ones_like(q)
    #α[q < 0.5] *= 2.6
    #α[q >= 0.5] *= 4.5

    #γ_R = q**α

    df['q'] = q
    df['γ_R'] = γ_R

    return df


if __name__ == '__main__':

    np.random.seed(42)

    ##############
    # INSTRUMENT #
    ##############
    A = 50*u.cm**2
    λ_min = 500*u.nm
    λ_max = 1000*u.nm

    #################
    # SURVEY PARAMS #
    #################
    T_obs = 4*u.year
    x_min = 25 # minimum SNR for detection.

    # Zombeck 2007, p.103: Vega, V=0.03 has a wavelength-specific photon flux of
    # 1e3 ph/s/cm^2/angstrom. So for our instrument's bandpass, we
    # expect 5e5 ph/s/cm^2. For the magnitude limit, we want this in
    # erg/s/cm^2.

    # Zero points, and magnitude limit. m_lim of 5 or 6 runs fast, but has
    # Poisson noise. ~7 is needed for negligible Poisson noise.
    λ_nominal = 750*u.nm #FIXME for a "realistic" survey, might want to modify.
    energy_per_ph = c.h * c.c / λ_nominal
    m_0 = 0
    F_0 = 5e5 * energy_per_ph * (u.s)**(-1) * (u.cm)**(-2)
    m_lim = 11
    F_lim = F_0 * 10**(-2/5 * (m_lim - m_0))

    ######################
    # STELLAR POPULATION #
    ######################
    # Binary fraction. BF = n_d / n_s.
    BF = 0.44
    # Bovy (2017) gives n_tot = n_s + n_d. We want the number density of single
    # star systems, n_s, and the number density of double star systems, n_d.
    #n_tot = 4.76e-4 / (u.pc**3)
    n_tot = 5e-4 / (u.pc**3)
    n_s = n_tot / (1+BF)
    n_d = BF * n_s

    ################
    # SINGLE STARS #
    ################
    M_1 = 1*u.Msun
    L_1 = get_L(M_1.value)*u.Lsun
    R_1 = 1*u.Rsun
    T_eff1 = (1/c.sigma_sb * L_1 / (4*π*R_1**2))**(1/4)

    # Distance limit for single star systems.
    d_max_s = (L_1 / (4*π*F_lim) )**(1/2)
    # Number of single star systems in sample. It is an integer.
    N_s = int(np.floor( (n_s * 4*π/3 * d_max_s**3).cgs.value ))
    # Get positions for the single star systems. Compute their fluxes.
    singles = draw_star_positions(d_max_s.to(u.pc), N_s, 'single')
    singles['F'] = (L_1 / \
                (4*π* (np.array(singles['r'])*u.pc)**2 )).cgs

    # Make a volume-limited sample of binary star systems out to
    # sqrt(2)d_max_s. Then perform the magnitude cut to get the SNR-limited
    # sample.

    # Maximum binary light ratio.
    γ_R_max = 1
    # Maximum double star system luminosity
    L_d_max = (1+γ_R_max)*L_1
    # Maximum distance out to which the maximum luminosity double star system
    # could be detected. (...)
    d_max_d_max = (L_d_max / (4*π*F_lim) )**(1/2)
    # Number of double star systems in a volume limited sample out to
    # d_max_d_max.
    N_d_max = int(np.floor( (n_d * 4*π/3 * d_max_d_max**3).cgs.value ))

    vl = draw_star_positions(d_max_d_max.to(u.pc), N_d_max, 'binary')

    vl = draw_vol_limited_light_ratios(vl, M_1.value, L_1.value)

    vl['L_d'] = L_1 * (1 + vl['γ_R'])

    vl['F'] = ((np.array(vl['L_d'])*u.Lsun) / \
                (4*π* (np.array(vl['r'])*u.pc)**2 )).cgs

    doubles = vl[vl['F'] > F_lim.cgs.value]
    N_d = int(len(doubles))
    print('\n')
    print(len(singles))
    print(len(doubles))
    print('\n')

    txt_vl = 'Vol limited binary star sample: {:d} systems'.format(
            len(vl))
    print(txt_vl)
    print(vl[['q','γ_R']].describe())

    txt_ml = 'Mag limited binary star sample: {:d} systems'.format(
            len(doubles))
    print(txt_ml)
    print(doubles[['q','γ_R']].describe())

    _make_distribution_plots(vl, doubles)

    #########################################################################
    # YOU HAVE SINGLES AND DOUBLES. COMPUTE TRANSIT PROBABILITIES, ASSUMING #
    # EQUAL OCCURRENCE RATES                                                #
    #########################################################################

    P = 1*u.year
    a_1 = (P**2 * c.G * M_1 / (4*π*π))**(1/3)
    ones = np.ones_like(np.array(singles['r']))
    f_s = R_1/a_1
    singles['f_s'] = f_s * ones

    R_d1 = R_1
    a_d1 = a_1
    # Demircan & Kahraman 1991 mass-radius relation.
    R_d2 = np.array(1.06 * (np.array(doubles['q'])*M_1)**(0.945))*u.Rsun
    a_d2 = (P**2 * c.G * M_1 / (4*π*π))**(1/3)

    ones = np.ones_like(np.array(doubles['r']))
    doubles['f_d'] = 0.5 * ( (R_1/a_1 * ones).cgs +    (R_d2 / a_d2).cgs )

    # Fraction of stars in single star systems with planet of (R_p, P).
    Γ_ts = 0.5
    # Fraction per star in double star systems with planet of (R_p, P).
    Γ_td = 0.5

    N_det_s = N_s * Γ_ts * f_s
    N_det_s = N_det_s.cgs

    N_det_d = 2 * Γ_td * np.sum(doubles['f_d'])

    print('\nN_s: {:d}\n'.format(int(N_s)))
    print('N_d: {:d}\n'.format(int(N_d)))
    print('N_det,s: {:d}\n'.format(int(N_det_s)))
    print('N_det,d: {:d}\n'.format(int(N_det_d)))

    print('fraction misclassified')
    print( N_det_d / (N_det_s + N_det_d) )
    print('\n')

    print('ratio of geometric transit probabilities, f_d/f_s')
    print( ((1/N_d) * np.sum(doubles['f_d']) / f_s).cgs )
    print('\n')

    print('''\nYou never ran full numerics for this.
    But the above is actually all you need for the Rp rate.
    For the Rp-prime rate, you might want numerics''')
