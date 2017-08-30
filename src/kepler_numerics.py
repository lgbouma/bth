#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
>  Can we ignore binarity when mapping KIC stars onto synthetic local stars?

Or better yet

> If we assume every KIC star is single, what OOM of error do we make in
> occurrence rates estimated for planets of different sizes (& periods)?

Answer the latter question with Monte Carlo simulations of the Kepler field.

with TRILEGAL/Galaxia:
    * construct the Kepler field (w/ known binarity properties, how?)
    * introduce various planet populations
    * numerically assess the error.
    * Compare w/ analytic results.
'''

from __future__ import print_function, division
import sys, os
import pickle
import numpy as np, pandas as pd

from astropy import units as u, constants as c
from astropy.coordinates import SkyCoord

global DATADIR
DATADIR = '/home/luke/local/GalaxiaData/kepler_synthesis/'

def T_dur(P, ρ_star, b):
    '''
    Compute the transit duration given orbital period, host star density, and
    impact parameter.
    E.g., Sullivan et al 2015, Eq 10. (Assumes zero eccentricity).
    '''

    from math import pi as π
    ρ_sun = 3*u.Msun/(4*π*u.Rsun**3)
    return 13*u.hr * (P/(1*u.year))**(1/3) * \
                     (ρ_star/ρ_sun)**(-1/3) * \
                     np.sqrt(1-b*b)


def are_synthetic_stars_on_silicon():
    '''
    Galaxia model was run with 200 sq deg FoV, centered at the true Kepler
    field center. Select synthetic Kepler field stars by comparing coordinates
    to viable KIC stars.
    '''
    import K2fov
    # CITE: http://adsabs.harvard.edu/abs/2016ascl.soft01009M, Mullaly,
    # Barclay, Barentsen 2017.

    gal = pickle.load(
            open(DATADIR+'keplersynthesis.p', 'rb'), encoding='latin1')

    c = SkyCoord(frame='galactic', l=gal['glon']*u.deg, b=gal['glat']*u.deg)

    df = pd.DataFrame(np.transpose([np.array(c.fk5.ra), np.array(c.fk5.dec)]),
                      columns=['ra','dec'])

    df['mags'] = np.ones_like(gal['glon'])

    df.to_csv(DATADIR+'raw_galaxia_coords.csv', index=False, header=False)

    # skimming source of K2fov, they *do* tabulate the original Kepler field,
    # as "preliminary" (for a single season). Access with "campaign number"
    # 1000.
    K2fov.K2onSilicon(
            DATADIR+'raw_galaxia_coords.csv',
            1000,
            do_nearSiliconCheck=True)


def select_synthetic_stars():

    df = pd.read_csv('target_siliconFlag.csv',
                     header=None,
                     names=['ra','dec','kepmag','on_silicon']
                     )
    # Take any targets "on" or "near" silicon. "Near" is whatever is defined by
    # k2fov as "near".
    sel = df[(df['on_silicon'] == 2) | (df['on_silicon'] == 1)]


def select_kic_stars():
    # Already ran ~/local/parse_kic.sh
    kic = pd.read_csv('/home/luke/local/kic_wanted_cols.txt',
                      delimiter=' ',
                      header=0)

    # cf http://certificate.ulo.ucl.ac.uk/modules/year_one/NASA_Kepler/fov.html
    # A region of the extended solar neighborhood in the Cygnus region along
    # the Orion arm centered on galactic coordinates (76.532562,+13.289502)
    # This is RA 291.05, dec 44.6.
    center = SkyCoord(76.532562*u.deg, 13.289502*u.deg, frame='galactic')

    # nb. "kic_degree_ra" and not the default nutty "kic_ra" which is in
    # decimal hours.
    coords = SkyCoord(np.array(kic['kic_degree_ra'])*u.deg,
                      np.array(kic['kic_dec'])*u.deg,
                      frame='fk5')

    sep = coords.separation(center)

    sel = kic[ (sep < 7.5*u.deg) & (kic['kic_rmag']<14) ]

    sel.to_csv('/home/luke/local/selected_kic_stars.csv', index=False)


def select_galaxia_stars(is_kepler_analog=False):

    gal = pickle.load(
            open(DATADIR+'keplersynthesis.p', 'rb'), encoding='latin1')

    c = SkyCoord(frame='galactic', l=gal['glon']*u.deg, b=gal['glat']*u.deg)

    center = SkyCoord(76.532562*u.deg, 13.289502*u.deg, frame='galactic')

    sep = c.separation(center)

    keys = [k for k in gal.keys() if k not in ['log', 'center']]
    df = pd.DataFrame()
    for k in keys:
        df[k] = gal[k]

    # These numbers from the Galaxia manual (coefficients for Schelgel
    # extinction map)
    ugriz_fi = [5.155, 3.793, 2.751, 2.086, 1.479]

    # n.b. "sdss_r" is absolute SDSS r mag. Compute apparent mags for all
    # bands as follows:
    for b_ind, b in enumerate(['u','g','r','i','z']):

        df['apparent_'+b] = df['sdss_'+b] + \
                           5*np.log10(100 * df['rad']) + \
                           df['exbv_schlegel'] * ugriz_fi[b_ind]

    mag_limit = 14 if not is_kepler_analog else 17
    sel = df[ (sep < 7.5*u.deg) & (df['apparent_r']<mag_limit) ]

    savedir = '/home/luke/local/'
    savename = 'selected_galaxia_stars.csv' if not is_kepler_analog \
            else 'selected_galaxia_stars_kepler_analog.csv'

    sel.to_csv(savedir+savename, index=False)



def plot_distributions():
    '''
    You have KIC and Galaxia stars of r<14, within 7.5deg of the Kepler FoV
    center.

    Plot distributions of apparent magnitudes, stellar properties, etc.
    '''

    import matplotlib.pyplot as plt
    from math import pi

    gal = pd.read_csv('~/local/selected_galaxia_stars.csv')
    kic = pd.read_csv('~/local/selected_kic_stars.csv')

    gal['rstar'] = np.sqrt( np.array(10**gal['lum'])*u.Lsun / \
                       (4*pi * c.sigma_sb * \
                       (np.array((10**gal['teff']))*u.K)**4)).to(u.Rsun)

    # To get a KIC mass, use the reported logg...
    kic['mstar'] = (10**(np.array(kic['kic_logg']))*u.cm/(u.s**2) * \
                   (np.array(kic['kic_radius'])*u.Rsun)**2 / \
                   (c.G)).to(u.Msun)

    dims = {
            'rstar_low': ['rstar', 'kic_radius'],
            'rstar_more': ['rstar', 'kic_radius'],
            'rstar_all': ['rstar', 'kic_radius'],
            'mstar':  ['mact', 'mstar'],
            'teff':  ['teff', 'kic_teff'],
            'logg':  ['grav', 'kic_logg'],
            'gmag':  ['apparent_g', 'kic_gmag'],
            'rmag':  ['apparent_r', 'kic_rmag'],
            'imag':  ['apparent_i', 'kic_imag'],
            'zmag':  ['apparent_z', 'kic_zmag']
           }
    bounds = {
            'rstar_low': np.arange(0, 3+0.1, 0.1),
            'rstar_more': np.arange(0, 15+0.25, 0.25),
            'rstar_all': np.logspace(-2, 2, 21),
            'mstar':  np.arange(0, 4+0.25, 0.25),
            'teff':  np.arange(0, 1e4+2.5e2, 2.5e2),
            'logg':  np.arange(3, 6+0.1, 0.1),
            'gmag':  np.arange(0, 15+0.25, 0.25),
            'rmag':  np.arange(0, 15+0.25, 0.25),
            'imag':  np.arange(0, 15+0.25, 0.25),
            'zmag':  np.arange(0, 15+0.25, 0.25)
           }
    colors = ['black', 'green']
    labels = ['Galaxia', 'KIC']

    for dk in ['rstar_low','rstar_more','rstar_all','mstar','teff','logg',
               'gmag','rmag','imag','zmag']:

        print(dk + 5*'.')
        plt.close('all')
        f,ax = plt.subplots(figsize=(4,4))

        for cat_ix, cat in enumerate([gal, kic]):

            arr = np.array(cat[dims[dk][cat_ix]])
            if dk == 'teff' and cat_ix == 0: # galaxia Teff format is log10
                arr = 10**arr
            arr = arr[np.isfinite(arr)]

            try:
                hist, bin_edges = np.histogram( arr, bins=bounds[dk],
                                                normed=True )
            except:
                import IPython; IPython.embed()
            if dk == 'rstar_all':
                hist, bin_edges = np.histogram( np.log10(arr), bins=bounds[dk],
                                                normed=True )

            ax.step(bin_edges[:-1], hist, where='post', color=colors[cat_ix],
                    label=labels[cat_ix])

        ax.legend(loc='best', fontsize='x-small')
        ax.set(xlabel=dk, ylabel='prob')
        if 'mag' in dk:
            ax.set_xlim([5,15])
        elif dk == 'teff':
            ax.set_xlim([2e3,10e3])
        elif dk == 'mstar':
            ax.set_xlim([0,4])
        elif dk == 'rstar_all':
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim([1e-1, 1e2])

        f.tight_layout()

        savedir = '../results/kic_v_galaxia_distrbns/'
        f.savefig(savedir+'{:s}_distribn.pdf'.format(dk), dpi=250,
                bbox_inches='tight')



def add_binaries_to_galaxia(is_kepler_analog=False):
    '''
    The Galaxia magnitude distributions are "not that bad". So keep the
    reported magnitudes as "system magnitudes" (total flux from system).
    We assume in all cases that `gal` is a magnitude limited sample. Then to
    choose binaries:

    * assume a binary fraction (e.g., from Duchene & Kraus 2013, or more
      directly from Raghavan et al 2010).
    * ignore higher order multiples
    * if a star is drawn to be a "binary":
      - its reported magnitude becomes a system magnitude.
      - draw q ~ pdf_magnitude_limited(q)
        (this works so long as the pdf is independent of primary mass. We
        assume this to be the case over 0.7-1.3Msun)
      - draw M_primary ~ pdf_single_star_primary_masses
        This is necessary b/c the primary mass is not known. L_d is known.
      - compute light ratio from empirical mass-luminosity relation
      - L_d known (inherited) -> get L_1,L_2,M_1,M_2.
      - Taking the empirical, messy Eq 57 from the same memo as the stellar
        mass-radius relation, we get a radius for each star in the system.


    '''

    from math import pi
    from scipy.interpolate import interp1d
    from scipy.integrate import trapz
    import binary_distribution as bd

    loadname = 'selected_galaxia_stars.csv' if not is_kepler_analog \
            else 'selected_galaxia_stars_kepler_analog.csv'
    gal = pd.read_csv('~/local/'+loadname)

    gal['lum'] = 10**gal['lum']

    # In the end, we are only interested in "solar-like". Take this to mean
    # "got initially assigned a mass between 0.7 and 1.3 Msun". (To match the
    # bounds reported by Duchene and Kraus, for which a BF of 0.45 is
    # reasonable).
    # Now OFC the "straight-forward" thing is to select on `mact`, but when we
    # later introduce our empirical mass-luminosity function, we see it's not
    # good. We get primary masses from 0.6 to 5.8Msun, with a 75th percentile
    # at 2.10Msun, 50th percentile at 1.37Msun.
    # So instead, we will post-select based on masses that come from Galaxia
    # luminosities (less "direct" than from masses, since in Galaxia the
    # luminosity is a by-product of the isochrones + the evolved mass function
    # [IMF + stellar evoln prescription]). However, initial selection based on
    # upper and lower luminosity limits works.

    minimum_mass = 0.7
    maximum_mass = 1.3
    minimum_lum = bd.get_L(minimum_mass)
    maximum_lum = bd.get_L(maximum_mass)

    # The system luminosity must be such that we think it is "solar".
    gal = gal[ (gal['lum'] < maximum_lum) & (gal['lum'] > minimum_lum)]

    # Select which "stars" are actually binaries.
    BF = 0.45
    mask = ( np.random.rand(len(gal)) < BF )

    bins = gal[mask]
    singles = gal[~mask]
    singles['Rstar'] = np.sqrt( np.array(singles['lum'])*u.Lsun / \
                       (4*pi * c.sigma_sb * \
                       (np.array((10**singles['teff']))*u.K)**4)).to(u.Rsun)

    bins = bins.rename(index=str,
            columns={'lum': 'L_d',
                     'apparent_u': 'sys_apparent_u',
                     'apparent_g': 'sys_apparent_g',
                     'apparent_r': 'sys_apparent_r',
                     'apparent_i': 'sys_apparent_i',
                     'apparent_z': 'sys_apparent_z',
                     })

    # Inverse transform sampling on the mass ratio distribution. This data
    # file is made by `binary_distribution.py` -- it is the numerical
    # cumulative distribution function of the light ratio distribution. To
    # generate samples from prob(q), we generate uniform random samples `u`
    # and invert the cdf: samples = cdf^{-1}(u). See e.g., Adrian
    # Price-Whelan's ipython notebook about this.

    ## METHOD 1: inverse sampling. Weirdly, it makes a mistake -- the resulting
    ## distribution of mass ratios has a tail at the low q end (wtf?)
    #df = pd.read_csv('../data/q_cumulative_distribution_function.csv')
    #func = interp1d(np.array(df['cdf']), np.array(df['q']))
    #mass_ratios = func(np.random.uniform(size=len(bins)))

    # METHOD 2: rejection sampling, AKA Monte Carlo sampling over the volume.
    print('...beginning rejection sampling')
    N = int(5e5) if not is_kepler_analog else int(3e6)
    x_sample = np.random.uniform(0, 1, size=N)
    y_sample = np.random.uniform(0, 2.5, size=N)

    df = pd.read_csv('../data/q_and_gamma_R.csv')
    hist, _ = np.histogram(
            df['q'],
            bins=np.append(np.linspace(0,1,501),42),
            normed=True)

    assert trapz(hist, np.linspace(0,1,501)) == 1, 'hist is normalized pdf'

    func = interp1d(np.linspace(0,1,501), hist)

    idx = y_sample < func(x_sample)
    x = x_sample[idx]

    mass_ratios = x[:len(bins)]

    bins['q'] = mass_ratios
    print('...completed rejection sampling')

    # At this point, we want the light ratio. However, for our broken-power law
    # L(M), γ_R(q) becomes multi-valued -- AKA not a thing that can be written
    # as a pure function of `q` (even with the total luminosity, L_d, known!).
    # The necessary workaround, at least in the case of q<5/7, is either to
    # specify the total mass, or the mass of the primary. I'll specify the mass
    # of the primary, by drawing it from the distribution of single star
    # masses.  Note that this method, though it leads to a reasonable
    # distribution of primary masses (i.e. we will still have the correct
    # marginalized distribution of the mass ratio, the light ratio, still have
    # q=M2/M1 always, and L_d = L_1+L_2 always), will have WRONG correlations.
    # I.e., the 2D distribution of primary mass vs host distance will be WRONG.
    # This might matter for planet detection numbers.

    # METHOD 1: produces not-too-biased q vs M_2/M_1.  Analytic solution works
    # for q>=5/7. Otherwise, sample from single star mass distribution. Do it
    # with inverse transform sampling.
    m_lo = 1.8818719873988132
    c_lo = -0.9799647314108376
    m_hi = 5.1540712426599882
    c_hi = 0.0127626185389781

    M2_gt_pt5_mask = (np.array(bins['q']) >= 5/7)

    light_ratio = np.zeros_like(np.array(bins['q']))

    light_ratio[M2_gt_pt5_mask] = np.array(bins[M2_gt_pt5_mask]['q'])**m_hi

    hist, bin_edges = np.histogram(
            singles['mact'],
            bins=np.append(np.linspace(0,15,501),42),
            normed=True)
    df = pd.DataFrame(
            {'mass':np.append(0, bin_edges[1:]),
             'cdf':np.append(0, np.cumsum(hist)/np.max(np.cumsum(hist)))
            })

    func = interp1d(np.array(df['cdf']),
                    np.array(df['mass']))

    lt_pt5_primary_masses = func(
            np.random.uniform(size=len(bins[~M2_gt_pt5_mask])))

    lt_pt5_secondary_masses = np.array(bins[~M2_gt_pt5_mask]['q']) * \
                              lt_pt5_primary_masses

    lt_pt5_primary_lum = bd.get_L(np.array(lt_pt5_primary_masses))
    lt_pt5_secondary_lum = bd.get_L(np.array(lt_pt5_secondary_masses))

    light_ratio[~M2_gt_pt5_mask] = lt_pt5_secondary_lum/lt_pt5_primary_lum

    bins['γ_R'] = light_ratio
    bins['L_1'] = bins['L_d'] / (1 + bins['γ_R'])
    bins['L_2'] = bins['L_1'] * bins['γ_R']

    # Invert the `get_L` function (Fig 2) to get mass as func of luminosity.
    _mass_arr = np.arange(0.001, 10+0.001, 0.001)
    _L_arr = bd.get_L(_mass_arr)
    func = interp1d(_L_arr, _mass_arr)

    M_1_full = func(np.array(bins['L_1']))
    M_2_full = func(np.array(bins['L_2']))
    M_1_full[~M2_gt_pt5_mask] = lt_pt5_primary_masses
    M_2_full[~M2_gt_pt5_mask] = lt_pt5_secondary_masses

    bins['M_1'] = M_1_full
    bins['M_2'] = M_2_full

    ## #NOTE: METHOD 2: produces biased q vs M_2/M_1 (but uses more information?!)
    ## Ld_gt_Lpt7_mask = (np.array(bins['L_d']) >= bd.get_L(0.7))

    ## light_ratio[(~M2_gt_pt5_mask) & Ld_gt_Lpt7_mask] = \
    ##             np.array(bins[(~M2_gt_pt5_mask) & Ld_gt_Lpt7_mask]['q'])**m_hi

    ## hist, bin_edges = np.histogram(
    ##         singles['mact'],
    ##         bins=np.append(np.linspace(0,15,501),42),
    ##         normed=True)
    ## df = pd.DataFrame(
    ##         {'mass':np.append(0, bin_edges[1:]),
    ##          'cdf':np.append(0, np.cumsum(hist)/np.max(np.cumsum(hist)))
    ##         })
    ## func = interp1d(np.array(df['cdf']),
    ##                 np.array(df['mass']))

    ## lt_pt5_primary_masses = func(
    ##         np.random.uniform(
    ##             size=len(bins[(~M2_gt_pt5_mask) & (~Ld_gt_Lpt7_mask)])))

    ## lt_pt5_secondary_masses = np.array(
    ##         bins[(~M2_gt_pt5_mask) & (~Ld_gt_Lpt7_mask)]['q']) \
    ##         * lt_pt5_primary_masses

    ## #bins['M_1'] = primary_masses
    ## #bins['M_2'] = bins['q'] * bins['M_1']

    ## lt_pt5_primary_lum = bd.get_L(np.array(lt_pt5_primary_masses))
    ## lt_pt5_secondary_lum = bd.get_L(np.array(lt_pt5_secondary_masses))
    ## #bins['L_1'] = bd.get_L(np.array(bins['M_1']))
    ## #bins['L_2'] = bd.get_L(np.array(bins['M_2']))

    ## light_ratio[[(~M2_gt_pt5_mask) & (~Ld_gt_Lpt7_mask)]] = \
    ##         lt_pt5_secondary_lum/lt_pt5_primary_lum
    ## #bins['γ_R'] = bins['L_2']/bins['L_1']

    ## bins['γ_R'] = light_ratio
    ## bins['L_1'] = bins['L_d'] / (1 + bins['γ_R'])
    ## bins['L_2'] = bins['L_1'] * bins['γ_R']

    ## # if M_2 > 0.5, then the following expression holds for M_2.
    ## M_2_temp = ( bins['L_d'] * 10**(-c_hi) * bins['q'] / (1 + bins['q']) )**(1/m_hi)

    ## # anywhere M_2_temp is < 0.5, we have a contradiction. The expression was
    ## # only valid for M_2 > 0.5, and if it gives a value <0.5, it means we need
    ## # a different expression for M_2:

    ## inds = M_2_temp < 0.5
    ## M_2_temp[inds] = np.ones_like(M_2_temp[inds]) * np.nan

    ## hist, bin_edges = np.histogram(
    ##         singles['mact'],
    ##         bins=np.append(np.linspace(0,15,501),42),
    ##         normed=True)
    ## df = pd.DataFrame(
    ##         {'mass':np.append(0, bin_edges[1:]),
    ##          'cdf':np.append(0, np.cumsum(hist)/np.max(np.cumsum(hist)))
    ##         })
    ## func = interp1d(np.array(df['cdf']),
    ##                 np.array(df['mass']))

    ## primary_masses = func(np.random.uniform(size=len(bins)))

    ## bins['M_1'] = primary_masses
    ## bins['M_2'] = bins['q'] * bins['M_1']

    ## bins['L_1'] = bd.get_L(np.array(bins['M_1']))
    ## bins['L_2'] = bd.get_L(np.array(bins['M_2']))
    ## bins['γ_R'] = bins['L_2']/bins['L_1']

    ## # Invert the `get_L` function (Fig 2) to get mass as a function of
    ## # luminosity.
    ## _mass_arr = np.arange(0.001, 10+0.001, 0.001)
    ## _L_arr = bd.get_L(_mass_arr)
    ## func = interp1d(_L_arr, _mass_arr)

    ## bins['M_1'] = func(np.array(bins['L_1']))
    ## bins['M_2'] = func(np.array(bins['L_2']))
    ## #bins['q'] = bins['M_2']/bins['M_1']
    ## #NOTE: END METHOD 2

    # Eq 57 of toy_analytic_surveys_170804.pdf, from Demircan & Kahraman 1991.
    # Does affect transit probabilities.
    bins['R_1'] = 1.06 * np.array(bins['M_1'])**0.945
    bins['R_2'] = 1.06 * np.array(bins['M_2'])**0.945

    # Select things inside our mass range.
    minimum_mass = 0.7
    maximum_mass = 1.3

    singles = singles[(singles['mact'] > minimum_mass) & \
                      (singles['mact'] < maximum_mass)]
    bins = bins[(bins['M_1'] > minimum_mass) & \
                (bins['M_1'] < maximum_mass)]

    # Plot distributions
    import matplotlib.pyplot as plt
    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    hist, bin_edges = np.histogram( bins['q'],
            bins=np.append(np.linspace(0,1,51),42),normed=True)
    ax.step(bin_edges[:-1], hist, where='post',
            label='saved q (from samples)', c='black', lw=1, zorder=1)
    hist, bin_edges = np.histogram( bins['M_2']/bins['M_1'],
            bins=np.append(np.linspace(0,1,51),42),normed=True)
    ax.step(bin_edges[:-1], hist, where='post',
            label='actual M2/M1 (from samples)', c='blue', lw=3, zorder=0)
    ax.set(xlabel='q', ylabel='prob')
    ax.legend(loc='best', fontsize='x-small')
    ax.set_title('galaxia binaries', fontsize='small')
    f.tight_layout()
    savedir='../results/galaxia_binaries/' if not is_kepler_analog \
            else '../results/galaxia_binaries_kepler_analog/'
    f.savefig(savedir+'galaxia_q_distribn_mag_limited.pdf',
            dpi=250, bbox_inches='tight')

    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    hist, bin_edges = np.histogram( bins['γ_R'],
            bins=np.append(np.linspace(0,1,201),42),normed=True)
    ax.step(bin_edges[:-1], hist, 'k-', where='post')
    ax.set(xlabel='$\gamma_R$', ylabel='prob')
    ax.set_title('galaxia binaries', fontsize='small')
    f.tight_layout()
    f.savefig(savedir+'galaxia_gammaR_distribn_mag_limited.pdf',
            dpi=250, bbox_inches='tight')

    # Plot scatters
    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    ax.scatter(bins['L_1'], bins['L_2'], s=1, c='black', alpha=1, lw=0,
            rasterized=True)
    ax.set(xlabel='L_1', ylabel='L_2')
    ax.set_title('galaxia binaries', fontsize='small')
    f.tight_layout()
    f.savefig(savedir+'galaxia_gammaR_scatter_luminosities.pdf',
            dpi=250, bbox_inches='tight')

    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    ax.scatter(bins[bins['q']>5/7]['M_1'], bins[bins['q']>5/7]['M_2'], s=1,
            c='black', alpha=1, lw=0, rasterized=True,
            label='$q>5/7$ (analytic solution)')
    ax.scatter(bins[bins['q']<5/7]['M_1'], bins[bins['q']<5/7]['M_2'], s=1,
            c='blue', alpha=1, lw=0, rasterized=True,
            label='$q<5/7$ (sample from $M_{\mathrm{single}}$)')
    ax.set(xlabel='M_1', ylabel='M_2')
    ax.legend(loc='best', fontsize='x-small')
    ax.set_title('galaxia binaries', fontsize='small')
    f.tight_layout()
    f.savefig(savedir+'galaxia_gammaR_scatter_masses.pdf',
            dpi=250, bbox_inches='tight')

    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    ax.scatter(bins['L_1'] + bins['L_2'], bins['L_d'], s=2, c='black',
            alpha=1, lw=0, rasterized=True)
    ax.set(xlabel='L_1 + L_2', ylabel='L_d (should be sum)')
    ax.set_title('galaxia binaries', fontsize='small')
    f.tight_layout()
    f.savefig(savedir+'galaxia_L1L2_vs_Ld_luminosities.pdf',
            dpi=250, bbox_inches='tight')

    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))
    ax.scatter(bins[bins['q']>5/7]['M_2']/bins[bins['q']>5/7]['M_1'],
            bins[bins['q']>5/7]['q'], s=2, c='black', alpha=1, lw=0,
            rasterized=True)
    ax.scatter(bins['M_2'][bins['q']<5/7]/bins[bins['q']<5/7]['M_1'],
            bins[bins['q']<5/7]['q'], s=2, c='blue', alpha=1, lw=0,
            rasterized=True)
    ax.set(xlabel='M_2/M_1', ylabel='q (should be ratio)',
            ylim=[0,1], xlim=[0,1])
    ax.set_title('galaxia binaries', fontsize='small')
    f.tight_layout()
    f.savefig(savedir+'galaxia_M1M2_vs_q.pdf', dpi=250, bbox_inches='tight')


    # Save output
    savename = 'selected_galaxia_systems.p' if not is_kepler_analog \
            else 'selected_galaxia_systems_kepler_analog.p'

    galaxia_w_bins = {'singles': singles, 'binaries': bins}
    pickle.dump(galaxia_w_bins,
            open('/home/luke/local/'+savename, 'wb'))



def add_planets_and_do_survey(true_occ_rates, is_kepler_analog=False):
    '''
    args:
        true_occ_rates: [Γ_t_s, Γ_t_d1, Γ_t_d2]
    '''

    from math import pi as π

    loadname = 'selected_galaxia_systems.p' if not is_kepler_analog \
            else 'selected_GIC_systems_kepler_analog.p'
    gal = pickle.load(
            open('/home/luke/local/'+loadname, 'rb'))
    singles = gal['singles']
    doubles = gal['binaries']

    singles = singles.rename(index=str,columns={'mact': 'Mstar'})

    ###################################################
    # YOU HAVE SINGLES AND DOUBLES. GIVE THEM PLANETS #
    ###################################################
    # Fraction of stars in single star systems (all "sun-like", AKA with
    # 0.7-1.3Msun), with planet of (R_p, P).
    Γ_t_s = true_occ_rates[0]
    # Fraction per sun-like primary of double star systems with planet of
    # (R_p, P).
    Γ_t_d1 = true_occ_rates[1]
    # Fraction per sun-like secondary of double star systems with planet of
    # (R_p, P).
    Γ_t_d2 = true_occ_rates[2]

    # Pick which stars get a planet.
    s_pmask = np.array(np.random.rand((len(singles))) < Γ_t_s)
    d1_pmask = np.array(np.random.rand((len(doubles))) < Γ_t_d1)
    d2_pmask = np.array(np.random.rand((len(doubles))) < Γ_t_d2)

    singles['s_has_planet'] = (s_pmask &
                                (np.array(singles['Mstar'])>0.7) & \
                                (np.array(singles['Mstar'])<1.3))
    doubles['d1_has_planet'] = (d1_pmask &
                                (np.array(doubles['M_1'])>0.7) & \
                                (np.array(doubles['M_1'])<1.3))
    doubles['d2_has_planet'] = (d2_pmask & \
                                (np.array(doubles['M_2'])>0.7) & \
                                (np.array(doubles['M_2'])<1.3))

    R_p = 1*u.Rearth
    P = 1*u.year

    # Get semi-major axes for singles.
    M_s = np.array(singles[s_pmask]['Mstar'])*u.Msun
    a_s = (P**2 * c.G * M_s / (4*π*π))**(1/3)

    M_d1 = np.array(doubles[d1_pmask]['M_1'])*u.Msun
    a_d1 = (P**2 * c.G * M_d1 / (4*π*π))**(1/3)
    M_d2 = np.array(doubles[d2_pmask]['M_2'])*u.Msun
    a_d2 = (P**2 * c.G * M_d2 / (4*π*π))**(1/3)

    R_s = np.array(singles[s_pmask]['Rstar'])*u.Rsun
    R_d1 = np.array(doubles[d1_pmask]['R_1'])*u.Rsun
    R_d2 = np.array(doubles[d2_pmask]['R_2'])*u.Rsun

    # Draw impact parameters.
    b_s =  (a_s/R_s).cgs.value * \
            np.random.uniform(low=0., high=1., size=len(singles[s_pmask]))
    b_d1 = (a_d1/R_d1).cgs.value * \
            np.random.uniform(low=0., high=1., size=len(doubles[d1_pmask]))
    b_d2 = (a_d2/R_d2).cgs.value * \
            np.random.uniform(low=0., high=1., size=len(doubles[d2_pmask]))

    # Transiting masks (to be applied to arrays with planet mask already on).
    s_tmask = (np.abs(b_s) < 1)
    d1_tmask = (np.abs(b_d1) < 1)
    d2_tmask = (np.abs(b_d2) < 1)

    ρ_s   = 3*M_s/(4*π*R_s**3)
    ρ_d1  = 3*M_d1/(4*π*R_d1**3)
    ρ_d2  = 3*M_d2/(4*π*R_d2**3)

    Tdur_s  = T_dur(P, ρ_s[s_tmask], b_s[s_tmask] )
    Tdur_d1 = T_dur(P, ρ_d1[d1_tmask], b_d1[d1_tmask] )
    Tdur_d2 = T_dur(P, ρ_d2[d2_tmask], b_d2[d2_tmask] )

    ##################
    # RUN THE SURVEY #
    ##################
    # Define instrument.
    A = 0.708*u.m**2    # Kepler area
    # Survey parameters.
    T_obs = 20*u.year
    x_min = 7.1 # minimum SNR for detection.

    # Zombeck 2007, p.103: Vega, V=0.03 has a wavelength-specific photon flux
    # of 1e3 ph/s/cm^2/angstrom. So for a 500nm bandpass, we expect 55555e5
    # ph/s/cm^2. For the magnitude limit, we would want this in erg/s/cm^2.
    # However, this is good enough for a zero-point. Imagine (for simplicity)
    # we have an instrument with bandpass s.t. an apparent r-band magnitude of
    # 0 means 1e6 ph/s/cm^2.

    # "0" b/c zero-point, "N" b/c number flux
    m_0 = 0
    F_0_N = 1e6 * (u.s)**(-1) * (u.cm)**(-2)

    m_obs_s = singles['apparent_r']
    m_obs_d = doubles['sys_apparent_r']

    # Compute observed photon number fluxes for every system (regardless of
    # whether it has a planet).
    F_obs_s = F_0_N * 10**(-2/5 * (m_obs_s - m_0))
    F_obs_d = F_0_N * 10**(-2/5 * (m_obs_d - m_0))

    # Compute a S/N for all the transiting planets.

    # NOTE: for multiple masking, the following fails because δ_s[mask]
    # involves fancy indexing, which creates a copy of the data, but then the
    # second masking modifies the copy, rather than the original.
    # δ_s[s_pmask][s_tmask] = ((R_p/R_s[s_tmask])**2).cgs.value
    # Instead, do this:

    δ_s = np.ones_like(m_obs_s)*np.nan
    δ_s[[_[s_tmask] for _ in np.where(s_pmask)]] = \
            ((R_p/R_s[s_tmask])**2).cgs.value

    δ_d1 = np.ones_like(m_obs_d)*np.nan
    δ_d1[[_[d1_tmask] for _ in np.where(d1_pmask)]] = \
            ((R_p/R_d1[d1_tmask])**2).cgs.value

    δ_d2 = np.ones_like(m_obs_d)*np.nan
    δ_d2[[_[d2_tmask] for _ in np.where(d2_pmask)]] = \
            ((R_p/R_d2[d2_tmask])**2).cgs.value

    # Verify the crazy indexing works as follows:
    assert len(δ_s[~np.isnan(δ_s)]) == len(s_tmask[s_tmask])

    # Compute SNR distribution of transit events, and number of detections.
    N_tra = T_obs / P

    T_dur_s = np.ones_like(m_obs_s)*np.nan
    T_dur_s[[_[s_tmask] for _ in np.where(s_pmask)]] = Tdur_s
    T_dur_s = T_dur_s*u.hr

    T_dur_d1 = np.ones_like(m_obs_d)*np.nan
    T_dur_d1[[_[d1_tmask] for _ in np.where(d1_pmask)]] = Tdur_d1
    T_dur_d1 = T_dur_d1*u.hr

    T_dur_d2 = np.ones_like(m_obs_d)*np.nan
    T_dur_d2[[_[d2_tmask] for _ in np.where(d2_pmask)]] = Tdur_d2
    T_dur_d2 = T_dur_d2*u.hr

    x_s = δ_s * np.sqrt( F_obs_s * A * N_tra * T_dur_s )
    x_s = x_s.cgs

    # Eq 17 of toy_analytic_survey_170804.pdf, binary and target is primary
    dil_d1 = 1/(1 + np.array(doubles['γ_R']))
    x_d1 = dil_d1 * δ_d1 * np.sqrt( F_obs_d * A * N_tra * T_dur_d1 )
    x_d1 = x_d1.cgs

    # Eq 17 of toy_analytic_survey_170804.pdf, planet orbits secondary
    dil_d2 = 1/(1 + 1/np.array(doubles['γ_R']))
    x_d2 = dil_d2 * δ_d2 * np.sqrt( F_obs_d * A * N_tra * T_dur_d2 )
    x_d2 = x_d2.cgs

    # Save output
    singles['T_dur_s'] = T_dur_s
    singles['snr'] = x_s
    singles['is_detected'] = x_s > x_min

    doubles['T_dur_d1'] = T_dur_d1
    doubles['snr_d1'] = x_d1
    doubles['d1_is_detected'] = x_d1 > x_min

    doubles['T_dur_d2'] = T_dur_d2
    doubles['snr_d2'] = x_d2
    doubles['d2_is_detected'] = x_d2 > x_min

    N_transiting_planets = len(x_s[~np.isnan(x_s)]) + \
                           len(x_d1[~np.isnan(x_d1)]) + \
                           len(x_d2[~np.isnan(x_d2)])

    N_det = len(singles[singles['is_detected']]) + \
            len(doubles[doubles['d1_is_detected']]) + \
            len(doubles[doubles['d2_is_detected']])

    print('detected {:d} planets (of {:d} that were transiting)'.
            format(int(N_det), int(N_transiting_planets)))

    # Save output
    savename = 'surveyed_galaxia_systems_rlt14.p' if not is_kepler_analog \
            else 'surveyed_galaxia_systems_kepler_analog.p'

    galaxia_w_bins = {'singles': singles, 'binaries': doubles}
    pickle.dump(galaxia_w_bins,
            open('/home/luke/local/'+savename, 'wb'))


def prioritize_target_stars():

    from math import pi as π

    loadname = 'selected_galaxia_systems_kepler_analog.p'
    gal = pickle.load(open('/home/luke/local/'+loadname, 'rb'))
    singles = gal['singles']
    doubles = gal['binaries']

    # We are in the position of Batalha et al, 2010. Tim Brown just gave us a
    # 10+ million object KIC with a good amount of info. We have telemetric
    # capacity for 150k postage stamps, and need to cook up a special sauce to
    # choose the best stars to discover Earth 2.0 (or at least its occurrence
    # rate).

    # Compute observed photon number fluxes for every system.
    m_0 = 0
    F_0_N = 1e6 * (u.s)**(-1) * (u.cm)**(-2)

    m_obs_s = singles['apparent_r']
    m_obs_d = doubles['sys_apparent_r']

    F_obs_s = F_0_N * 10**(-2/5 * (m_obs_s - m_0))
    F_obs_d = F_0_N * 10**(-2/5 * (m_obs_d - m_0))

    A = 0.708*u.m**2    # Kepler area
    T_obs = 3.5*u.year  # Batalha+ 2010 says this is the mission lifetime.

    x_min = 7.1         # minimum SNR for detection.

    # To assign incorrect single star parameters to binary systems (as was and
    # still is done in the KIC), keep the total system luminosity. Use it to
    # find a mass (by inverting the `get_L` function of Fig 2 in 170804's
    # memo). Then apply the coarse mass-radius relation.

    import binary_distribution as bd
    from scipy.interpolate import interp1d
    _mass_arr = np.arange(0.001, 10+0.001, 0.001)
    _L_arr = bd.get_L(_mass_arr)
    func = interp1d(_L_arr, _mass_arr)

    doubles['M_as_if_single'] = func(np.array(doubles['L_d']))

    doubles['R_as_if_single'] = 1.06 * \
            np.array(doubles['M_as_if_single'])**0.945

    # Following Batalha+ 2010, the HZ radius is calculated as Kasting+ 1993's
    # value, scaled by the (square root of) the system luminosity.
    a_HZ_s = (0.95*u.AU)*np.sqrt(singles['lum'])
    a_HZ_d = (0.95*u.AU)*np.sqrt(doubles['L_d'])

    semimajs = [
        ('Rpmin_HZ', '_HZ', a_HZ_s, a_HZ_d),
        ('Rpmin_half_HZ', '_half_HZ', 0.5*a_HZ_s, 0.5*a_HZ_d),
        ('Rpmin_close', '_close', 5*np.array(singles['Rstar'])*u.Rsun,
             5*np.array(doubles['R_as_if_single'])*u.Rsun)
        ]

    # Compute the minimum detectable planet radius at three semimajor axes.
    for semimaj in semimajs:

        a_s = semimaj[2]
        a_d = semimaj[3]

        # Batalha+ 2010 Eq 4
        R_s = np.array(singles['Rstar'])*u.Rsun
        M_s = np.array(singles['mact'])*u.Msun
        T_dur_s = 2 * R_s * np.sqrt(a_s / (c.G * M_s) )

        # Compute number of transits
        P_s = ((a_s**3 * 4*π*π / (c.G * M_s))**(1/2)).cgs
        N_tra_s = (T_obs.cgs / P_s).cgs.value

        R_d = np.array(doubles['R_as_if_single'])*u.Rsun
        M_d = np.array(doubles['M_as_if_single'])*u.Msun
        T_dur_d = 2 * R_d * np.sqrt(a_d / (c.G * M_d) )

        P_d = ((a_d**3 * 4*π*π / (c.G * M_d))**(1/2)).cgs
        N_tra_d = (T_obs.cgs / P_d).cgs.value

        # "Noise" or "N" in my notation. sigma_tot in Batalha+ 2010 notation.
        σ_tot_s = 1/(np.sqrt( F_obs_s * A * N_tra_s * T_dur_s )).cgs.value
        σ_tot_d = 1/(np.sqrt( F_obs_d * A * N_tra_d * T_dur_d )).cgs.value

        # Dilution parameter from Batalha+ 2010. We ignore it.
        r = 1

        Rp_min_s = (R_s * np.sqrt( x_min * σ_tot_s / r )).to(u.Rearth)
        Rp_min_d = (R_d * np.sqrt( x_min * σ_tot_d / r )).to(u.Rearth)

        singles[semimaj[0]] = Rp_min_s
        singles['Ntra'+semimaj[1]] = N_tra_s
        doubles[semimaj[0]] = Rp_min_d
        doubles['Ntra'+semimaj[1]] = N_tra_d

    # Apply Batalha+ 2010 Table 1 prioritization
    sp0 = ((singles['Ntra_HZ'] >= 3) & \
          (singles['Rpmin_HZ'] < 1) & \
          (singles['apparent_r'] < 13))
    sp1 = ((singles['Ntra_HZ'] >= 3) & \
          (singles['Rpmin_HZ'] < 2) & \
          (singles['apparent_r'] < 13))
    sp2 = ((singles['Ntra_HZ'] >= 3) & \
          (singles['Rpmin_HZ'] < 1) & \
          (singles['apparent_r'] < 14))
    sp3 = ((singles['Ntra_HZ'] >= 3) & \
          (singles['Rpmin_HZ'] < 2) & \
          (singles['apparent_r'] < 14))
    sp4 = ((singles['Ntra_HZ'] >= 3) & \
          (singles['Rpmin_HZ'] < 1) & \
          (singles['apparent_r'] < 15))
    sp5 = ((singles['Ntra_HZ'] >= 3) & \
          (singles['Rpmin_HZ'] < 1) & \
          (singles['apparent_r'] < 16))
    sp6 = ((singles['Ntra_half_HZ'] >= 3) & \
          (singles['Rpmin_half_HZ'] < 1) & \
          (singles['apparent_r'] < 14))
    sp7 = ((singles['Ntra_half_HZ'] >= 3) & \
          (singles['Rpmin_half_HZ'] < 2) & \
          (singles['apparent_r'] < 14))
    sp8 = ((singles['Ntra_close'] >= 3) & \
          (singles['Rpmin_close'] < 2) & \
          (singles['apparent_r'] < 14))
    sp9 = ((singles['Ntra_HZ'] >= 3) & \
          (singles['Rpmin_HZ'] < 2) & \
          (singles['apparent_r'] < 15))
    sp10 = ((singles['Ntra_HZ'] >= 3) & \
          (singles['Rpmin_HZ'] < 2) & \
          (singles['apparent_r'] < 16))
    sp11 = ((singles['Ntra_close'] >= 3) & \
          (singles['Rpmin_close'] < 2) & \
          (singles['apparent_r'] < 15))
    sp12 = ((singles['Ntra_close'] >= 3) & \
          (singles['Rpmin_close'] < 2) & \
          (singles['apparent_r'] < 16))

    dp0 = ((doubles['Ntra_HZ'] >= 3) & \
          (doubles['Rpmin_HZ'] < 1) & \
          (doubles['sys_apparent_r'] < 13))
    dp1 = ((doubles['Ntra_HZ'] >= 3) & \
          (doubles['Rpmin_HZ'] < 2) & \
          (doubles['sys_apparent_r'] < 13))
    dp2 = ((doubles['Ntra_HZ'] >= 3) & \
          (doubles['Rpmin_HZ'] < 1) & \
          (doubles['sys_apparent_r'] < 14))
    dp3 = ((doubles['Ntra_HZ'] >= 3) & \
          (doubles['Rpmin_HZ'] < 2) & \
          (doubles['sys_apparent_r'] < 14))
    dp4 = ((doubles['Ntra_HZ'] >= 3) & \
          (doubles['Rpmin_HZ'] < 1) & \
          (doubles['sys_apparent_r'] < 15))
    dp5 = ((doubles['Ntra_HZ'] >= 3) & \
          (doubles['Rpmin_HZ'] < 1) & \
          (doubles['sys_apparent_r'] < 16))
    dp6 = ((doubles['Ntra_half_HZ'] >= 3) & \
          (doubles['Rpmin_half_HZ'] < 1) & \
          (doubles['sys_apparent_r'] < 14))
    dp7 = ((doubles['Ntra_half_HZ'] >= 3) & \
          (doubles['Rpmin_half_HZ'] < 2) & \
          (doubles['sys_apparent_r'] < 14))
    dp8 = ((doubles['Ntra_close'] >= 3) & \
          (doubles['Rpmin_close'] < 2) & \
          (doubles['sys_apparent_r'] < 14))
    dp9 = ((doubles['Ntra_HZ'] >= 3) & \
          (doubles['Rpmin_HZ'] < 2) & \
          (doubles['sys_apparent_r'] < 15))
    dp10 = ((doubles['Ntra_HZ'] >= 3) & \
          (doubles['Rpmin_HZ'] < 2) & \
          (doubles['sys_apparent_r'] < 16))
    dp11 = ((doubles['Ntra_close'] >= 3) & \
          (doubles['Rpmin_close'] < 2) & \
          (doubles['sys_apparent_r'] < 15))
    dp12 = ((doubles['Ntra_close'] >= 3) & \
          (doubles['Rpmin_close'] < 2) & \
          (doubles['sys_apparent_r'] < 16))

    sel_s = singles[(sp0 | sp1 | sp2 | sp3 | sp4 | sp5 | sp6 | sp7 | sp8 | \
                     sp9 | sp10 )]

    sel_d = doubles[(dp0 | dp1 | dp2 | dp3 | dp4 | dp5 | dp6 | dp7 | dp8 | \
                     dp9 | dp10 )]

    # NOTE: ignoring priority class 11 and 12 because this already provides a
    # total of 200316 stars.

    savename = 'selected_GIC_systems_kepler_analog.p'
    out = {'singles':sel_s, 'binaries':sel_d}
    pickle.dump(out, open('/home/luke/local/'+savename, 'wb'))


def evaluate_occurrence_rate_errors(true_occ_rates):

    gal = pickle.load(open(
        '/home/luke/local/surveyed_galaxia_systems_kepler_analog.p', 'rb'))

    singles = gal['singles']
    doubles = gal['binaries']

    ##########################################################################
    # QUESTION # 1:                                                          #
    # If we ignore binarity, how wrong is our occurrence rate for planets of #
    # radius Rp?                                                             #
    ##########################################################################

    N_s = len(singles)
    N_d = len(doubles)
    N_det_s = len(singles[singles['is_detected']])

    # Compute transit probabilities (needed for completeness calculation).
    # First for single stars.
    from math import pi as π
    P = 1*u.year
    R_s = np.array(singles['Rstar'])*u.Rsun
    M_s = np.array(singles['Mstar'])*u.Msun
    a_s = (P**2 * c.G * M_s / (4*π*π))**(1/3)

    f_s_g = (R_s/a_s).cgs.value

    # Incorrect transit probability for double stars.
    M_as_if_single = np.array(doubles['M_as_if_single'])*u.Msun
    a_as_if_single = (P**2 * c.G * M_as_if_single / (4*π*π) )**(1/3)
    R_as_if_single = np.array(doubles['R_as_if_single'])*u.Rsun

    f_d_as_if_s_g = (R_as_if_single / a_as_if_single).cgs.value

    # Correct transit probabilities for double stars.
    R_d1 = np.array(doubles['R_1'])*u.Rsun
    R_d2 = np.array(doubles['R_2'])*u.Rsun
    M_d1 = np.array(doubles['M_1'])*u.Msun
    M_d2 = np.array(doubles['M_2'])*u.Msun
    a_d1 = (P**2 * c.G * M_d1 / (4*π*π))**(1/3)
    a_d2 = (P**2 * c.G * M_d2 / (4*π*π))**(1/3)

    f_d1_g = (R_d1/a_d1).cgs.value
    f_d2_g = (R_d2/a_d2).cgs.value

    ###########################################################################
    #       (subquestion)                                                     #
    # "Completeness" calculation: if this star did have (Rp, P) planet, would #
    # it be detected? (binary completeness). There's code duplication here..  #
    ###########################################################################
    R_p = 1*u.Rearth

    # Compute observed photon number fluxes for every system (regardless of
    # whether it has a planet).
    m_0 = 0
    F_0_N = 1e6 * (u.s)**(-1) * (u.cm)**(-2)
    m_obs_s = singles['apparent_r']
    m_obs_d = doubles['sys_apparent_r']
    F_obs_s = F_0_N * 10**(-2/5 * (m_obs_s - m_0))
    F_obs_d = F_0_N * 10**(-2/5 * (m_obs_d - m_0))

    # Compute a S/N for all systems under the assumption that they have a
    # transiting planet. (For doubles, compute the true S/N, as well as the one
    # making the mistake of thinking that they're singles).

    δ_s = ((R_p/R_s)**2).cgs.value
    δ_d_as_if_s = ((R_p/R_as_if_single)**2).cgs.value
    δ_d1 = ((R_p/R_d1)**2).cgs.value
    δ_d2 = ((R_p/R_d2)**2).cgs.value

    # Get transit durations, averaging over the impact parameter.
    T_obs = 20*u.year
    N_tra = T_obs / P

    def _transit_dur_no_b(P, ρ):
        from math import pi as π
        ρ_sun = 3*u.Msun/(4*π*u.Rsun**3)
        return 13*u.hr * (P / (1*u.yr))**(1/3) * (ρ/ρ_sun)**(-1/3) * π/4

    ρ_s   = 3*M_s/(4*π*R_s**3)
    ρ_as_if_single = 3*M_as_if_single/(4*π*R_as_if_single**3)
    ρ_d1  = 3*M_d1/(4*π*R_d1**3)
    ρ_d2  = 3*M_d2/(4*π*R_d2**3)

    T_dur_s = _transit_dur_no_b(P, ρ_s)
    T_dur_d_as_if_s = _transit_dur_no_b(P, ρ_as_if_single)
    T_dur_d1 = _transit_dur_no_b(P, ρ_d1)
    T_dur_d2 = _transit_dur_no_b(P, ρ_d2)

    # Compute SNR distribution of transit events, and number of detections.
    A = 0.708*u.m**2    # Kepler area

    x_s = δ_s * np.sqrt( F_obs_s * A * N_tra * T_dur_s )
    x_s = x_s.cgs

    x_d_as_if_s = δ_d_as_if_s * np.sqrt(F_obs_d * A * N_tra * T_dur_d_as_if_s)
    x_d_as_if_s = x_d_as_if_s.cgs

    # Eq 17 of toy_analytic_survey_170804.pdf, binary and target is primary
    dil_d1 = 1/(1 + np.array(doubles['γ_R']))
    x_d1 = dil_d1 * δ_d1 * np.sqrt( F_obs_d * A * N_tra * T_dur_d1 )
    x_d1 = x_d1.cgs

    # Eq 17 of toy_analytic_survey_170804.pdf, planet orbits secondary
    dil_d2 = 1/(1 + 1/np.array(doubles['γ_R']))
    x_d2 = dil_d2 * δ_d2 * np.sqrt( F_obs_d * A * N_tra * T_dur_d2 )
    x_d2 = x_d2.cgs

    # Finally: completeness fractions for each star. f_c_s is an array (over
    # star systems) of zeros (where the [Rp, P] planet could not be detected)
    # and ones (where it could).
    x_min = 7.1

    f_s_c = np.array(list(map(int, x_s > x_min)))
    f_d_as_if_s_c = np.array(list(map(int, x_d_as_if_s > x_min)))
    f_d1_c = np.array(list(map(int, x_d1 > x_min)))
    f_d2_c = np.array(list(map(int, x_d2 > x_min)))

    # Total detection efficiency: product of geometric transit probability and
    # completeness fraction.
    Q_s = f_s_g * f_s_c
    Q_d_as_if_s = f_d_as_if_s_g * f_d_as_if_s_c
    Q_d1 = f_d1_g * f_d1_c
    Q_d2 = f_d2_g * f_d2_c

    # Astronomer A occurrence rate:
    Q_astronomer_A = np.concatenate((Q_s, Q_d_as_if_s))
    Z = (N_s + N_d) * np.sum(Q_astronomer_A) / len(Q_astronomer_A)

    Γ_A_Rp = N_det_s / Z

    #print('Γ_A_Rp: {:.3f}'.format(Γ_A_Rp))

    ##########################################################################
    # QUESTION # 2:                                                          #
    # What if we ignore binarity, but count derived planet radii that are    #
    # "close enough"?                                                        #
    ##########################################################################

    x = 0.1

    doubles['d1_R_p_obs'] = np.sqrt(dil_d1) * R_p
    doubles['d2_R_p_obs'] = np.sqrt(dil_d2) * R_p

    # Find minimum detectable planet radii for each double (that is thought to
    # be a single). Use it to construct the completeness fraction, i.e. the
    # probability of a planet in the {(1-x)Rp < Rp_obs < Rp} interval of being
    # detected, under the prior assumption of Rp ~ uniform((1-x)Rp, Rp), so
    # that between the two edge cases the completeness is just the fraction of
    # the radius interval for which the detection was possible.
    noise = 1 / np.sqrt(F_obs_d * A * N_tra * T_dur_d_as_if_s)
    r = 1
    min_detectable_Rp = (np.sqrt(x_min * noise) * R_as_if_single).to(u.Rearth)

    f_d_as_if_s_Rpprime_c = np.ones_like(f_d_as_if_s_g)*np.nan

    R_p_min = R_p * (1-x)
    R_p_max = R_p

    f_d_as_if_s_Rpprime_c[ min_detectable_Rp > R_p_max ] = 1
    f_d_as_if_s_Rpprime_c[ min_detectable_Rp < R_p_min ] = 0

    mask = (min_detectable_Rp > R_p_min) & (min_detectable_Rp < R_p_max)
    f_d_as_if_s_Rpprime_c[mask] = (R_p_max - min_detectable_Rp[mask])\
                                 /(R_p_max-R_p_min)

    assert len(f_d_as_if_s_Rpprime_c[np.isnan(f_d_as_if_s_Rpprime_c)]) == 0

    # Geometric transit probability stays the same. Completeness changes
    # slightly.
    Q_s_in_Rpprime = f_s_g * f_s_c
    Q_d_as_if_s_Rpprime = f_d_as_if_s_g * f_d_as_if_s_Rpprime_c

    # Astronomer A' occurrence rate:
    Q_astronomer_Aprime = np.concatenate((Q_s_in_Rpprime, Q_d_as_if_s_Rpprime))
    Z = (N_s + N_d) * np.sum(Q_astronomer_Aprime) / len(Q_astronomer_Aprime)

    d1_R_p_obs = np.array(doubles['d1_R_p_obs'])*u.Rearth
    d2_R_p_obs = np.array(doubles['d2_R_p_obs'])*u.Rearth
    d1_is_detected = np.array(doubles['d1_is_detected'])
    d2_is_detected = np.array(doubles['d2_is_detected'])

    N_det_d_Rpprime = len(doubles[(d1_R_p_obs>R_p_min) & (d1_R_p_obs<R_p_max) \
                                   & d1_is_detected ]) + \
                      len(doubles[(d2_R_p_obs>R_p_min) & (d2_R_p_obs<R_p_max) \
                                   & d2_is_detected ])

    Γ_A_Rpprime = (N_det_s + N_det_d_Rpprime) / Z

    #print('Γ_A_Rpprime: {:.3f}'.format(Γ_A_Rpprime))

    Γ_t_s = true_occ_rates[0]
    Γ_t_d1= true_occ_rates[1]
    Γ_t_d2= true_occ_rates[2]

    Γ_D_Rp = (Γ_t_s*N_s + (Γ_t_d1 + Γ_t_d2)*N_d) / (N_s + 2*N_d)

    #print('Γ_D_Rp: {:.3f}'.format(Γ_D_Rp))

    chi = (Γ_t_d1 + Γ_t_d2)/Γ_t_s
    #print('chi: {:.3f}'.format(chi))

    X_Γ = Γ_D_Rp / Γ_A_Rpprime

    return X_Γ


def do_kic_complete_survey():
    '''
    Follow Sharma et al 2016 ApJ 822:15.

    Take all KIC and Galaxia stars within 7.5 degrees of the center of Kepler's
    field, and only take stars with r<14.

    The KIC should be complete for these relatively bright stars, so they
    provide a good point for comparison with Galaxia.

    To execute, "convert_galaxia_output" must be run first in 2.7 because of
    the `ebf` dependency. Then the rest is fine in 3.X
    '''

    Γ_t_s = 0.75
    Γ_t_d1 = 0.75
    Γ_t_d2 = 0.
    true_occ_rates = [Γ_t_s, Γ_t_d1, Γ_t_d2]

    if not os.path.exists('/home/luke/local/selected_kic_stars.csv'):
        print('selecting KIC stars')
        select_kic_stars()

    if not os.path.exists('/home/luke/local/selected_galaxia_stars.csv'):
        print('selecting galaxia stars')
        select_galaxia_stars()

    print('making distribution plots')
    plot_distributions()

    print('adding binaries to galaxia systems')
    add_binaries_to_galaxia()

    add_planets_and_do_survey(true_occ_rates)




def do_kepler_analog():
    '''
    Ignore the KIC comparison.

    Take all Galaxia stars within 7.5 degrees of the center of Kepler's
    field, and only take stars with r<17.

    This is beyond where the KIC is complete, so comparisons between the two
    will be confused.

    Then apply analogs of the Batalha+ 2010 Table 1 prioritization.
    '''

    N_trials = 5
    Γ_t_s = 0.75
    Γ_t_d1 = 0.75

    if not os.path.exists('/home/luke/local/selected_galaxia_stars_kepler_analog.csv'):
        print('selecting galaxia stars (kepler analog)')
        select_galaxia_stars(is_kepler_analog=True)

    if not os.path.exists('/home/luke/local/selected_galaxia_systems_kepler_analog.p'):
        print('adding binaries to galaxia systems (kepler analog)')
        add_binaries_to_galaxia(is_kepler_analog=True)

    if not os.path.exists('/home/luke/local/selected_GIC_systems_kepler_analog.p'):
        print('prioritizing selected stars (constructing "KIC")')
        prioritize_target_stars()

    Γ_t_d2_arr = np.arange(0, 0.80, 0.05)
    avg_X_Γs, std_X_Γs = [], []

    for Γ_t_d2 in Γ_t_d2_arr:

        X_Γs = []

        for i in range(N_trials):

            print(i, Γ_t_d2)

            np.random.seed(i)

            true_occ_rates = [Γ_t_s, Γ_t_d1, Γ_t_d2]

            add_planets_and_do_survey(true_occ_rates, is_kepler_analog=True)

            X_Γ = evaluate_occurrence_rate_errors(true_occ_rates)

            X_Γs.append(X_Γ)

        avg_X_Γs.append(np.mean(X_Γs))
        std_X_Γs.append(np.std(X_Γs))

    out = np.array(avg_X_Γs)
    out_err = np.array(std_X_Γs)

    out = pd.DataFrame({'X_Γ': out,
                        'X_Γ_err': out_err,
                        'Γ_t_d2': Γ_t_d2_arr})

    out.to_csv('../results/error_v_occrate_data.csv', index=False)


def plot_error_v_occrate(Γ_t_d2_best_guess):

    import matplotlib.pyplot as plt
    import numpy as np

    Γ_t = 0.75

    df = pd.read_csv('../results/error_v_occrate_data.csv')
    X_Γs = df['X_Γ']
    yerr = df['X_Γ_err']
    Γ_t_d2 = df['Γ_t_d2']

    f, ax = plt.subplots(figsize=(4,4))

    ax.errorbar(Γ_t_d2/Γ_t, X_Γs, yerr=yerr, color='black',
            label='occ rate correction factor')

    ymin, ymax = min(ax.get_ylim()), max(ax.get_ylim())
    ax.vlines(Γ_t_d2_best_guess/Γ_t, ymin, ymax, colors='k',
            linestyles='solid', alpha=0.4,
            label='most reasonable $\Gamma_{t,d2}$ value')

    ax.set(
        xlabel='$\Gamma_{t,d2} / \Gamma_t$, for '+\
               '$\Gamma_t = \Gamma_{t,s} = \Gamma_{t,d1} $',
        ylabel='$X_\Gamma = \Gamma_{\mathrm{D},R_p}/\Gamma_{\mathrm{A},R_p\'}$',
        ylim=(ymin, ymax)
        )
    ax.legend(
        loc='upper right',
        fontsize='x-small'
        )

    # Get estimate of the intercept value
    from scipy.interpolate import interp1d
    func = interp1d(Γ_t_d2/Γ_t, X_Γs)
    x_new = np.arange(0,1,1e-4)
    y_new = func(x_new)

    def find_nearest_idx(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx

    idx = find_nearest_idx(x_new, Γ_t_d2_best_guess/Γ_t)
    best_err_estimate = y_new[idx]
    print('at estimated Γ_t_d2/Γ_t of {:.3f}, err is {:.3f}'.format(
        x_new[idx], best_err_estimate))
    print('this should be close to best guess Γ_t_d2/Γ_t, which is {:.3f}'.
            format(Γ_t_d2_best_guess/0.75))

    f.tight_layout()
    f.savefig('error_vs_occrate.pdf', dpi=250)


def find_best_guess_secondary_occ_rate():
    '''
    While `do_kepler_analog` makes no assumption about Γ_t_d2, the best
    assumption we can make is that the occ rate for the secondary is the same
    as for the primary (and single stars) if it's a sun-like.

    Otherwise, it's zero.
    '''

    gal = pickle.load(open(
        '/home/luke/local/surveyed_galaxia_systems_kepler_analog.p', 'rb'))

    singles = gal['singles']
    doubles = gal['binaries']

    N_sunlike_secondaries = len(
            doubles[(doubles['M_2'] > 0.7) & (doubles['M_2'] < 1.3)])

    frac_of_secondaries_sunlike = N_sunlike_secondaries / len(doubles)

    Γ_t_s = 0.75
    Γ_t_d1 = 0.75

    Γ_t = Γ_t_s

    Γ_t_d2_best_guess = Γ_t * frac_of_secondaries_sunlike
    print(frac_of_secondaries_sunlike)

    return Γ_t_d2_best_guess





if __name__ == '__main__':

    np.random.seed(42)

    # Thing #1: verify the KIC vs GIC comparison with r<14 complete comparison.

    #do_kic_complete_survey()

    # Thing #2: see how the occ rate error scales vs Γ_t_d2.
    do_kepler_analog()

    Γ_t_d2_best_guess = find_best_guess_secondary_occ_rate()

    plot_error_v_occrate(Γ_t_d2_best_guess)



