'''
fixed primary L1. varying light ratio (L2/L1). varying Rp.

This code is the Monte Carlo simulation needed to produce summary statistics
for X_Γ.

(To the extent possible, it is the same as model_2.py)
'''
from __future__ import division, print_function

import numpy as np, pandas as pd
from astropy import units as u, constants as c
from math import pi as π
import os
from scipy.interpolate import interp1d

global α
α = 3.5 # coefficient in the L~M^α power law scaling.

#####################
# UTILITY FUNCTIONS #
#####################
def get_L(M):
    '''
    assume input mass, M, a float or vector, is in solar units.
    '''

    return M**α


def draw_planet_radii(df, singleordouble, δ=None, R_pl=None, R_pu=None):
    '''
    Assign planets radii from power law radius probability distribution
    function:
    $$
        ρ_Rpt (R_pt) ~ R_pt^δ
    $$

    df: DataFrame of singles or doubles

    singleordouble (str): "single" or "double"

    δ (required float): power of the distribution. For instance, Howard et al
    (2012) find δ=-2.92 +/- 0.11.

    R_pl, R_pu (optional floats): lower and upper bound for truncation. If
    None, assumed the distribution is not truncated
    '''

    assert δ

    power = δ
    ΔR = 0.01
    R_max = 50 # earth Radii, for grid.
    R_min = 0

    if isinstance(R_pu,float):
        assert R_pu < R_max
    if not isinstance(R_pu,float):
        R_pu = R_max

    R_pt_grid = np.arange(R_min, R_max+ΔR, ΔR)
    prob_Rpt = np.zeros_like(R_pt_grid)

    inds = (R_pt_grid > R_pl) & (R_pt_grid < R_pu)
    prob_Rpt[inds] = (R_pt_grid[inds])**power
    prob_Rpt /= trapz(prob_Rpt, R_pt_grid)

    # inverse transform sampling to get the true planet radii.
    cdf_Rpt = np.append(0, np.cumsum(prob_Rpt)/np.max(np.cumsum(prob_Rpt)))
    func = interp1d(cdf_Rpt, np.append(0,R_pt_grid))

    def _draw_radii(func, df):

        drawn_Rpts = func(np.random.uniform(size=len(df)))

        return drawn_Rpts

    #assign planet radii to star systems of desired type with planets. systems
    #without planets get a 0 in this column.
    if singleordouble == 'single':
        df['R_pt'] = _draw_radii(func, df) * df[singleordouble+'_has_planet']

    elif singleordouble == 'double':
        #separate draws for primaries and secondaries.
        df['primary_R_pt'] = _draw_radii(func, df) * df['primary_has_planet']
        df['secondary_R_pt'] = _draw_radii(func, df) * df['secondary_has_planet']

    else:
        raise Exception

    return df



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

    # Vectorized draw:
    # vol sphere / vol cube = π/6 ~= half. Draw 10* the number of systems you
    # really need. Then cut.
    _x = d_max * np.random.rand( (10*N_sys) )
    _y = d_max * np.random.rand( (10*N_sys) )
    _z = d_max * np.random.rand( (10*N_sys) )
    _r = np.sqrt(_x**2 + _y**2 + _z**2)

    # This assertion ensures you initially drew enough.
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
        print('draw of volume limited light ratios failed.')
        import IPython; IPython.embed()

    df['q'] = q
    df['γ_R'] = γ_R

    return df


if __name__ == '__main__':

    #############################
    # !!! BEGIN MONTE CARLO !!! #
    #############################
    for index, seed in enumerate(np.arange(int(0e3),int(1e3),1)):

        ################################################################
        # BEGIN OBNOXIOUS PARAMETERS NEEDED FOR MONTE CARLO SIMULATION #
        ################################################################

        # Zombeck 2007, p.103: Vega, V=0.03 has a wavelength-specific photon flux of
        # 1e3 ph/s/cm^2/angstrom. So for our instrument's bandpass, we
        # expect 5e5 ph/s/cm^2. For the magnitude limit, we want this in
        # erg/s/cm^2.

        # Zero points, and magnitude limit.
        λ_nominal = 750*u.nm
        energy_per_ph = c.h * c.c / λ_nominal
        m_0 = 0
        F_0 = 5e5 * energy_per_ph * (u.s)**(-1) * (u.cm)**(-2)
        m_lim = 10 + 2*np.random.rand()
        F_lim = F_0 * 10**(-2/5 * (m_lim - m_0)) # limiting flux, in erg/s/cm^2

        ##############################################################
        # END OBNOXIOUS PARAMETERS NEEDED FOR MONTE CARLO SIMULATION #
        ##############################################################

        ######################
        # STELLAR POPULATION #
        ######################
        # Binary fraction. BF = n_d / (n_s+n_d). Raghavan+ 2010 solar.
        BF = 0.44
        # Bovy (2017) gives n_tot = n_s + n_d. We want the number density of single
        # star systems, n_s, and the number density of double star systems, n_d.
        #n_tot = 4.76e-4 / (u.pc**3)
        n_tot = 5e-4 / (u.pc**3)
        n_d = BF * n_tot
        n_s = (1-BF)*n_d/BF

        ################
        # SINGLE STARS #
        ################
        M_1 = 1*u.Msun
        L_1 = get_L(M_1.value)*u.Lsun
        R_1 = 1*u.Rsun

        # Distance limit for selecting single star systems.
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
        # Corresponding maximum double star system luminosity
        L_d_max = (1+γ_R_max)*L_1
        # Maximum distance out to which the maximum luminosity double star system
        # could be detected. (...stupid notation)
        d_max_d_max = (L_d_max / (4*π*F_lim) )**(1/2)
        # Number of double star systems in a volume limited sample out to
        # d_max_d_max.
        N_d_max = int(np.floor( (n_d * 4*π/3 * d_max_d_max**3).cgs.value ))

        # NOTE: ACTUAL RANDOM PART OF THE SIMULATION BEGINS HERE
        vl = draw_star_positions(d_max_d_max.to(u.pc), N_d_max, 'binary')
        vl = draw_vol_limited_light_ratios(vl, M_1.value, L_1.value)

        vl['L_d'] = L_1 * (1 + vl['γ_R'])

        vl['F'] = ((np.array(vl['L_d'])*u.Lsun) / \
                    (4*π* (np.array(vl['r'])*u.pc)**2 )).cgs

        # impose magnitude cut.
        doubles = vl[vl['F'] > F_lim.cgs.value]

        if index==0:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.integrate import trapz

            f,ax=plt.subplots()
            sns.distplot(doubles['q'], ax=ax, kde=False, norm_hist=True)
            q = np.arange(0.1,1+1e-3,1e-3)
            prob_ml_q = (1+q**α)**(3/2)
            prob_ml_q /= trapz(prob_ml_q, q)
            ax.plot(q, prob_ml_q)
            f.savefig('tests/doubles_q_distribution.pdf')

            plt.close('all')
            f,ax=plt.subplots()
            sns.distplot(doubles['γ_R'], ax=ax, kde=False, norm_hist=True)
            γ_R = np.arange(0.1**3.5,1+1e-3,1e-3)
            prob_ml_γR = (1+γ_R)**(3/2) * γ_R**(-5/7)
            prob_ml_γR /= trapz(prob_ml_γR, γ_R)
            ax.plot(γ_R, prob_ml_γR)
            ax.set_yscale('log')

            f.savefig('tests/doubles_gammaR_distribution.pdf')

        N_d = int(len(doubles))

        ################################################
        # YOU HAVE SINGLES AND DOUBLES. ASSIGN PLANETS #
        ################################################
        # Fraction of stars in single star systems with planet of (R_p, P). NB this
        # is the same as the average number of planets per star in the single
        # planet system limit. (...)
        Γ_ts = 0.3
        # Fraction per primary of double star systems with planet of (R_p, P).
        Γ_td = 0.3
        # Weight of secondaries of desired type with planets of desired type
        w_d2 = 1

        # First, randomly select which stars of desired type get single planet systems. 
        single_has_planet = (np.random.rand( N_s ) < Γ_ts)
        primary_has_planet = (np.random.rand( N_d ) < Γ_td)

        # Note only some fraction of secondaries are of the desired type... (and
        # only "has planet" if planet is of desired type and star is of desired
        # type)
        M_secondary_min = 0.7
        M_secondary_max = 1
        f_d = len(doubles[(doubles['q']>M_secondary_min) &\
                          (doubles['q']<M_secondary_max)])\
              /N_d

        secondary_has_planet = (np.random.rand( N_d ) < Γ_td*w_d2) \
                & (doubles['q']>M_secondary_min) \
                & (doubles['q']<M_secondary_max)

        singles['single_has_planet'] = single_has_planet
        doubles['primary_has_planet'] = primary_has_planet
        doubles['secondary_has_planet'] = secondary_has_planet

        N_stars = N_s + (1+f_d)*N_d
        N_planets = np.sum(single_has_planet) + \
                    np.sum(primary_has_planet) + \
                    np.sum(secondary_has_planet)

        #####################
        # DRAW PLANET RADII #
        #####################
        # draw radii for stars that are in the sample, and that are of the
        # desired type.

        singles = draw_planet_radii(singles, 'single', δ=-2.92, R_pl=2, R_pu=20)
        doubles = draw_planet_radii(doubles, 'double', δ=-2.92, R_pl=2, R_pu=20)

        # compute apparent radii:
        singles['R_pa'] = singles['R_pt']

        dilution = (1 + np.array(doubles['γ_R']))**(-1)
        doubles['primary_R_pa'] = doubles['primary_R_pt'] * dilution**(1/2)

        dilution = (1 + np.array(doubles['γ_R'])**(-1))**(-1)
        doubles['secondary_R_pa'] = doubles['secondary_R_pt'] \
                * dilution**(1/2) \
                * np.array(doubles['γ_R'])**(-1/α) # term for R_star_a/R_star_t

        doubles['secondary_R_pa_onlydilution'] = doubles['secondary_R_pt'] \
                * dilution**(1/2) \

        del dilution

        #################################
        # COMPUTE TRANSIT PROBABILITIES #
        #################################
        P = 15*u.day
        a_1 = (P**2 * c.G * M_1 / (4*π*π))**(1/3)
        ones = np.ones_like(np.array(singles['r']))
        f_sg = (R_1/a_1).cgs.value * ones

        s_has_transiting_planet = (np.random.rand(N_s) < f_sg) & single_has_planet
        singles['single_has_transiting_planet'] = s_has_transiting_planet

        # Transiting planets in binary systems.
        R_d1 = R_1
        a_d1 = a_1
        M_d2 = np.array(doubles['q'])*M_1
        R_d2 = (M_d2.value)*u.Rsun
        a_d2 = (P**2 * c.G * M_d2 / (4*π*π))**(1/3)

        ones = np.ones_like(np.array(doubles['r']))

        f_d1g = (R_d1/a_d1).cgs.value * ones
        d1_has_transiting_planet = (np.random.rand(N_d) < f_d1g) & primary_has_planet
        doubles['primary_has_transiting_planet'] = d1_has_transiting_planet

        f_d2g = (R_d2/a_d2).cgs.value * ones
        d2_has_transiting_planet = (np.random.rand(N_d) < f_d2g) & secondary_has_planet
        doubles['secondary_has_transiting_planet'] = d2_has_transiting_planet

        #########################################
        # COMPUTE COMPLETENESSES AND DETECTIONS #
        #########################################

        f_sc = 1*np.ones_like(np.array(singles['r']))

        singles['single_has_detected_planet'] = \
                (np.random.rand(N_s) < f_sc) \
                & s_has_transiting_planet

        f_d1c = (1+np.array(doubles['γ_R']))**(-3)
        f_d2c = (1+np.array(doubles['γ_R'])**(-1))**(-3) \
                *np.array(doubles['γ_R'])**(-5/α)

        doubles['primary_has_detected_planet'] = \
                (np.random.rand(N_d) < f_d1c) \
                & d1_has_transiting_planet

        doubles['secondary_has_detected_planet'] = \
                (np.random.rand(N_d) < f_d2c) \
                & d2_has_transiting_planet

        ##########################
        # COMPUTE SURVEY RESULTS #
        ##########################

        N_det_s = np.sum(np.array(singles['single_has_detected_planet']))
        N_det_d1 = np.sum(np.array(doubles['primary_has_detected_planet']))
        N_det_d2 = np.sum(np.array(doubles['secondary_has_detected_planet']))

        N_det_d = N_det_d1 + N_det_d2

        ########################################
        # COMPUTE X_Γ, as a function of radius #
        ########################################

        singles_R_pt = np.array(
                singles['R_pt'][singles['single_has_detected_planet']])
        singles_R_pa = np.array(
                singles['R_pa'][singles['single_has_detected_planet']])

        primaries_R_pt = np.array(
                doubles['primary_R_pt'][doubles['primary_has_detected_planet']])
        primaries_R_pa = np.array(
                doubles['primary_R_pa'][doubles['primary_has_detected_planet']])

        secondaries_R_pt = np.array(
                doubles['secondary_R_pt'][doubles['secondary_has_detected_planet']])
        secondaries_R_pa = np.array(
                doubles['secondary_R_pa'][doubles['secondary_has_detected_planet']])

        if index==0:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.integrate import trapz

            f,ax=plt.subplots()
            sns.distplot(primaries_R_pt, bins=np.arange(0,20+1,1), ax=ax,
                    kde=False, norm_hist=True, label='true')
            sns.distplot(primaries_R_pa, bins=np.arange(0,20+1,1), ax=ax,
                    kde=False, norm_hist=True, label='apparent, $R_{p,a}=R_{p,t}D^{1/2}$')
            ax.legend(loc='best')
            ax.set_xlabel('$R_p\ [R_\oplus]$')
            ax.set_title('primaries')
            f.savefig('tests/primaries_Rp_distribution.pdf')

            plt.close('all')
            f,ax=plt.subplots()
            sns.distplot(secondaries_R_pt, bins=np.arange(0,20+1,1), ax=ax,
                    kde=False, norm_hist=True, label='true')
            sns.distplot(secondaries_R_pa, bins=np.arange(0,20+1,1), ax=ax,
                    kde=False, norm_hist=True,
                    label='apparent, $R_{p,a}=R_{p,t}R_{\star,a}D^{1/2}/R_{star,t}$')
            ax.set_xlabel('$R_p\ [R_\oplus]$')
            ax.set_title('secondaries')
            ax.legend(loc='best')
            f.savefig('tests/secondaries_Rp_distribution.pdf')

            raise Exception


        #TODO FIXME: bin to compare occ rates? Must do. & bin + compare over
        #different montecarlo realizations.
        #TODO below

        Γ_t = N_planets/N_stars
        Γ_a = (N_det_s + N_det_d)/(N_s + N_d) * (1/f_sg[0])

        # the above is in the limit of counting all planets in binaries in the
        # occ rate. the below does not count any of them.
        Γ_a_upperlimit = N_det_s/(N_s+N_d)*(1/f_sg[0])

        X_Γ = Γ_t/Γ_a
        X_Γ_upperlimit = Γ_t/Γ_a_upperlimit

        ######################
        # INTERESTING OUTPUT #
        ######################
        outdict = {'seed':seed,
                   'N_s':N_s,
                   'N_d':N_d,
                   'β':N_d/N_s,
                   'N_det_s':N_det_s,
                   'N_det_d':N_det_d,
                   'N_planets':N_planets,
                   'N_stars':N_stars,
                   'Γ_t':Γ_t,
                   'Γ_a':Γ_a,
                   'X_Γ':X_Γ,
                   'f_sg':f_sg[0],
                   'Γ_a_upperlimit':Γ_a_upperlimit,
                   'X_Γ_upperlimit':X_Γ_upperlimit,
                   'm_lim':m_lim
                   }

        outdf = pd.DataFrame(outdict, index=[0])
        print(outdf)

        outfname = 'model2_results.csv'

        if not os.path.exists(outfname):
            outdf.to_csv(outfname, index=False, header=True)
        else:
            with open(outfname, 'a') as f:
                outdf.to_csv(f, index=False, header=False)

