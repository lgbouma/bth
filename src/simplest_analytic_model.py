'''
verify the numerical expression for the number of detected planets in the
"simplest analytic model" case (tex'd up 17/07/26) is correct.
'''
from __future__ import division, print_function

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy import units as u, constants as c
from math import pi as π

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


if __name__ == '__main__':

    np.random.seed(42)

    ##############
    # INSTRUMENT #
    ##############
    A = 50*u.cm**2

    λ_min = 500*u.nm
    λ_max = 1000*u.nm

    ######################
    # STELLAR POPULATION #
    ######################

    # Binary fraction. BF = n_d / n_s.
    BF = 0.45
    # Bovy (2017) gives n_tot = n_s + n_d. We want the number density of single
    # star systems, n_s, and the number density of double star systems, n_d.
    #n_tot = 4.76e-4 / (u.pc**3)
    n_tot = 5e-4 / (u.pc**3)
    n_s = n_tot / (1+BF)
    n_d = BF * n_s

    # Binary light ratio.
    γ_R = 1

    # Stellar luminosities, radii, Teffs, and masses.
    L_1 = 1*u.Lsun
    L_d = (1+γ_R)*L_1 # double star system luminosity
    L_2 = L_d - L_1
    R_1 = 1*u.Rsun
    R_2 = 1*u.Rsun
    T_eff1 = (1/c.sigma_sb * L_1 / (4*π*R_1**2))**(1/4)
    T_eff2 = (1/c.sigma_sb * L_2 / (4*π*R_2**2))**(1/4)
    M_1 = 1*u.Msun
    M_2 = 1*u.Msun

    # Dilution parameter
    dil = L_1 / L_d

    # Surface photon number fluxes
    u1_lower = c.h*c.c/(λ_max*c.k_B*T_eff1)
    u1_upper = c.h*c.c/(λ_min*c.k_B*T_eff1)
    u1 = np.linspace(u1_lower, u1_upper, num=5e4)
    integral = np.trapz( u1*u1/(np.exp(u1) - 1),  u1)
    F_s1γ_N = 8*π*c.c * (c.k_B * T_eff1 / (c.h * c.c))**3 * integral

    u2_lower = c.h*c.c/(λ_max*c.k_B*T_eff2)
    u2_upper = c.h*c.c/(λ_min*c.k_B*T_eff2)
    u2 = np.linspace(u2_lower, u2_upper, num=5e4)
    integral = np.trapz( u2*u2/(np.exp(u2) - 1),  u2)
    F_s2γ_N = 8*π*c.c * (c.k_B * T_eff2 / (c.h * c.c))**3 * integral

    c_s = R_1**2 * F_s1γ_N
    c_d = R_1**2 * F_s1γ_N  +  R_2**2 * F_s2γ_N

    #####################
    # PLANET POPULATION #
    #####################

    R_p = 1*c.R_earth
    P = 100*u.day

    assert M_1 == M_2, 'currently a is written independent of which star '+\
                       'is being transited'
    a = (P**2 * c.G * M_1 / (4*π*π) )**(1/3)
    assert R_1 == R_2, 'currently T_dur is written independent of which star'+\
                       ' is being transited. δ is too.'
    T_dur = R_1 * P / (4*a) # averages over impact parameter

    δ = (R_p/R_1)**2

    # Fraction of stars in single star systems with planet of (R_p, P).
    Γ_ts = 0.5
    # Fraction per star in double star systems with planet of (R_p, P).
    Γ_td = 0.5

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
    m_lim = 7.0
    F_lim = F_0 * 10**(-2/5 * (m_lim - m_0))

    # Get distance limits
    d_max_s = (L_1 / (4*π*F_lim) )**(1/2)
    d_max_d = (L_d / (4*π*F_lim) )**(1/2)

    # Get number of stars in sample. It is an integer.
    N_s = int(np.floor( (n_s * 4*π/3 * d_max_s**3).cgs.value ))
    N_d = int(np.floor( (n_d * 4*π/3 * d_max_d**3).cgs.value ))


    ##################
    # BEGIN "SURVEY" #
    ##################

    singles = draw_star_positions(d_max_s.to(u.pc), N_s, 'single')
    doubles = draw_star_positions(d_max_d.to(u.pc), N_d, 'binary')
    #stars = pd.concat([singles, doubles])

    # Compute received number fluxes for idealized population of singles and
    # doubles. N.b. these are the received number fluxes of PLANETS (under the
    # assumption that every star gets a planet) not systems. Thus you need to
    # double the length of r_d.
    r_s = np.array(singles['r'])*u.pc
    r_d = np.array(pd.concat([doubles['r'],doubles['r']]))*u.pc
    F_sγ_N = c_s / (r_s)**2
    F_dγ_N = c_d / (r_d)**2

    # Compute analytic SNR distribution of transit events, and number of
    # detections.
    N_tra = T_obs / P

    x_s_a = δ*np.sqrt( F_sγ_N * A * N_tra * T_dur )
    x_s_a = x_s_a.cgs
    x_d_a = dil*δ*np.sqrt( F_dγ_N * A * N_tra * T_dur )
    x_d_a = x_d_a.cgs

    prob_x_s_a = 3/(d_max_s**3) * c_s**(3/2) * δ**3 * \
                (A*N_tra*T_dur)**(3/2) * x_s_a**(-4)
    prob_x_s_a = prob_x_s_a.cgs
    prob_x_d_a = 3/(d_max_d**3) * c_d**(3/2) * (dil*δ)**3 * \
                (A*N_tra*T_dur)**(3/2) * x_d_a**(-4)
    prob_x_d_a = prob_x_d_a.cgs


    # INDEPENDENT CHECK SHOWS EQ 31 AND 33, HARD-INTEGRATION, WORKS.
    #xs_ordered = [x.value for x,y in sorted(list(zip(x_s_a, prob_x_s_a))) 
    #               if x.value > x_min]
    #prob_xs_ordered = [y.value for x,y in sorted(list(zip(x_s_a, prob_x_s_a)))
    #                    if x.value > x_min]
    #xd_ordered = [x.value for x,y in sorted(list(zip(x_d_a, prob_x_d_a))) 
    #               if x.value > x_min]
    #prob_xd_ordered = [y.value for x,y in sorted(list(zip(x_d_a, prob_x_d_a)))
    #                    if x.value > x_min]
    #f_s_x_gt_xmin = np.trapz(prob_xs_ordered, xs_ordered)
    #f_d_x_gt_xmin = np.trapz(prob_xd_ordered, xd_ordered)
    #N_det_s_a = N_s * Γ_ts * f_s_x_gt_xmin
    #N_det_d_a = 2 * N_d * Γ_td * f_d_x_gt_xmin

    # Equations 32 and 34 are valid only when the SNR distributions lead to
    # fractions less than one.
    N_det_s_a = N_s * Γ_ts * \
                min(1/(d_max_s**3) * c_s**(3/2) * δ**3 * \
                    (A*N_tra*T_dur)**(3/2) * x_min**(-3),
                    1)
    N_det_d_a = 2 * N_d * Γ_td * \
                min(1/(d_max_d**3) * c_d**(3/2) * (dil*δ)**3 * \
                    (A*N_tra*T_dur)**(3/2) * x_min**(-3),
                    1)

    N_det_s_a = N_det_s_a
    N_det_d_a = N_det_d_a

    N_det_a = int(N_det_s_a + N_det_d_a)

    # Compute SNR distribution and number of detections, numerically. AKA we
    # actually drew the positions from a MC grid in this case.

    # First, construct arrays that randomly select which stars get single
    # planet systems. 
    planet_mask_s, planet_mask_d = [], []
    ind = 0
    while ind < N_s:
        if np.random.rand() < Γ_ts:
            planet_mask_s.append(ind)
        ind += 1
    ind = 0
    while ind < 2*N_d:
        if np.random.rand() < Γ_td:
            planet_mask_d.append(ind)
        ind += 1

    planet_mask_s = np.array(planet_mask_s)
    planet_mask_d = np.array(planet_mask_d)

    r_s = np.array(singles['r'])[planet_mask_s]*u.pc
    # Need to double the positions because these are now _planet_ positions.
    r_d = np.array(pd.concat([doubles['r'], doubles['r']]))[planet_mask_d]*u.pc

    F_sγ_N = c_s / (r_s)**2
    F_dγ_N = c_d / (r_d)**2

    x_s_n = δ*np.sqrt( F_sγ_N * A * N_tra * T_dur )
    x_s_n = x_s_n.cgs
    # "observed" signal to noise distribution for doubles.
    x_d_n = dil*δ*np.sqrt( F_dγ_N * A * N_tra * T_dur )
    x_d_n = x_d_n.cgs

    # FIXME: we are assuming every planet that exists in this universe transits.
    N_det_s_n = len( x_s_n[x_s_n > x_min] )
    N_det_d_n = len( x_d_n[x_d_n > x_min] )

    N_det_n = N_det_s_n + N_det_d_n

    ####################
    # SUMMARIZE SURVEY #
    ####################
    print('N detected numerically: {:d}\n N detected analytically: {:d}\n'.
          format(N_det_n, N_det_a))

    ####################################################
    # COMPARE ANALYTIC AND NUMERICAL SNR DISTRIBUTIONS #
    ####################################################
    f,ax = plt.subplots()

    # Singles
    hist_sn, bin_edges = np.histogram(x_s_n,
            bins=np.arange(0,500+2.5,2.5), normed=True)
    ax.step(bin_edges[:-1], hist_sn, where='post', color='black',
            label='bins: singles, numeric', zorder=-1)

    snr_ordered = [x.value for x,y in sorted(list(zip(x_s_a, prob_x_s_a)))]
    prob_snr_ordered_s = [y.value for x,y in
                            sorted(list(zip(x_s_a, prob_x_s_a)))]
    ax.plot(snr_ordered, prob_snr_ordered_s, color='black',
            label='line: singles, analytic', zorder=0)


    # Doubles
    hist_sn, bin_edges = np.histogram(x_d_n,
            bins=np.arange(0,500+2.5,2.5), normed=True)
    ax.step(bin_edges[:-1], hist_sn, where='post', color='black',
            label='bins: doubles, numeric', alpha=0.5, lw=1, zorder=-1)

    snr_ordered = [x.value for x,y in sorted(list(zip(x_d_a, prob_x_d_a)))]
    prob_snr_ordered_d = [y.value for x,y in
                            sorted(list(zip(x_d_a, prob_x_d_a)))]
    ax.plot(snr_ordered, prob_snr_ordered_d, color='black',
            label='line: doubles, analytic', alpha=0.5, lw=1, zorder=0)

    ax.vlines(x_min, 0, 1, color='black', alpha=0.2, linestyles='dotted',
            label='SNR treshold: {:.1f}'.format(x_min), zorder=-10)

    ax.legend(loc='upper right', fontsize='small')

    dist_ratio = (d_max_s / d_max_d)**3 * (c_d / c_s)**(-1/2) * (1/dil)

    txt = 'm_lim: {:.1f}\n'.format(m_lim)+\
          'N_s: {:d} single systems\n'.format(N_s)+\
          'N_d: {:d} double systems\n'.format(N_d)+\
          'N det planets: {:d} analytic. {:d} single, {:d} double\n'.format(
          N_det_a, int(N_det_s_a), int(N_det_d_a))+\
          'N det planets: {:d} numeric. {:d} single, {:d} double\n\n'.format(
          N_det_n, int(N_det_s_n), int(N_det_d_n))+\
          'median(doubles)/median(singles) (numerical): {:.3f}\n'.format(
          np.median(prob_snr_ordered_s)/np.median(prob_snr_ordered_d))+\
          '25th pct(doubles)/25th pct(singles) (numerical): {:.3f}\n'.format(
          np.percentile(prob_snr_ordered_s, 25)/
          np.percentile(prob_snr_ordered_d, 25))+\
          '75th pct(doubles)/75th pct(singles) (numerical): {:.3f}\n'.format(
          np.percentile(prob_snr_ordered_s, 75)/
          np.percentile(prob_snr_ordered_d, 75))+\
         'ratio of distributions (analytic, fixed distance): {:.3f}'.format(
          dist_ratio)

    ax.text(0.96,0.5,txt,horizontalalignment='right',
            verticalalignment='center',
            transform=ax.transAxes, fontsize='xx-small')

    ax.set_xlabel('SNR', fontsize='small')
    ax.set_ylabel('probability', fontsize='small')
    ax.set_title('doubles and singles normalized independently. "numeric"\n'+\
        'means draw positions, draw planets, then compute the SNRs of the\n'+\
        'planets. "analytic" means I drew positions and evaluted equations',
        fontsize='xx-small')
    ax.set_xlim([0,100])
    ax.set_yscale('log')
    ax.set_ylim([1e-4,1])
    #ax.set_ylim([0,.1])

    f.tight_layout()

    f.savefig('simplest_analytic_model_SNR_distribution.pdf', dpi=300)






# MAYBE KEEP: PLOT DISTRIBUTION OF DISTANCES

#hist, bin_edges = np.histogram(r_a, bins=np.arange(0,1+0.05,0.05), normed=True)
#
#f,ax = plt.subplots()
#
#ax.step(bin_edges[:-1], hist, where='post')
#ax.plot(bin_edges[:-1], 3*bin_edges[:-1]**2)
