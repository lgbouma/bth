'''
This program simulates the following toy surveys:

MODEL #1: Fixed stars, fixed planets, twin binaries.

MODEL #2: Fixed planets and primaries, varying secondaries.

For Model #1, there are analytic predictions for the correction at various
points -- these can be verified.

For Model #2, there are fewer, but there are still good tests we can apply
along the way to ensure the result is believable.
'''

from __future__ import division, print_function

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy import units as u, constants as c
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from math import pi as π
import os

global α, β, γ

if __name__ == '__main__':
    np.random.seed(41)

    ##########################################
    # Inputs.
    quickrun = False
    slowrun = not quickrun
    model_number = 3

    BF = 0.44 # binary fraction. BF = n_d / (n_s+n_d). Raghavan+ 2010 solar.

    α = 3.5     # exponent in L~M^α
    β = 0       # exponent in p_vol_limited(q) ~ q^β
    γ = 0       # exponent in p(has planet|secondary w/ q) ~ q^γ
    δ = -2.92   # exponent in p(r) ~ r^δ, from Howard+ 2012

    # planet occ rates (only relative values matter).
    Λ_0 = 0.5 # fraction of selected singles with planet
    Λ_1 = 0.5 # fraction of selected primaries with planet
    Λ_2 = 0.5 # fraction of selected secondaries with planet
    ##########################################

    if γ != 0:
        raise NotImplementedError

    ######################
    # STELLAR POPULATION #
    ######################
    # Construct a pandas DataFrame in which each row is a star that has been
    # "selected". Each selected star has properties:
    #   * star_type (str): single, primary, or secondary
    #   * is_searchable (bool): whether the star is searchable (determined by
    #       completeness for star-specific q).
    #   * q (float): binary mass ratio, if applicable
    ######################

    # arbitrary number of selected single stars
    N_0 = int(5e2) if quickrun else int(1e6)

    B = BF / (1-BF) # prefactor in definition of μ

    # Get number of selected primaries.
    if model_number == 1:
        q = 1
        μ = B * (1 + q**α)**(3/2)
        N_1 = int(N_0 * μ) # number of selected primaries
    elif model_number == 2 or model_number == 3:
        integral = 0.484174086070513 # 17/10/14.2 analytic result
        N_d = N_0 * B * (2**(3/2) - 3*integral)
        N_1 = int(N_d/2)

    N_2 = N_1 # number of selected secondaries

    # Construct population of stars: singles, primaries, secondaries.
    df = pd.concat((
        pd.DataFrame({'star_type':[type_i for _ in range(N_i)]}) for N_i,type_i in
        [(N_0,'single'),(N_1,'primary'),(N_2,'secondary')]
        ), ignore_index=True )

    # Assign values of `q` to primaries and secondaries
    if model_number == 1:
        df['q'] = q
        df.loc[df['star_type'] == 'single', 'q'] = np.nan
        q = np.ones(N_1)

    elif model_number == 2 or model_number == 3:
        df['q'] = np.nan

        # Inverse transform sampling to get samples of q
        q_grid = np.arange(0, 1+1e-6, 1e-6)
        prob_ml_q = q_grid**β * (1+q_grid**α)**(3/2)
        prob_ml_q /= trapz(prob_ml_q, q_grid)
        cdf_ml_q = np.append(0, np.cumsum(prob_ml_q)/np.max(np.cumsum(prob_ml_q)))
        func = interp1d(cdf_ml_q, np.append(0, q_grid))
        q_samples = func(np.random.uniform(size=int(N_1)))

        df.loc[df['star_type'] == 'primary', 'q'] = q_samples
        df.loc[df['star_type'] == 'secondary', 'q'] = q_samples

        q = q_samples

    np.testing.assert_equal(
            len(df[df['star_type']=='primary']),
            len(df[df['star_type']=='secondary'])
            )

    np.testing.assert_array_equal(
            df[df['star_type']=='primary']['q'],
            df[df['star_type']=='secondary']['q'])

    np.testing.assert_array_less(
            np.zeros(len(q)),
            q
            )

    # Select stars that are searchable. This means they win the completeness
    # lottery. NB it makes no comment about geometric transit probability.
    Q_c0 = 1
    single_is_searchable = np.ones(N_0).astype(bool)

    Q_c1 = (1+q**α)**(-3)
    primary_is_searchable = ( np.random.rand((N_1)) < Q_c1 )

    Q_c2 = (1+q**(-α))**(-3) * q**(-5)
    secondary_is_searchable = ( np.random.rand((N_2)) < Q_c2 )

    is_searchable = np.concatenate(
        (single_is_searchable,  primary_is_searchable, secondary_is_searchable)
        )
    df['is_searchable'] = is_searchable

    _1 = len(df[(df['is_searchable']==True)&(df['star_type']=='primary')])/\
            len(df[df['star_type']=='primary'])
    print("population's completeness fraction for primaries: {:.3g}".
            format(_1))

    _2 = len(df[(df['is_searchable']==True)&(df['star_type']=='secondary')])/\
            len(df[df['star_type']=='secondary'])
    print("population's completeness fraction for secondaries: {:.3g}".
            format(_2))

    if model_number == 1:
        if slowrun and N_0>=1e6:
            # we know that the completeness in model 1 should be 1/8.
            assert np.isclose(_1, 1/8, rtol=1e-2) \
                   and \
                   np.isclose(_2, 1/8, rtol=1e-2)
    if model_number == 2:
        # primaries should be more complete than 1/8
        np.testing.assert_array_less(
                1/8, #smaller
                _1   #larger
                )
        # secondaries should be (much) less complete than 1/8
        np.testing.assert_array_less(
                _2, #smaller
                1/8 #larger
                )

    #####################
    # PLANET POPULATION #
    #####################

    # Select stars with planets, and assign planet radii.
    single_has_planet = ( np.random.rand((N_0)) < Λ_0 )
    primary_has_planet = ( np.random.rand((N_1)) < Λ_1 )
    secondary_has_planet = ( np.random.rand((N_2)) < Λ_2 )

    has_planet = np.concatenate(
        (single_has_planet,  primary_has_planet, secondary_has_planet)
        )
    df['has_planet'] = has_planet

    if model_number == 1 or model_number == 2:
        # Assign arbitrary planet radius.
        r_p = 1 
        df['r'] = r_p
        df.loc[df['has_planet'] == False, 'r'] = np.nan

    elif model_number == 3:
        r_pl, r_pu = 2, 22.5 # [Rearth]. Lower and upper bound for truncation.

        # Inverse transform sample to get radii. Drawing from powerlaw
        # distribution above r_pl, and constant below (to avoid pileup).
        Δr = 1e-3
        r_grid = np.arange(0, r_pu+Δr, Δr)
        prob_r = np.minimum( r_grid**δ, r_pl**δ )
        prob_r /= trapz(prob_r, r_grid)
        cdf_r = np.append(0, np.cumsum(prob_r)/np.max(np.cumsum(prob_r)))
        func = interp1d(cdf_r, np.append(0, r_grid))
        r_samples = func(np.random.uniform(size=N_0+N_1+N_2))

        df['r'] = r_samples
        df.loc[df['has_planet'] == False, 'r'] = np.nan

    # Detected planets are those that transit, and whose hosts are searchable.
    Q_g0 = 0.3 # arbitrary geoemtric transit probability around singles
    Q_g1 = Q_g0
    Q_g2 = Q_g0 * q**(2/3)

    transits_0 = ( np.random.rand((N_0)) < Q_g0 )
    transits_1 = ( np.random.rand((N_1)) < Q_g1 )
    transits_2 = ( np.random.rand((N_2)) < Q_g2 )

    planet_transits = np.concatenate(
        (transits_0, transits_1, transits_2)
        )
    df['planet_transits'] = planet_transits
    # Cleanup: only stars with planets have "planets that transit"
    df.loc[df['has_planet'] == False, 'planet_transits'] = np.nan

    # Assign detected planets
    detected_planets = has_planet.astype(bool) & \
                       planet_transits.astype(bool) & \
                       is_searchable.astype(bool)
    df['has_det_planet'] = detected_planets

    ##########################
    # WHAT THE OBSERVER SEES #
    ##########################
    # Compute apparent radii for detected planets.
    # NB. we're saying the detectability is entirely determined by the mass
    # ratio (and so the "apparent radius" happens post-detection)
    df['r_a'] = np.nan

    locind = df['star_type']=='single'
    df.loc[locind, 'r_a'] = df.loc[locind, 'r']

    locind = (df['star_type']=='primary') & (df['has_det_planet']==True)
    this_q = df.loc[locind, 'q']
    this_r = df.loc[locind, 'r']
    df.loc[locind, 'r_a'] = this_r *(1+this_q**α)**(-1/2)

    locind = (df['star_type']=='secondary') & (df['has_det_planet']==True) 
    this_q = df.loc[locind, 'q']
    this_r = df.loc[locind, 'r']
    df.loc[locind, 'r_a'] = this_r *(1+this_q**(-α))**(-1/2) * (1/this_q)

    # Only things with detected planets should have an apparent radius.
    assert np.all(np.isfinite(np.array(
        df[(df['star_type']=='primary') & \
        (df['has_det_planet']==True)]['r_a'])))
    assert np.all(~np.isfinite(np.array(
        df[(df['star_type']=='primary') & \
        (df['has_det_planet']==False)]['r_a'])))

    assert np.all(np.isfinite(np.array(
        df[(df['star_type']=='secondary') & \
        (df['has_det_planet']==True)]['r_a'])))
    assert np.all(~np.isfinite(np.array(
        df[(df['star_type']=='secondary') & \
        (df['has_det_planet']==False)]['r_a'])))


    ##############################################
    # TRUE RATE DENSITY VS APPARENT RATE DENSITY #
    ##############################################

    if model_number == 1 or model_number == 2:
        Δr = 0.01
        radius_bins = np.arange(0, 1+Δr, Δr)
        r_pu = 1
    elif model_number == 3:
        Δr = 0.5
        radius_bins = np.arange(0, r_pu+Δr, Δr)

    true_dict = {}
    inferred_dict = {}

    types = ['single','primary','secondary']
    for type_i in types:

        r = df[(df['star_type']==type_i) & (df['has_planet'])]['r']
        N_p, edges = np.histogram(r, bins=radius_bins)

        true_dict[type_i] = {}
        true_dict[type_i]['N_p'] = N_p
        true_dict[type_i]['edges'] = edges

        # Apparent detected rate densities
        r_a = df[(df['star_type']==type_i) & (df['has_det_planet'])]['r_a']
        N_p_det, edges = np.histogram( r_a, bins=radius_bins )

        # Inferred rate densities
        inferred_dict[type_i] = {}
        inferred_dict[type_i]['N_p'] = N_p_det/Q_g0
        inferred_dict[type_i]['edges'] = edges

    N_tot = N_0 + N_1 + N_2
    true_dict['Γ'] = (true_dict['single']['N_p'] + \
                     true_dict['primary']['N_p'] + \
                     true_dict['secondary']['N_p'])/N_tot
    true_dict['r'] = radius_bins

    N_tot_apparent = N_0 + N_1
    inferred_dict['Γ'] = \
                     (inferred_dict['single']['N_p'] + \
                     inferred_dict['primary']['N_p'] + \
                     inferred_dict['secondary']['N_p'])/N_tot_apparent
    inferred_dict['r'] = radius_bins

    outdf = pd.DataFrame(
            {'bin_left': edges[:-1],
             'true_Γ': true_dict['Γ'],
             'inferred_Γ': inferred_dict['Γ']
            }
            )
    fname = '../data/results_model_'+repr(model_number)+'.out'
    outdf.to_csv(fname, index=False)
    print('wrote output to {:s}'.format(fname))
    
    #######################
    # Final sanity checks #
    #######################
    X_Γ_at_rp = inferred_dict['Γ'][-1] / true_dict['Γ'][-1]
    print(X_Γ_at_rp)

    if model_number == 1:
        w_a = (1+μ)**(-1)
        w_b = μ*(1+μ)**(-1)

        w0 = (1+2*μ)**(-1)
        w1 = μ * (1+2*μ)**(-1)
        w2 = μ * (1+2*μ)**(-1)

        X_Γ_analytic = w_a/(w0+w1+w2)
        print(X_Γ_analytic)

        # this gets down to 3 or 4 decimals when using more things
        np.testing.assert_almost_equal(
                X_Γ_at_rp,
                X_Γ_analytic,
                decimal=2)

    #NOTE: it would be great if we had analytic predictions for model #2 output
