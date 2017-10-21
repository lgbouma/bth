# -*- coding: utf-8 -*-
'''
see README.md
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
    np.random.seed(42)

    ##########################################
    # Inputs.
    quickrun = False
    slowrun = not quickrun
    model_number = 3

    BF = 0.44   # binary fraction. BF = n_d / (n_s+n_d). Raghavan+ 2010 solar.

    α = 3.5     # exponent in L~M^α
    β = 0       # exponent in p_vol_limited(q) ~ q^β
    γ = 0       # exponent in p(has planet|secondary w/ q) ~ q^γ
    δ = -2.92   # exponent in p(r) ~ r^δ, from Howard+ 2012

    # planet occ rates (only relative values matter).
    Λ_0 = 0.5 # fraction of selected singles with planet
    Λ_1 = 0.5 # fraction of selected primaries with planet
    Λ_2 = 0.5 # fraction of selected secondaries with planet
    ##########################################

    print('Running Model #{:d}.'.format(model_number))

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

    # arbitrary number of selected single stars. 1e6 is pretty fast.
    N_0 = int(5e2) if quickrun else int(5e6)

    B = BF / (1-BF) # prefactor in definition of μ

    # Get number of selected primaries.
    if model_number == 1:
        q = 1
        μ = B * (1 + q**α)**(3/2)
        N_1 = int(N_0 * μ) # number of selected primaries
    elif model_number == 2 or model_number == 3:
        integral = 0.484174086070513 # 17/10/14.2 analytic result
        N_d = N_0 * B * (2**(3/2) - 3*integral)
        N_1 = int(N_d) # number of primaries = number of double star systems

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
        q_grid = np.arange(1e-6, 1+1e-6, 1e-6)
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
    elif model_number == 2 or model_number == 3:
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
        #fine-tune lower bound to get Howard's HJ rate
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

    locind = (df['star_type']=='single') & (df['has_det_planet']==True)
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
    for type_i in ['single', 'primary', 'secondary']:
        assert np.all(np.isfinite(np.array(
            df[(df['star_type']==type_i) & \
            (df['has_det_planet']==True)]['r_a'])))
        assert np.all(~np.isfinite(np.array(
            df[(df['star_type']==type_i) & \
            (df['has_det_planet']==False)]['r_a'])))


    ##############################################################
    # TRUE RATE DENSITY VS APPARENT RATE DENSITY (NOMINAL MODEL) #
    ##############################################################

    if model_number == 1 or model_number == 2:
        Δr = 0.01
        radius_bins = np.arange(0, 1+Δr, Δr)
        r_pu = 1
    elif model_number == 3:
        Δr = 0.5
        radius_bins = np.arange(0, r_pu+Δr, Δr)

    true_dict = {}
    inferred_dict = {}
    N_p_at_r_p_inferred = 0
    N_p_at_r_p_true = 0

    types = ['single','primary','secondary']
    for type_i in types:

        # True rate densities
        r = df[(df['star_type']==type_i) & (df['has_planet'])]['r']
        N_p, edges = np.histogram(r, bins=radius_bins)

        true_dict[type_i] = {}
        true_dict[type_i]['N_p'] = N_p
        true_dict[type_i]['edges'] = edges

        # Inferred rate densities
        r_a = df[(df['star_type']==type_i) & (df['has_det_planet'])]['r_a']
        N_p_det, edges = np.histogram( r_a, bins=radius_bins )

        inferred_dict[type_i] = {}
        inferred_dict[type_i]['N_p'] = N_p_det/Q_g0
        inferred_dict[type_i]['edges'] = edges

        if model_number == 1 or model_number == 2:
            N_p_at_r_p_inferred += (len(r_a[r_a == r_p])/Q_g0)
            N_p_at_r_p_true += len(r[r == r_p])

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
    fname = '../data/results_model_{:d}_Lambda2_{:.1f}'.format(
            model_number, Λ_2)
    outdf.to_csv(fname+'.out', index=False)
    print('wrote output to {:s}'.format(fname))

    # The sum of the true distribution's histogrammed rate densities is the true
    # average occurrence rate, to three decimals.
    np.testing.assert_almost_equal(
            np.sum(outdf['true_Γ']),
            (N_0*Λ_0 + N_1*Λ_1 + N_2*Λ_2)/N_tot,
            decimal=3
            )

    ###############################
    # Nominal model sanity checks #
    ###############################
    if model_number == 1 or model_number== 2:
        Γ_inferred = N_p_at_r_p_inferred / N_tot_apparent
        Γ_true = N_p_at_r_p_true / N_tot
        X_Γ_rp_numerics =  Γ_inferred / Γ_true

    else:
        X_Γ_rp_numerics = inferred_dict['Γ'][-1] / true_dict['Γ'][-1]

    print(X_Γ_rp_numerics)

    if model_number == 1:
        w_a = (1+μ)**(-1)
        w_b = μ*(1+μ)**(-1)

        w0 = (1+2*μ)**(-1)
        w1 = μ * (1+2*μ)**(-1)
        w2 = μ * (1+2*μ)**(-1)

        X_Γ_rp_analytic = w_a/(w0+w1+w2)
        print(X_Γ_rp_analytic)

        # when using larger N_0, gets down to 3 or 4 decimals
        np.testing.assert_almost_equal(
                X_Γ_rp_numerics,
                X_Γ_rp_analytic,
                decimal=2
                )

    elif model_number == 2:
        c_0 = 0.494087 # mathematica, 171011_integrals.nb
        X_Γ_rp_analytic = 3*c_0*Λ_0/(Λ_0+Λ_1+Λ_2)

        np.testing.assert_almost_equal(
                X_Γ_rp_numerics,
                X_Γ_rp_analytic,
                decimal=2
                )

    ###############
    # ERROR CASES #
    ###############
    for error_case_number in [1,2,3]:
        print('Beginning error case {:d}'.format(error_case_number))

        true_dict = {}
        inferred_dict = {}

        for type_i in types:

            # True rate densities are invariant of error case.
            r = df[(df['star_type']==type_i) & (df['has_planet'])]['r']
            N_p, edges = np.histogram(r, bins=radius_bins)

            true_dict[type_i] = {}
            true_dict[type_i]['N_p'] = N_p
            true_dict[type_i]['edges'] = edges

            # Inferred rate densities. Error case #3 gets incorrect, other two
            # are correct.
            if error_case_number == 1 or error_case_number == 2:
                r_a = df[(df['star_type']==type_i) & (df['has_det_planet'])]['r']
            elif error_case_number == 3:
                r_a = df[(df['star_type']==type_i) & (df['has_det_planet'])]['r_a']

            inferred_dict[type_i] = {}

            if error_case_number == 1 or error_case_number == 3:
                # How does the observer ideally account for Q_gi and Q_ci, the
                # geometric and completeness terms on a system-type specific
                # basis?

                df_detd = df[(df['star_type']==type_i) & (df['has_det_planet'])]

                weighted_N_p_det = []

                for j in range(len(radius_bins)-1):

                    this_N_p_det = 0

                    lower_edge = radius_bins[j]
                    upper_edge = radius_bins[j+1]

                    inds = (r_a > lower_edge) & (r_a <= upper_edge)

                    if not np.any(inds):
                        # If there are no detected planets in this bin,
                        # detection efficiency is irrelevant for inferred rate.
                        this_weighted_N_p_det = 0

                    else:
                        # This bin has detected planets. Each planet contibutes
                        # to "N_p_det" inversely weighted by the detection
                        # efficiency.
                        q = df_detd[inds]['q']

                        if type_i == 'single':
                            #Need this for above weighting to be fulfilled.
                            Q_g = Q_g0*np.ones(len(inds[inds]))
                            Q_c = 1
                        elif type_i == 'primary':
                            Q_g = Q_g1
                            Q_c = (1+q**α)**(-3)
                        elif type_i == 'secondary':
                            Q_g = Q_g0 * q**(2/3)
                            Q_c = (1+q**(-α))**(-3) * q**(-5)

                        Q = Q_g * Q_c

                        this_weighted_N_p_det = np.sum(1/Q)

                    weighted_N_p_det.append(this_weighted_N_p_det)

                inferred_dict[type_i]['N_p'] = np.array(weighted_N_p_det)

            elif error_case_number == 2:

                N_p_det, edges = np.histogram( r_a, bins=radius_bins )
                Q = Q_g0
                inferred_dict[type_i]['N_p'] = N_p_det/Q

            inferred_dict[type_i]['edges'] = edges

        N_tot = N_0 + N_1 + N_2
        true_dict['Γ'] = (true_dict['single']['N_p'] + \
                         true_dict['primary']['N_p'] + \
                         true_dict['secondary']['N_p'])/N_tot
        true_dict['r'] = radius_bins

        if error_case_number == 2 or error_case_number == 3:
            N_tot_apparent = N_tot
        elif error_case_number == 1:
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
        fname = '../data/results_model_{:d}_error_case_{:d}_Lambda2_{:.1f}.out'.format(
                model_number, error_case_number, Λ_2)
        outdf.to_csv(fname, index=False)
        print('\twrote output to {:s}'.format(fname))
