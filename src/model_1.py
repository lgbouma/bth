'''
MODEL #1: FIXED STARS, FIXED PLANETS, TWIN BINARIES

The point of this exercise is to develop a numerical procedure to simulate
these toy surveys.
'''

from __future__ import division, print_function

import numpy as np, pandas as pd
from astropy import units as u, constants as c
from math import pi as π
import os

global α, β
α = 3.5

#####################
# UTILITY FUNCTIONS #
#####################
def get_L(M):
    '''
    assume input mass, M, a float or vector, is in solar units.
    '''

    return M**α


if __name__ == '__main__':

    ######################
    # STELLAR POPULATION #
    ######################
    #
    # Construct a pandas DataFrame in which each row is a star that has been
    # "selected". Each selected star has properties:
    #   * star_type (str): single, primary, or secondary
    #   * is_searchable (bool): whether the star is searchable.
    #   * q (float): binary mass ratio, if applicable

    quickrun = False
    slowrun = not quickrun

    # arbitrary number of selected single stars
    N_0 = int(5e2) if quickrun else int(1e5)

    BF = 0.44 # binary fraction. BF = n_d / (n_s+n_d). Raghavan+ 2010 solar.
    B = BF / (1-BF) # prefactor in definition of μ

    # For a given light ratio (or q), everything is defined.
    # In model #1, q=1.
    q = 1
    μ = B * (1 + q**α)**(3/2)

    N_1 = int(N_0 * μ) # number of selected primaries
    N_2 = int(N_1)     # number of selected secondaries

    # End with a population of stars: singles, primaries, secondaries.
    df = pd.concat((
        pd.DataFrame({'star_type':[type_i for _ in range(N_i)]}) for N_i,type_i in
        [(N_0,'single'),(N_1,'primary'),(N_2,'secondary')]
        ), ignore_index=True )

    df['q'] = q

    # Make sure only primaries and secondaries are assigned a value of `q`
    df.loc[df['star_type'] == 'single', 'q'] = np.nan

    # Select stars that are searchable. This means they win the completeness
    # lottery. NB it makes no commnt about geometric transit probability.
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

    _1 = len(df[(df['is_searchable']==True)&(df['star_type']=='secondary')])/\
            len(df[df['star_type']=='secondary'])

    _2 = len(df[(df['is_searchable']==True)&(df['star_type']=='primary')])/\
            len(df[df['star_type']=='primary'])

    # we know that the completeness in model 1 is 1/8.
    if slowrun:
        assert np.isclose(_1, 1/8, rtol=1e-2) \
               and \
               np.isclose(_2, 1/8, rtol=1e-2)

    #####################
    # PLANET POPULATION #
    #####################

    # Define rates (only relative values matter).
    Λ_0 = 0.5 # fraction of selected singles with planet
    Λ_1 = 0.5 # fraction of selected primaries with planet
    Λ_2 = 0.5 # fraction of selected secondaries with planet

    # Select stars with planets, and assign planet radii.
    single_has_planet = ( np.random.rand((N_0)) < Λ_0 )
    primary_has_planet = ( np.random.rand((N_1)) < Λ_1 )
    secondary_has_planet = ( np.random.rand((N_2)) < Λ_2 )

    has_planet = np.concatenate(
        (single_has_planet,  primary_has_planet, secondary_has_planet)
        )
    df['has_planet'] = has_planet

    r_p = 1
    df['r'] = r_p
    df.loc[df['has_planet'] == False, 'r'] = np.nan

    # Detected planets are those that transit, and whose hosts are searchable.
    Q_g0 = 0.1 # some number
    Q_g1 = Q_g0
    Q_g2 = Q_g0

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
    # ratio (and the "apparent radius" happens post-detection)
    df['r_a'] = np.nan

    df.loc[df['star_type']=='single', 'r_a'] = r_p

    df.loc[(df['star_type']=='primary') & (df['has_det_planet']==True), 'r_a'] = \
    df.loc[(df['star_type']=='primary') & (df['has_det_planet']==True), 'r']\
                   *(1+q**α)**(-1/2)

    df.loc[(df['star_type']=='secondary') & (df['has_det_planet']==True), 'r_a'] = \
    df.loc[(df['star_type']=='secondary') & (df['has_det_planet']==True), 'r']\
                   *(1+q**(-α))**(-1/2) ** (1/q)

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

    radius_bins = np.arange(0,1+1e-1,1e-1)

    w_a = (1+μ)**(-1)
    w_b = μ*(1+μ)**(-1)

    w0 = (1+2*μ)**(-1)
    w1 = μ * (1+2*μ)**(-1)
    w2 = μ * (1+2*μ)**(-1)

    true_dict = {}
    apparent_dict = {}
    inferred_dict = {}

    types = ['single','primary','secondary']
    for type_i in types:

        N_i = len(df[(df['star_type']==type_i)])

        r = df[(df['star_type']==type_i) & (df['has_planet'])]['r']
        N_p, edges = np.histogram(r, bins=radius_bins)

        true_dict[type_i] = {}
        true_dict[type_i]['N_p'] = N_p
        true_dict[type_i]['edges'] = edges

        # Apparent detected rate densities
        r_a = df[(df['star_type']==type_i) & (df['has_det_planet'])]['r_a']
        N_p_det, edges = np.histogram( r_a, bins=radius_bins )

        apparent_dict[type_i] = {}
        apparent_dict[type_i]['N_p_det'] = N_p_det
        apparent_dict[type_i]['edges'] = edges

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

    X_Γ_at_rp = inferred_dict['Γ'][-1] / true_dict['Γ'][-1]
    X_Γ_analytic = w_a/(w0+w1+w2)

    # this gets down to 3 or 4 decimals when using more things
    np.testing.assert_almost_equal(
            X_Γ_at_rp,
            X_Γ_analytic,
            decimal=2)

    print(X_Γ_at_rp)
    print(X_Γ_analytic)
