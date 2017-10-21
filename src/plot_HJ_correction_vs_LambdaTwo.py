'''
Given that hot Jupiters are thought to be less common around less-massive
stars, it would be more sensible to consider Λ_2 < Λ_0 , while letting single
stars and primaries host planets at the same rate.

This plots the HJ multiplicative correction factor as a function of Λ_2 /Λ_0.
'''
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import matplotlib as mpl
mpl.use('pgf')
pgf_with_custom_preamble = {
    'pgf.texsystem': 'pdflatex', # xelatex is default; i don't have it
    'font.family': 'serif', # use serif/main font for text elements
    'text.usetex': True,    # use inline math for ticks
    'pgf.rcfonts': False,   # don't setup fonts from rc parameters
    'pgf.preamble': [
        '\\usepackage{amsmath}',
        '\\usepackage{amssymb}'
        ]
    }
mpl.rcParams.update(pgf_with_custom_preamble)

import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import os

if __name__ == '__main__':

    model_number = 3

    Λ_2_arr = np.arange(0, 0.5+0.05, 0.05)

    Λ_HJ_true_arr, Λ_HJ_inferred_arr = [], []

    # compute the HJ correction factors for each Λ_2
    for Λ_2 in Λ_2_arr:

        savdir = '../data/numerics/'
        fname = 'results_model_{:d}_Lambda2_{:.2f}.out'.format(
                model_number, Λ_2)
        df = pd.read_csv(savdir+fname)

        lower_bound = 8
        inds = df['bin_left'] > lower_bound
        Λ_HJ_true = np.sum(df[inds]['true_Γ'])
        Λ_HJ_inferred = np.sum(df[inds]['inferred_Γ'])

        Λ_HJ_true_arr.append(float(Λ_HJ_true))
        Λ_HJ_inferred_arr.append(float(Λ_HJ_inferred))

    Λ_HJ_true_arr = np.array(Λ_HJ_true_arr)
    Λ_HJ_inferred_arr = np.array(Λ_HJ_inferred_arr)

    X_correction = Λ_HJ_inferred_arr/Λ_HJ_true_arr

    Λ_0 = 0.5

    ##########################################
    # First plot: the inferred rates vs the true values.

    f,ax = plt.subplots(figsize=(4,4))

    ax.plot(Λ_2_arr/Λ_0, Λ_HJ_true_arr*1e3, marker='o', label='true')
    ax.plot(Λ_2_arr/Λ_0, Λ_HJ_inferred_arr*1e3, marker='o', label='inferred')

    ax.legend(loc='upper left',fontsize='medium')

    ax.set_xlabel('$\Lambda_2/\Lambda_0$', fontsize='large')
    ax.set_ylabel('HJ rate (planets per thousand stars)', fontsize='large')

    outname = '../results/HJ_correction_inputrate_vs_HJratevalues'
    f.savefig(outname+'.pdf', bbox_inches='tight')

    ##########################################
    # Then plot: the inferred rates vs the ratio.

    plt.close('all')
    f,ax = plt.subplots(figsize=(4,4))

    ax.plot(Λ_2_arr/Λ_0, X_correction, marker='o')

    ax.set_xlabel('$\Lambda_2/\Lambda_0$', fontsize='large')
    ax.set_ylabel('inferred HJ rate / true HJ rate', fontsize='large')

    outname = '../results/HJ_correction_inputrate_vs_HJrate_correction_factor'
    f.savefig(outname+'.pdf', bbox_inches='tight')
