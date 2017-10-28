'''
What is the rate correction for earth analogs in Model #3?

Print the result.

Make the plot of rate as a function of Λ_2/Λ_0 as well.
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

if __name__ == '__main__':

    model_number = 3

    Λ_0 = 0.5
    Λ_2_arr = np.arange(0, 0.5+0.05, 0.05)

    Λ_earth_true_arr, Λ_earth_inferred_arr, Λ_earth_true_primary_arr = [], [], []

    # compute the rate for each Λ_2
    for Λ_2 in Λ_2_arr:

        savdir = '../data/numerics/'
        fname = 'results_model_{:d}_Lambda2_{:.2f}.out'.format(
                model_number, Λ_2)
        df = pd.read_csv(savdir+fname)

        for ix, bl in enumerate([0.5, 1.0]):

            inds = (df['bin_left'] == bl)

            Λ_earth_true = df[inds]['true_Γ']
            Λ_earth_inferred = df[inds]['inferred_Γ']
            Λ_earth_true_primary = df[inds]['true_primary_Γ']

            Λ_earth_true_arr.append(float(Λ_earth_true))
            Λ_earth_inferred_arr.append(float(Λ_earth_inferred))
            Λ_earth_true_primary_arr.append(float(Λ_earth_true_primary))

    Λ_earth_true_arr = np.array(Λ_earth_true_arr)
    Λ_earth_inferred_arr = np.array(Λ_earth_inferred_arr)
    Λ_earth_true_primary_arr = np.array(Λ_earth_true_primary_arr)

    true_pt5_to_1 = Λ_earth_true_arr[0::2]
    true_1_to_1pt5 = Λ_earth_true_arr[1::2]
    inferred_pt5_to_1 = Λ_earth_inferred_arr[0::2]
    inferred_1_to_1pt5 = Λ_earth_inferred_arr[1::2]
    true_primary_pt5_to_1 = Λ_earth_true_primary_arr[0::2]
    true_primary_1_to_1pt5 = Λ_earth_true_primary_arr[1::2]

    Λ_earth_true = true_1_to_1pt5 + true_pt5_to_1
    Λ_earth_inferred = inferred_pt5_to_1 + inferred_1_to_1pt5
    Λ_earth_true_primary = true_primary_pt5_to_1 + true_primary_1_to_1pt5

    print('true')
    print(Λ_earth_true)
    print('inferred')
    print(Λ_earth_inferred)
    print('true primary')
    print(Λ_earth_true_primary)

    ################################################
    # Plot: the inferred rates vs the true values. #
    ################################################

    f,ax = plt.subplots(figsize=(4,4))

    ax.plot(Λ_2_arr/Λ_0, Λ_earth_true*1e2, marker='o',
        label='true (whole population)')
    ax.plot(Λ_2_arr/Λ_0, Λ_earth_inferred*1e2, marker='o', label='inferred')
    ax.plot(Λ_2_arr/Λ_0, Λ_earth_true_primary*1e2, marker='o',
        label='true (only primaries)')

    ax.legend(loc='best',fontsize='medium')

    ax.set_xlabel('$\Lambda_2/\Lambda_0$', fontsize='large')
    ax.set_ylabel('Earth rate (planets per hundred stars)', fontsize='large')

    outname = '../results/earth_inputrate_vs_etaearthratevalues'
    f.savefig(outname+'.pdf', bbox_inches='tight')
