'''
first, for models 1,2, 3:
>>> python numerical_models.py
then:
>>> python plot_rate_densities_vs_radius.py

makes plots for paper
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
import pandas as pd
import numpy as np

from brokenaxes import brokenaxes

def make_plot(model_number, logx=None, logy=None, withtext=None,
        stdout=False, brokenx=None, Λ_2=None):

    # Make summary plot
    if isinstance(Λ_2,float) or isinstance(Λ_2,int):
        fname = '../data/results_model_{:d}_Lambda2_{:.1f}.out'.format(
                model_number, Λ_2)
    else:
        fname = '../data/results_model_'+repr(model_number)+'.out'

    df = pd.read_csv(fname)

    if not brokenx:
        f, ax = plt.subplots(figsize=(4,4))
    else:
        #cf https://github.com/bendichter/brokenaxes/blob/master/brokenaxes.py
        f = plt.figure(figsize=(4,4))
        bigax = brokenaxes(
                xlims=((0.695, .715), (.985, 1.005)),
                d=0.02,
                tilt=85,
                hspace=.05,
                despine=True)
        ax=bigax

    if model_number == 3:
        xvals = np.append(0, df['bin_left'])
        ytrue = np.append(0, df['true_Γ'])
        yinferred = np.append(0, df['inferred_Γ'])
    elif model_number == 1 or model_number == 2:
        xvals = np.append(df['bin_left'],[1,1.1])
        ytrue = np.append(df['true_Γ'],[0,0])
        yinferred = np.append(df['inferred_Γ'],[0,0])

    ax.step(xvals, ytrue, where='post', label='true')

    ax.step(xvals, yinferred, where='post', label='inferred')

    if brokenx:
        ax.legend(loc='upper left',fontsize='medium')
    else:
        ax.legend(loc='best',fontsize='medium')

    ax.set_xlabel('planet radius, $r$ [$R_\oplus$]', fontsize='large')

    ax.set_ylabel('planets per star, $\Lambda$', fontsize='large')

    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    if logx or logy:
        outname = '../results/rate_density_vs_radius_logs_model_'+\
                 repr(model_number)
    else:
        outname = '../results/rate_density_vs_radius_model_'+\
                 repr(model_number)
    if brokenx:
        outname += '_brokenx'

    if not brokenx:
        if (model_number == 1 or model_number == 2) and not (logx or logy):
            ax.set_xlim([0.5,1.02])
        elif (model_number == 1 or model_number == 2) and (logx or logy):
            ax.set_xlim([-0.02,1.02])

    if model_number == 3:
        # Assess HJ rate difference.
        from scipy.integrate import trapz

        #Howard 2012 boundary #1 and boundary #2:
        for lower_bound in [5.6,8]:
            inds = df['bin_left'] > lower_bound
            #Λ_HJ_true = trapz(df[inds]['true_Γ'], df[inds]['bin_left'])
            #Λ_HJ_inferred = trapz(df[inds]['inferred_Γ'], df[inds]['bin_left'])
            Λ_HJ_true = np.sum(df[inds]['true_Γ'])
            Λ_HJ_inferred = np.sum(df[inds]['inferred_Γ'])

            #Λ_true = trapz(df['true_Γ'], df['bin_left'])
            #Λ_inferred = trapz(df['inferred_Γ'], df['bin_left'])
            Λ_true = np.sum(df['true_Γ'])
            Λ_inferred = np.sum(df['inferred_Γ'])

            txt = \
            '''
            with $r>${:.1f}$R_\oplus$,
            $\Lambda$_HJ_true:      {:.4f} planets per star
            $\Lambda$_HJ_inferred:  {:.4f} planets per star
            true/inferred:  {:.2f}.

            Integrated over all $r$,
            $\Lambda$_true:         {:.3f} planets per star
            $\Lambda$_inferred:     {:.3f} planets per star
            true/inferred:  {:.2f}.
            '''.format(
            lower_bound,
            Λ_HJ_true,
            Λ_HJ_inferred,
            Λ_HJ_true/Λ_HJ_inferred,
            Λ_true,
            Λ_inferred,
            Λ_true/Λ_inferred
            )
            if stdout:
                print(txt)

        if withtext:
            ax.text(0.96,0.5,txt,horizontalalignment='right',
                    verticalalignment='center',
                    transform=ax.transAxes, fontsize='x-small')
            outname += '_withtext'

        if isinstance(Λ_2,float) or isinstance(Λ_2,int):
            outname += '_Lambda2_{:.1f}'.format(Λ_2)

    f.savefig(outname+'.pdf', bbox_inches='tight')



if __name__ == '__main__':

    for model_number in [1,2,3]:

        make_plot(model_number)

    make_plot(1, brokenx=True)
    make_plot(2, logy=True)
    make_plot(3, logy=True)
    make_plot(3, withtext=True, stdout=True)

    make_plot(3, Λ_2=0)
    make_plot(3, logy=True, Λ_2=0)
    make_plot(3, withtext=True, stdout=True, Λ_2=0)
