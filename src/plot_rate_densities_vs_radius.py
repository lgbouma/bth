'''
first, for models 1,2, 3:
>>> python numerical_models.py
then:
>>> python rate_densities_vs_radius.py
'''
import matplotlib.pyplot as plt
import pandas as pd

def make_plot(model_number, logx=None, logy=None, withtext=None,
        stdout=False):

    # Make summary plot
    fname = '../data/results_model_'+repr(model_number)+'.out'
    df = pd.read_csv(fname)

    f, ax = plt.subplots(figsize=(4,4))

    ax.step(df['bin_left'], df['true_Γ'], where='pre', label='true')

    ax.step(df['bin_left'], df['inferred_Γ'], where='pre', label='inferred')

    ax.legend(loc='best',fontsize='small')

    ax.set_xlabel('planet radius, $r$ [$R_\oplus$]')

    ax.set_ylabel('planets per star per bin, $\Gamma(r)$ [$R_\oplus^{-1}$]')

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

    if model_number == 3:
        # Assess HJ rate difference.
        from scipy.integrate import trapz

        #Howard 2012 boundary #1 and boundary #2:
        for lower_bound in [8,5.6]:
            inds = df['bin_left'] > lower_bound
            Λ_HJ_true = trapz(df[inds]['true_Γ'], df[inds]['bin_left'])
            Λ_HJ_inferred = trapz(df[inds]['inferred_Γ'], df[inds]['bin_left'])

            Λ_true = trapz(df['true_Γ'], df['bin_left'])
            Λ_inferred = trapz(df['inferred_Γ'], df['bin_left'])

            txt = \
            '''
            with $r>${:.1f}$R_\oplus$,
            Λ_HJ_true:      {:.4f} planets per star
            Λ_HJ_inferred:  {:.4f} planets per star
            true/inferred:  {:.2f}.

            Integrated over all $r$,
            Λ_true:         {:.3f} planets per star
            Λ_inferred:     {:.3f} planets per star
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

    f.savefig(outname+'.pdf', bbox_inches='tight')



if __name__ == '__main__':

    for model_number in [1,2,3]:

        make_plot(model_number)

    make_plot(2, logy=True)
    make_plot(3, withtext=True, stdout=True)
