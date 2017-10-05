'''
How does binarity affect the apparent radius distribution? (vs an input true
radius distribution)

you can do it all semianalytically, following notes of 17/09/03.2-3

and, more ACCURATELY (AKA with math bug fixes) the notes of 17/10/04.3-4
'''

import numpy as np
from scipy.integrate import trapz
import matplotlib.pyplot as plt

global α
α = 3.5

def prob_gammaR(γ_R):

    prob_γ_R = np.zeros_like(γ_R)

    inds = (γ_R<1) & (γ_R>0.1**α)
    prob_γ_R[inds] = (1 + γ_R[inds])**(3/2) * γ_R[inds]**((1-α)/α)

    return prob_γ_R/trapz(prob_γ_R, γ_R)


def prob_Rpt_uniform(R_pt, R_pu, R_pl):
    '''
    uniform distribution
    R_pt: array for to return the probability over
    R_pu, R_pl: floats of upper and lower bounds
    '''

    c_Rp = 1/(R_pu - R_pl)

    prob_Rpt = np.zeros_like(R_pt)

    inds = (R_pt > R_pl) & (R_pt < R_pu)
    prob_Rpt[inds] = c_Rp

    return prob_Rpt


def prob_Rpt_loguniform(R_pt, R_pu, R_pl):
    '''
    loguniform distribution (not normalized correctly; done in post)
    R_pt: array for to return the probability over
    R_pu, R_pl: floats of upper and lower bounds
    '''

    const = 1

    prob_Rpt = np.zeros_like(R_pt)

    inds = (R_pt > R_pl) & (R_pt < R_pu)
    prob_Rpt[inds] = const / R_pt[inds]

    return prob_Rpt


def prob_Rpt_lognormal(R_pt, μ, σ):
    '''
    lognormal distribution (not normalized correctly; done in post)
    see e.g. https://en.wikipedia.org/wiki/Log-normal_distribution
    R_pt: array for to return the probability over
    μ,σ: mean and standard deviation
    '''

    const = 1/(σ*2*np.pi)

    prob_Rpt = np.zeros_like(R_pt)

    inds = R_pt > 0

    values = (1/R_pt[inds]) * np.exp( -0.5 * (np.log(R_pt[inds]) - μ)**2/(σ**2) )

    prob_Rpt[inds] = const * values

    return prob_Rpt


def prob_Rpt_trunclognormal(R_pt, R_pl, R_pu, μ, σ):
    '''
    truncated lognormal distribution (not normalized correctly; done in post)
    R_pt: array for to return the probability over
    R_pl, R_pu: lower and upper bound for truncation
    μ,σ: mean and standard deviation
    '''

    const = 1/(σ*2*np.pi)

    prob_Rpt = np.zeros_like(R_pt)

    inds = R_pt > 0

    values = (1/R_pt[inds]) * np.exp( -0.5 * (np.log(R_pt[inds]) - μ)**2/(σ**2) )
    prob_Rpt[inds] = const * values

    inds = (R_pt > R_pl) & (R_pt < R_pu)
    prob_Rpt[~inds] = 0

    return prob_Rpt


def prob_Rpt_Howardpower(R_pt, R_pl, R_pu):
    '''
    Howard et al. (2012) radius distribution of HJs follows a power law of
    slope -2.92 \pm 0.11

    R_pt: array for to return the probability over
    R_pl, R_pu: lower and upper bound for truncation
    '''

    const = 1
    power = -2.92

    prob_Rpt = np.zeros_like(R_pt)

    inds = (R_pt > R_pl) & (R_pt < R_pu)
    prob_Rpt[inds] = const * (R_pt[inds])**power

    return prob_Rpt


def logistic_fn(x, L, k, x_0):

    val = L / (1 + np.exp(-k * (x-x_0)))
    return val

def _make_Rpobs_plot(R_pt, prob_Rpts, R_pas, prob_Rpos_d1, prob_Rpos_d2,
        sstr=None, howardpower=False):

    nrows = 10
    f, axs = plt.subplots(figsize=(4,nrows*(7/4)), nrows=nrows, ncols=1, sharex=True)

    norm = trapz(prob_Rpts, R_pt)
    prob_Rpts /= norm

    axs[0].plot(R_pt, prob_Rpts, label='true', c='gray')

    axs[0].set_ylabel(r'$\rho_{R_p^{\mathrm{true}}} (R_p^{\mathrm{true}}) $')
    axs[0].legend(loc='upper right', fontsize='small')

    norm_d1 = trapz(prob_Rpos_d1, R_pas)
    prob_Rpos_d1 /= norm_d1
    norm_d2 = trapz(prob_Rpos_d2, R_pas)
    prob_Rpos_d2 /= norm_d2

    axs[1].plot(R_pt, prob_Rpts, label='s')
    axs[1].plot(R_pas, prob_Rpos_d1, label='d1')
    axs[1].plot(R_pas, prob_Rpos_d2, label='d2')

    axs[1].set_ylabel(r'$\rho_{R_p^{\mathrm{obs}}} (R_p^{\mathrm{obs}}) $')
    axs[1].legend(loc='upper right', fontsize='small')

    if not howardpower:
        f_det = logistic_fn(R_pas, 0.95, 1, 10)
    else:
        f_det = logistic_fn(R_pas, 0.97, 2, 3)
    axs[2].plot(R_pas, f_det, 'k--')
    axs[2].set_ylabel('$f_{\mathrm{det}}(R_p^{\mathrm{obs}})$')

    prob_det_Rpo_s = prob_Rpts * logistic_fn(R_pt, 0.95, 1, 10)
    prob_det_Rpo_s /= trapz(prob_det_Rpo_s, R_pt)
    prob_det_Rpo_d1 = prob_Rpos_d1 * f_det
    prob_det_Rpo_d1 /= trapz(prob_det_Rpo_d1, R_pas)
    prob_det_Rpo_d2 = prob_Rpos_d2 * f_det
    prob_det_Rpo_d2 /= trapz(prob_det_Rpo_d2, R_pas)

    axs[3].plot(R_pt, prob_det_Rpo_s, label='s')
    axs[3].plot(R_pas, prob_det_Rpo_d1, label='d1')
    axs[3].plot(R_pas, prob_det_Rpo_d2, label='d2')

    axs[3].set_ylabel(r'$\rho_{\mathrm{det}}(R_p^{\mathrm{obs}})$')
    axs[3].legend(loc='upper right', fontsize='small')

    # From GIC. "singles" defined to all be sun-like. "doubles" with a
    # mass-ratio taken from the appropriate mag-limited distribution, and the
    # fraction of them that are sunlike computed numerically by `model_3.py`
    # N singles: 84827
    # N doubles: 67855
    # frac of secondaries sunlike: 0.3736
    N_s = 84827
    N_d1 = 67855
    N_d2 = N_d1

    frac_d2_sunlike = 0.3736
    N_d2_sunlike = N_d2 * frac_d2_sunlike

    N_tot = N_s + N_d1 + N_d2
    w_s = N_s / N_tot
    w_d1 = N_d1 / N_tot
    w_d2 = N_d2 / N_tot # first assume that secondaries get same input occ rate

    axs[4].plot(R_pt, prob_det_Rpo_s*w_s, label='s')
    axs[4].plot(R_pas, prob_det_Rpo_d1*w_d1, label='d1')
    axs[4].plot(R_pas, prob_det_Rpo_d2*w_d2, label='d2')

    summed = prob_det_Rpo_s*w_s + prob_det_Rpo_d1*w_d1 + prob_det_Rpo_d2*w_d2
    axs[4].plot(R_pas, summed, label='total', c='black')

    txt = '$N_s$: {:d}\n$N_d$ {:d}'.format(
            N_s, N_d1) + '\n$w_i = N_i/N_{\mathrm{tot}}$'
    axs[4].text(0.03, 0.97, txt, verticalalignment='top',
            horizontalalignment='left', transform=axs[4].transAxes,
            fontsize='xx-small')

    axs[4].set_ylabel('$N_{\mathrm{det}}(R_p^{\mathrm{obs}}) / N_{\mathrm{tot}}$')
    axs[4].legend(loc='upper right', fontsize='x-small')


    axs[5].plot(R_pt, prob_det_Rpo_s*w_s, label='s')
    axs[5].plot(R_pas, prob_det_Rpo_d1*w_d1, label='d1')
    axs[5].plot(R_pas, prob_det_Rpo_d2*w_d2*frac_d2_sunlike, label='d2')
    summed = prob_det_Rpo_s*w_s + prob_det_Rpo_d1*w_d1 + \
                    prob_det_Rpo_d2*w_d2*frac_d2_sunlike
    axs[5].plot(R_pas, summed, label='total', c='black')

    txt = '$N_s$: {:d}\n$N_d$ {:d}\nfrac d2, sunlike {:.3f}'.format(
            N_s, N_d1, frac_d2_sunlike) + \
          '\n$w_i = f_{i,\mathrm{sunlike}}\cdot N_i/N_{\mathrm{tot}}$'
    axs[5].text(0.03, 0.97, txt, verticalalignment='top',
            horizontalalignment='left', transform=axs[5].transAxes,
            fontsize='xx-small')
    axs[5].legend(loc='upper right', fontsize='x-small')
    axs[5].set_ylabel('$N_{\mathrm{det}}(R_p^{\mathrm{obs}}) / N_{\mathrm{tot}}$')


    axs[6].plot(R_pt, prob_Rpts, label='true', c='gray')

    norm = trapz(summed/f_det, R_pas)
    inferred = (summed/f_det)/norm
    axs[6].plot(R_pas, inferred,
            label='inferred = total/f_det\n(normalized. matters!!)', c='black')

    txt = 'inference ignoring binaries'
    axs[6].text(0.03, 0.97, txt, verticalalignment='top',
            horizontalalignment='left', transform=axs[6].transAxes,
            fontsize='xx-small')
    axs[6].legend(loc='upper right', fontsize='xx-small')
    axs[6].set_ylabel(r'$\rho_{R_p^{\mathrm{true}}}, '
                      r'\rho_{R_p^{\mathrm{inferred}}}$')


    true = prob_Rpts
    inds = (true > 0) & (R_pas > 0.1) # odd numerical thing w/ lognormal
    axs[7].plot(R_pas[inds], (inferred[inds]-true[inds])/true[inds],
                label='(inferred-true)/true', c='black')
    axs[7].hlines(0, min(R_pas), max(R_pas), colors='black', alpha=0.2, zorder=-1)

    txt = 'inference ignoring binaries'
    axs[7].text(0.03, 0.97, txt, verticalalignment='top',
            horizontalalignment='left', transform=axs[7].transAxes,
            fontsize='xx-small')
    axs[7].legend(loc='upper right', fontsize='x-small')
    axs[7].set_ylabel(r'relative error on pdfs ($\rho$)'
            '\nNB I\'m treating them as occ rates!',
            fontsize=5)



    N_d = N_d1
    axs[8].plot(R_pt, prob_Rpts*N_s,
            label='true s')
    axs[8].plot(R_pt, prob_Rpts*N_d,
            label='true d1')
    axs[8].plot(R_pt, prob_Rpts*frac_d2_sunlike*N_d,
            label='true d2')

    axs[8].plot(R_pt, prob_Rpts*(N_s + (1+frac_d2_sunlike)*N_d),
            label=r'$N_p^{\mathrm{true}} = \rho_{R_p^{\mathrm{true}}}'
                   '\cdot (N_s + (1+f_{d2}^{\mathrm{sunlike}})N_d)$',
            c='gray')
    axs[8].plot(R_pas, inferred*(N_s + N_d),
            label=r'$N_p^{\mathrm{inferred}} = \rho_{R_p^{\mathrm{inferred}}}'
                   '\cdot (N_s + N_d)$',
            c='black')

    axs[8].legend(loc='upper right', fontsize=4)
    axs[8].set_ylabel(r'$N^p_{{\mathrm{true}}}, '
                      r'N^p_{{\mathrm{inferred}}}$')


    tru = prob_Rpts*(N_s + (1+frac_d2_sunlike)*N_d)
    inferre = inferred*(N_s + N_d)
    axs[9].plot(R_pas, (inferre-tru)/tru,
                label='(inferred-true)/true', c='black')
    axs[9].hlines(0, min(R_pas), max(R_pas), colors='black', alpha=0.2, zorder=-1)

    axs[9].legend(loc='upper right', fontsize='x-small')
    axs[9].set_ylabel(r'relative error on N planets'
                       '\nyou think exist ($N^p$)',
            fontsize='x-small')



    axs[9].set_xlabel('$R_p [R_\oplus]$ (observed or true)')

    for ax in [axs[7], axs[9]]:
        if max(ax.get_ylim()) > 1:
            if ax == axs[9]: #fml
                ax.set_ylim(-1, 1)
            else:
                ax.set_ylim(min(ax.get_ylim()), 1)

    for ax in axs:
        ax.tick_params(which='both', direction='in', zorder=0)

    f.tight_layout(h_pad=0)
    f.savefig('prob_Rpobs/prob_Rpobs_{savstr}.pdf'.format(savstr=sstr),
               dpi=250, bbox_inches='tight')






def case_1(γ_R, R_pt, R_pas, R_pu=20, R_pl=10, sstr='uniform'):
    ######################################
    # CASE 1: UNIFORM INPUT DISTRIBUTION #
    ######################################
    ρ_γR = prob_gammaR(γ_R)

    prob_Rpos_d1, prob_Rpos_d2 = [], []

    for ix, R_pa in enumerate(R_pas):

        if ix % 500 == 0:
            print('{:d}/{:d}'.format(ix, len(R_pas)))

        ρ_Rpt_d1 = prob_Rpt_uniform(
                        R_pa * (1+γ_R)**(1/2), R_pu, R_pl )
        ρ_Rpt_d2 = prob_Rpt_uniform(
                        R_pa * (1+γ_R**(-1))**(1/2)*γ_R**(1/α), R_pu, R_pl )

        ρ_Rpo_at_Rpo_d1 = trapz( ρ_γR * ρ_Rpt_d1, γ_R  )
        ρ_Rpo_at_Rpo_d2 = trapz( ρ_γR * ρ_Rpt_d2, γ_R  )

        prob_Rpos_d1.append(ρ_Rpo_at_Rpo_d1)
        prob_Rpos_d2.append(ρ_Rpo_at_Rpo_d2)

    prob_Rpts = prob_Rpt_uniform(R_pt, R_pu, R_pl)

    _make_Rpobs_plot(R_pt, prob_Rpts, R_pas, prob_Rpos_d1, prob_Rpos_d2,
            sstr=sstr)
    print('did case 1 (uniform)')


def case_2(γ_R, R_pt, R_pas):
    ##########################################
    # CASE 2: LOG-UNIFORM INPUT DISTRIBUTION #
    ##########################################
    ρ_γR = prob_gammaR(γ_R)

    R_pu, R_pl = 20, 10

    prob_Rpos_d1, prob_Rpos_d2 = [], []

    for ix, R_pa in enumerate(R_pas):

        if ix % 500 == 0:
            print('{:d}/{:d}'.format(ix, len(R_pas)))

        ρ_Rpt_d1 = prob_Rpt_loguniform( R_pa * (1+γ_R)**(1/2), R_pu, R_pl )
        ρ_Rpt_d2 = prob_Rpt_loguniform( R_pa * (1+γ_R**(-1))**(1/2)*γ_R**(1/α), R_pu, R_pl )

        ρ_Rpo_at_Rpo_d1 = trapz( ρ_γR * ρ_Rpt_d1, γ_R  )
        ρ_Rpo_at_Rpo_d2 = trapz( ρ_γR * ρ_Rpt_d2, γ_R  )

        prob_Rpos_d1.append(ρ_Rpo_at_Rpo_d1)
        prob_Rpos_d2.append(ρ_Rpo_at_Rpo_d2)

    prob_Rpts = prob_Rpt_loguniform(R_pt, R_pu, R_pl)

    _make_Rpobs_plot(R_pt, prob_Rpts, R_pas, prob_Rpos_d1, prob_Rpos_d2, sstr='loguniform')
    print('did case 2 (loguniform)')


def case_3(γ_R, R_pt, R_pas):
    ########################################
    # CASE 3: LOG-NORMAL INPUT DISTIBUTION #
    ########################################
    μ, σ = 3, 1.2 # mean of log, stddev of log

    ρ_γR = prob_gammaR(γ_R)

    prob_Rpos_d1, prob_Rpos_d2 = [], []

    for ix, R_pa in enumerate(R_pas):

        if ix % 500 == 0:
            print('{:d}/{:d}'.format(ix, len(R_pas)))

        ρ_Rpt_d1 = prob_Rpt_lognormal( R_pa * (1+γ_R)**(1/2), μ, σ)
        ρ_Rpt_d2 = prob_Rpt_lognormal( R_pa * (1+γ_R**(-1))**(1/2)*γ_R**(1/α), μ, σ)

        ρ_Rpo_at_Rpo_d1 = trapz( ρ_γR * ρ_Rpt_d1, γ_R  )
        ρ_Rpo_at_Rpo_d2 = trapz( ρ_γR * ρ_Rpt_d2, γ_R  )

        prob_Rpos_d1.append(ρ_Rpo_at_Rpo_d1)
        prob_Rpos_d2.append(ρ_Rpo_at_Rpo_d2)

    prob_Rpts = prob_Rpt_lognormal(R_pt, μ, σ)

    _make_Rpobs_plot(R_pt, prob_Rpts, R_pas, prob_Rpos_d1, prob_Rpos_d2, sstr='lognormal')
    print('did case 3 (lognormal)')


def case_4(γ_R, R_pt, R_pas):
    ##################################################
    # CASE 4: TRUNCATED LOG-NORMAL INPUT DISTIBUTION #
    ##################################################
    R_pl, R_pu = 2, 20 #truncation
    μ, σ = 3, 1.2 # mean of log, stddev of log

    ρ_γR = prob_gammaR(γ_R)

    prob_Rpos_d1, prob_Rpos_d2 = [], []

    for ix, R_pa in enumerate(R_pas):

        if ix % 500 == 0:
            print('{:d}/{:d}'.format(ix, len(R_pas)))

        ρ_Rpt_d1 = prob_Rpt_trunclognormal( R_pa * (1+γ_R)**(1/2), R_pl, R_pu,
                                            μ, σ)
        ρ_Rpt_d2 = prob_Rpt_trunclognormal( R_pa * (1+γ_R**(-1))**(1/2)*γ_R**(1/α), R_pl,
                                            R_pu, μ, σ)

        ρ_Rpo_at_Rpo_d1 = trapz( ρ_γR * ρ_Rpt_d1, γ_R  )
        ρ_Rpo_at_Rpo_d2 = trapz( ρ_γR * ρ_Rpt_d2, γ_R  )

        prob_Rpos_d1.append(ρ_Rpo_at_Rpo_d1)
        prob_Rpos_d2.append(ρ_Rpo_at_Rpo_d2)

    prob_Rpts = prob_Rpt_trunclognormal(R_pt, R_pl, R_pu, μ, σ)

    _make_Rpobs_plot(R_pt, prob_Rpts, R_pas, prob_Rpos_d1, prob_Rpos_d2, sstr='trunclognormal')
    print('did case 4 (trunclognormal)')


def case_5(γ_R, R_pt, R_pas):
    ########################################################
    # CASE 5: HOWARD+12 CLAIMED P<50D POWER LAW FOR GIANTS #
    ########################################################
    # inspiration also taken from Wang+ 2015.

    R_pl, R_pu = 5, 22 #truncation

    ρ_γR = prob_gammaR(γ_R)

    prob_Rpos_d1, prob_Rpos_d2 = [], []

    for ix, R_pa in enumerate(R_pas):

        if ix % 500 == 0:
            print('{:d}/{:d}'.format(ix, len(R_pas)))

        ρ_Rpt_d1 = prob_Rpt_Howardpower( R_pa * (1+γ_R)**(1/2), R_pl, R_pu )
        ρ_Rpt_d2 = prob_Rpt_Howardpower( R_pa * (1+γ_R**(-1))**(1/2)*γ_R**(1/α),
                                                                R_pl, R_pu )

        ρ_Rpo_at_Rpo_d1 = trapz( ρ_γR * ρ_Rpt_d1, γ_R  )
        ρ_Rpo_at_Rpo_d2 = trapz( ρ_γR * ρ_Rpt_d2, γ_R  )

        prob_Rpos_d1.append(ρ_Rpo_at_Rpo_d1)
        prob_Rpos_d2.append(ρ_Rpo_at_Rpo_d2)

    prob_Rpts = prob_Rpt_Howardpower(R_pt, R_pl, R_pu)

    _make_Rpobs_plot(R_pt, prob_Rpts, R_pas, prob_Rpos_d1, prob_Rpos_d2,
            sstr='howardpower', howardpower=True)
    print('did case 5 (howardpower)')



if __name__ == '__main__':

    γ_R = np.logspace(-4, 0, num=int(4e3))
    R_pt = np.linspace(0, 30, int(3e3)) # Re units
    R_pas = np.linspace(0, 30, int(3e3)) # Re units

    case_1(γ_R, R_pt, R_pas, R_pu=15, R_pl=14.99, sstr='uniform_narrow')
    case_1(γ_R, R_pt, R_pas)
    case_2(γ_R, R_pt, R_pas)
    case_3(γ_R, R_pt, R_pas)
    case_4(γ_R, R_pt, R_pas)
    case_5(γ_R, R_pt, R_pas)

    #TODO: bin it. 
