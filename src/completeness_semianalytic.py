import pickle
from scipy.integrate import trapz
import matplotlib.pyplot as plt
import numpy as np

def primary_star_completeness():
    '''
    Find prob(N_d * f_d1,c)
    and < N_d * f_d1,c >.

    These are the relevant terms for the observed occ rate.
    '''
    alpha = 3.5
    f_d1c_upper = (1+0.1**alpha)**(-3)
    f_d1c_lower = 2**(-3)

    f_d1c = np.linspace(f_d1c_lower, f_d1c_upper, num=int(1e4))

    # N_s is a survey dependent number. The following verifies that it doesn't
    # matter. (We just need it to implement semi-analytics)
    for N_s in np.logspace(3,6,7):

        N_s = int(N_s)
        R = N_s**(-2/3)
        BF = 0.44 # star-type dependent binary fraction
        n_d_by_n_s = BF / (1-BF)
        N_d_lower = (0.1**(3.5) + 1)**(1/2) * N_s * n_d_by_n_s
        N_d_upper = (2)**(1/2) * N_s * n_d_by_n_s


        # Numerically integrate 17/09/25.6 expression
        curlyF_min = f_d1c_lower * N_d_lower
        curlyF_max = f_d1c_upper * N_d_upper

        F_grid = np.linspace(curlyF_min, curlyF_max, num=int(5e3))

        prob_product = []

        # calculate prob_maglimited(N_d * f_{d1,c})
        for ind, F in enumerate(F_grid):

            if ind%500 == 0:
                print(repr(ind) + '/' + repr(len(F_grid)))

            integrand = f_d1c**(-1/2) * \
                        (f_d1c**(-1/3) - 1)**(-5/7) * \
                        np.abs(f_d1c**(-4/3)) * \
                        (F/f_d1c)**(2/3) * \
                        (F**(2/3)*f_d1c**(-2/3)*R - 1)**(-5/7)

            val = trapz(integrand, f_d1c)

            prob_product.append(val)

        prob_arr = np.nan_to_num(prob_product)

        norm = trapz(prob_arr, F_grid)
        print('normalization is: {:.8f}'.format(norm))

        prob_arr /= norm

        savstr = repr(N_s)
        pickle.dump(prob_arr, open('prob_curly_F_checks/prob_curlyF_arr_{:s}.p'.format(savstr), 'wb'))

        f,ax=plt.subplots()
        ax.plot(F_grid, prob_arr)

        ax.set(
        xlabel='$N_d f_{d1,c}$',
        ylabel='prob',
        yscale='log'
        )

        f.savefig('prob_curly_F_checks/prob_curlyF_{:s}.pdf'.format(savstr),
                bbox_inches='tight')


        # calculate <N_d * f_{d1,c}>
        #   = integral of F * prob(F)

        # also express in terms of N_s
        expected_val = trapz(F_grid * prob_arr, F_grid)

        expected_val_by_N_s = expected_val / N_s

        outstr = \
        '''

        <N_d f_d1c> = {:.2f}

        <N_d f_d1c>/N_s = {:.5f}

        '''.format(expected_val, expected_val_by_N_s)

        print(outstr)
        with open('prob_curly_F_checks/expected_val_{:s}.out'.format(savstr),'w') as f:
            f.write(outstr)


def secondary_star_completeness():
    '''
    Find prob(N_d * f_d2,c * f_d2,g)
    and < N_d * f_d2,c * f_d2,g >.

    These are the relevant terms for the observed occ rate.
    '''
    alpha = 3.5

    f_d2c_lower = (1+0.1**(-alpha))**(-3)
    f_d2c_upper = 2**(-3)

    f_d2c = np.logspace(np.log10(f_d2c_lower), np.log10(f_d2c_upper),
                        num=int(2e3))

    # N_s is a survey dependent number. f_sg is a planet/star-dependent number
    # (what is the star size being observed? the period? etc). The following
    # verifies that neither matters. (We just need them to implement
    # semi-analytics).
    for f_sg in np.logspace(-3,-1,5):
        for N_s in np.logspace(6,8,3):

            f_d2g_lower = 0.1**(2/3) * f_sg
            f_d2g_upper = f_sg
            f_d2g = np.logspace(np.log10(f_d2g_lower), np.log10(f_d2g_upper),
                                num=int(1.1e3))

            N_s = int(N_s)
            R = N_s**(-2/3)
            BF = 0.44 # star-type dependent binary fraction
            n_d_by_n_s = BF / (1-BF)
            N_d_lower = (0.1**(3.5) + 1)**(1/2) * N_s * n_d_by_n_s
            N_d_upper = (2)**(1/2) * N_s * n_d_by_n_s

            # Numerically integrate 17/10/02.2 expression, and
            # calculate prob_maglimited(N_d * f_{d2,c} * f_{d2,g})
            curlyQ_min = f_d2g_lower * f_d2c_lower * N_d_lower
            curlyQ_max = f_d2g_upper * f_d2c_upper * N_d_upper

            Q_grid = np.logspace(np.log10(curlyQ_min), np.log10(curlyQ_max),
                                 num=int(2e3))

            prob_product = []

            for ind, Q in enumerate(Q_grid):

                if ind%50 == 0:
                    print(repr(ind) + '/' + repr(len(Q_grid)))

                # shape(inner_integrand): shape(f_d2c) * shape(f_d2g)
                inner_integrand = f_d2c[:,None]**(-5/21) * \
                                  (1 - f_d2c[:,None]**(1/3))**(-11/14) * \
                                  np.abs( (1-f_d2c[:,None]**(1/3))**(-2) * f_d2c[:,None]**(-2/3) ) * \
                                  ( Q / (f_d2c[:,None]*f_d2g[None,:] ) )**(2/3) * \
                                  (Q**(2/3)*f_d2c[:,None]**(-2/3)*f_d2g[None,:]**(-2/3)*R - 1)**(-5/7)

                int1 = trapz(inner_integrand, f_d2c, axis=0)

                val = trapz(int1 * \
                            (1 + (f_d2g/f_sg)**(21/4))**(3/2) * \
                            f_d2g**(1/2)* f_sg**(-3/2),
                            f_d2g)

                prob_product.append(val)

            prob_arr = np.nan_to_num(prob_product)

            norm = trapz(prob_arr, Q_grid)
            print('normalization is: {:.8f}'.format(norm))

            prob_arr /= norm

            savstr = repr(N_s) + '_' + repr(f_sg)
            pickle.dump(prob_arr, open('prob_curly_Q_checks/prob_curlyQ_arr_{:s}.p'.format(savstr), 'wb'))

            f,ax=plt.subplots()
            ax.plot(Q_grid, prob_arr)

            ax.set(
            xlabel='$N_d f_{d2,c} f_{d2,g}$',
            ylabel='prob',
            yscale='log'
            )

            f.savefig('prob_curly_Q_checks/prob_curlyQ_{:s}.pdf'.format(savstr),
                    bbox_inches='tight')


            # calculate <N_d * f_{d2,c} * f_{d2,g}>
            #   = integral of Q * prob(Q)

            # also express in terms of N_s
            expected_val = trapz(Q_grid * prob_arr, Q_grid)

            expected_val_by_Ns_times_fsg = expected_val / (N_s*f_sg)

            outstr = \
            '''

            <N_d f_d2c f_d2g> = {:.2f}

            <N_d f_d2c f_d2g>/(N_s*f_sg) = {:.5f}

            '''.format(expected_val, expected_val_by_Ns_times_fsg)

            print(outstr)
            with open('prob_curly_Q_checks/expected_val_{:s}.out'.format(savstr),'w') as f:
                f.write(outstr)




if __name__ == '__main__':

    #primary_star_completeness()
    secondary_star_completeness()
