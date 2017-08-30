import numpy as np
import matplotlib.pyplot as plt

Nd_by_Ns = 0.59

ξ = np.arange(0, Nd_by_Ns, 1e-3)

f,ax = plt.subplots(nrows=1, ncols=1, figsize=(4,4))

χ = 2 # greek chi
β = 0.59
Δprime = abs(
        1 - ( (1 + χ*β)/(1+ξ) )*( (1+β)/(1+2*β)  )
        )

ax.plot(ξ, Δprime, 'k-', label=r'$\chi = 2, \beta = 0.59$')

χ = 1 # greek chi
β = 0.59
Δprime = abs(
        1 - ( (1 + χ*β)/(1+ξ) )*( (1+β)/(1+2*β)  )
        )

ax.plot(ξ, Δprime, 'b-', label=r'$\chi = 1, \beta = 0.59$')


χ = 1.5 # greek chi
β = 0.59
Δprime = abs(
        1 - ( (1 + χ*β)/(1+ξ) )*( (1+β)/(1+2*β)  )
        )

ax.plot(ξ, Δprime, 'g-', label=r'$\chi = 1.5, \beta = 0.59$')


ax.legend(loc='best', fontsize='small', frameon=False)

ax.set_xlim([0,β])
ax.set_ylim([0,0.6])

ax.set_ylabel("$\Delta'$ (relative error btwn A' and D)")
ax.set_xlabel(r"$\xi$ (max is upper bound of $\beta$)")

f.tight_layout()

f.savefig('deltaprime_vs_xi.pdf', dpi=300, bbox_inches='tight')
