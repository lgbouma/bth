import numpy as np
import matplotlib.pyplot as plt

γ_R = np.arange(0, 1, 1e-3)

dil_pri = (1 + γ_R)**(-1)
dil_sec = γ_R * (1 + γ_R)**(-1)

Rpobs_by_Rptrue_pri = np.sqrt(dil_pri)
Rpobs_by_Rptrue_sec = np.sqrt(dil_sec)

f,axs = plt.subplots(nrows=2, ncols=1, figsize=(4,6), sharex=True)

# Top plot: dilution vs gamma_R
ax = axs[0]
ax.plot(γ_R, dil_pri, 'k-', label='planet orbits primary')
ax.plot(γ_R, dil_sec, 'b-', label='planet orbits secondary')
ax.hlines(0.5, 0, 1, colors='k', linestyles='dashed', alpha=0.5)

ax.legend(loc='best', fontsize='xx-small', frameon=False)

ax.set_xlim([0,1])
ax.set_ylim([0,1])

ax.set_ylabel('$\mathcal{D}$')

# Bottom plot: radius ratio vs gamma_R
ax = axs[1]
ax.plot(γ_R, Rpobs_by_Rptrue_pri, 'k-')
ax.plot(γ_R, Rpobs_by_Rptrue_sec, 'b-')
ax.hlines(1/2**(1/2), 0, 1, colors='k', linestyles='dashed', alpha=0.5)

ax.set_xlim([0,1])
ax.set_ylim([0,1])

ax.set_xlabel('$\gamma_R = L_2/L_1$')
ax.set_ylabel('$R_p^\mathrm{obs} / R_p^\mathrm{true} = \sqrt{\mathcal{D}}$')

f.tight_layout()

f.savefig('observed_radii_vs_gammaR.pdf', dpi=300, bbox_inches='tight')
