'''
Bastien et al (2014, ApJ 788:L9) showed that for a Kepmag<13 sample, the NASA
Exoplanet Archive derived stellar parameters were biased to have smaller logg's
than those measured with their flicker method (cf Bastien+ 2013).

This matters for the derived planetary radius distribution -- in Bastien et al
(2014), it systematically shifted the derived planetary properties for this
sub-population by a factor of ~1.3x

However, it also probably matters for the HJ rate of this subpopulation. This
is because the HJ occ rate is claimed to be lower for subgiants than MS stars.
This claim probably first came from Hatzes et al. (2003), but was substantiated
by the California group adding "retired A stars" to their target list
("retired" because they were thought to be formerly F stars by mass, but now A
stars and evolved -- though there has been lots of confusion about this). This
work was associated with publications by Johnson et al. (2007, 2008, 2010c,
2010b, 2011b, 2011a).  Robinson et al.  (2007), Fischer et al.  (2007), and
Bowler et al. (2010) also contributed to this observational question.

Regardless, Schlaufman & Winn (2013) looked at this, came up with an
independent kinematic argument for why HJ occ rates are probably lower around
subgiants, and then argued that tidal evolution is probably the best
explanation (though no physical models convincingly produce the rates).

Regardless, ASSUMING we believe HJ occ rates are lower about subgiants,
subgiant contamination of the Kepler sample then matters. (For both the radius
distribution, and the occurrence rates of any close-in planets).

Wang et al (2015 ApJ 799:229) noticed this in the context of the
barely-statistically significant difference between the HJ occ rates measured
by transit surveys (mostly Kepler, Howard et al 2012, & Fressin et al 2013),
Γ_HJ,Kepler ~= 0.5%, and RV surveys (mostly CKS, Wright et al 2012, but also
the unpublished Mayor et al 2011), Γ_HJ,Kepler ~= 1%.

Wang et al (2015) did the following. Start with 1e5 planets. Give them a power
law radius distribution of -2.9 (Howard et al 2012), from 5Re to 22Re. (Note
that Howard+ 2012's lower cutoff was 8Re, Fressin+ 2013's was 6Re). Wang+
(2015) went down to 5Re because this produces a 0.1Mj mass when using Lissauer+
2011's mass radius relationship (which is calibrated how?).

(Regardless, using the radius distribution is something I should do!)

Then take Kepler stellar properties from NASA Exoplanet Archive (NEA). Cut in
Teff (4100-6100K), cut in logg (4-4.9) to match Howard+ 2012. Get 109335
""solar-type"" dwarf stars.

I THINK (they never explicitly said): ASSIGN EACH PLANET TO A STAR, ASSUMING
IT ORBITS THE PRIMARY.
(I think this assignment is random, but randomized over Monte Carlo trials)

They then assumed the following for the binary population of these ~110k Kepler
target stars:

* the multiplicity fraction for a_bin < 1000AU from Wang+ 2014a's rate. For
    a_bin > 1000AU from D&M 1991's rate a>1000AU.

    (Probably should have used Raghavan+ 2010 for the latter).

    (More important, this is the MF for the stellar population ASSUMING EVERY
    STAR HOSTS A HJ. This is completely artificial -- in reality KIC target
    stars were selected without regard for binarity, b/c their binary
    properties were not known. So W+15's synthetic population underestimates
    the MF vs the KIC).

* take prob(q) to be a normal distribution w/ μ=0.23, σ=0.42 (DM 1991 Fig 10).

    (This ignores the magnitude limit imposing a bias towards large q -- so
    they will have a binary population w/ smaller q)

    (N.B. also that Raghavan+10's Fig 16, Sec 5.3.5, contradict DM91’s
    conclusions, which showed no preference for like-mass pairs, and a mass
    ratio distribution that could be fit this way. Basically -- the data
    support a different mass ratio distribution from what Wang+15 took -- one
    with more q>0.5 binaries)

* assume the primary mass from KIC. Then you get secondary mass from q. Convert
    the secondary masses to radii with Feiden & Chaboyer (2012)'s relation.
    Convert from mass to stellar flux in the Kepler band using Kraus &
    Hillenbrand (2007)'s prescription.

    (N.B. this means that yes, the mass-radius relation for the secondary
    companions will be different from for the KIC-derived primaries.
    Introducing funky biases.)

With these assumptions, you're set.

Compute the photometric dilution from (a) gravitationally bound multiple star
systems and (b) due to optical doubles and multiples using TRILEGAL.

For subgiant contamination, from NEA statistics 27% of Kepler stars are
subgiants. But Bastien+ 2014 showed 48% of the bright stellar sample were
subgiants (with larger radii!).

So then W+2015 selected 29% of the Kepler "dwarf" stars (actually subgiants),
and artificially inflated their radii by 25%.

Then compute the SNRs, and you're done.

Under all these assumptions, W+15 found 3% of their HJs weren't correctly ID'd
b/c of dilution (grav bound + optical), and ~10% of their HJ sample was
mis-ID'd b/c of subgiant contamination.

But here we ask:
    * did Wang+15's assumption inre: Bastien+ 2014's "50% of them are
    subgiants!" make any sense, given that B+14's sample had a bright mag cut?

i.e.:
look at the pdf of logg for simulated (GIC) r<13, and then r<15, or r<16.

PREDICT:
r<15 will have relatively FEWER 3.5<logg<4.1 stars than r<13. In other
words, the Malmquist bias towards subgiants should be less strong as you go
deeper.

THE RESULT:
in r<13, 14.6% are "subgiants"
in r<15, 8.4% are "subgiants"
in the full GIC (which qualitatively followed the exact same selection
    procedure as Batalha+10), 3.8% are "subgiants"

So yeah, Wang+15 are going to overstate the importance of this stellar radius
misestimate by like a factor of 14/4 ~= 3 at the population level.

NB. their formulation of Eqs 4 and 5 is still cool. So is their Table 1.

'''

import numpy as np, matplotlib.pyplot as plt
import pandas as pd
import pickle

# Took all Galaxia stars within 7.5 degrees of the center of Kepler's
# field, and only took stars with r<17.
# Then added binaries. (But never changed logg properties!)
# Then applied Batalha+ 2010's selection procedure (with binaries).
# So we can recover the original logg distribution by concatinating the singles
# and binaries.
gal = pickle.load(open('/home/luke/local/selected_GIC_systems_kepler_analog.p', 'rb'))
singles = gal['singles']
doubles = gal['binaries']

loggs = np.array(pd.concat((singles['grav'], doubles['grav'])))


f, ax = plt.subplots(figsize=(4,4))

bins = np.arange(3.5, 5+0.1, 0.1)

N_subgiant = len(loggs[(loggs<4.1) & (loggs>3.5)])/len(loggs)
labstr = '. {:.1f}% are "subgiants".'.format(N_subgiant*100)
hist, bin_edges = np.histogram(loggs, bins=bins, normed=True)
ax.step(bin_edges[:-1], hist, where='post', label='full GIC'+labstr, lw=1)

inds = np.concatenate((np.array(singles['apparent_r'] < 15),
                       np.array(doubles['sys_apparent_r'] < 15)))
N_subgiant = len(loggs[inds][(loggs[inds]<4.1) & (loggs[inds]>3.5)])/len(loggs[inds])
labstr = '. {:.1f}% are "subgiants".'.format(N_subgiant*100)
hist, bin_edges = np.histogram(loggs[inds], bins=bins, normed=True)
ax.step(bin_edges[:-1], hist, where='post', label='$r<15$'+labstr, lw=1)

inds = np.concatenate((np.array(singles['apparent_r'] < 13),
                       np.array(doubles['sys_apparent_r'] < 13)))
N_subgiant = len(loggs[inds][(loggs[inds]<4.1) & (loggs[inds]>3.5)])/len(loggs[inds])
labstr = '. {:.1f}% are "subgiants".'.format(N_subgiant*100)
hist, bin_edges = np.histogram(loggs[inds], bins=bins, normed=True)
ax.step(bin_edges[:-1], hist, where='post', label='$r<13$'+labstr, lw=1)

ymax = max(ax.get_ylim())
ax.vlines([3.5, 4.1], 0, ymax, colors='black', zorder=-1, lw=1, alpha=0.5,
          linestyles='--')

ax.set_xlim([3.4, 5])
ax.set_ylim([0, ymax])

ax.set_xlabel('log(g) [cgs]')
ax.set_ylabel('prob (indept normalizn)')

ax.legend(loc='upper left', fontsize='xx-small')

ax.set_title('Q: does the Malquist bias towards subgiants get less important'
             '\nwhen you go to deeper than Wang+ 2015\'s $r<13$? A: yes.',
             fontsize='xx-small')

f.tight_layout()
f.savefig('logg_subgiant_bias_check.pdf', bbox_inches='tight')
