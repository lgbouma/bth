----------
DESCRIPTION

This program simulates the following toy surveys:

MODEL #1: Fixed stars, fixed planets, twin binaries.

MODEL #2: Fixed planets and primaries, varying secondaries.

MODEL #3: Fixed primaries, varying planets and secondaries.

In the "nominal" models, all classes of errors are included when calculating
what the observer infers. These errors are

1. misinterpretation of planetary radii (b/c of transit depth dillution; also
   b/c of wrongly assumed host star radii),
2. incorrectly assumed detection efficiency (transit probability and
   fraction of selected stars that are searchable),
3. incorrectly assumed number of selected stars.

Apart from the "nominal" models, we also consider "error case 1", "error case
2", and "error case 3". They are defined s.t.:

    error_case_1: errors 1 & 2 are corrected, leaving error #3
    error_case_2: errors 1 & 3 are corrected, leaving error #2
    error_case_3: errors 2 & 3 are corrected, leaving error #1.

----------
The simulation works as follows.

First, the user specifies their inputs: the binary fraction, the model class
(#1,2,3), various exponents (α,β,γ,δ), and the true occurrence rates, `Λ_i`.

The population of selected stars is constructed as follows.  First, we note
that each selected star has a `star_type` (single, primary, secondary), a
binary mass ratio, if applicable, and the property of whether it is
"searchable".

The absolute number of stars is arbitrary; we take 10^6 single stars
throughout. The number of binaries is calculated according to analytic
formulae. The binary mass ratios are, when applicable, samples from the
appropriate magnitude-limited distribution (given α and β).

Whether a star is "searchable" depends entirely on its "completeness" fraction.
By "completeness", we mean the ratio of the actual number of searchable stars
to the assumed number of searchable stars (for a given planet size, period,
etc.).  Assuming homogeneuously distributed stars, this is equivalent to the
ratio of the searchable to selected volumes.  In our model, this volume ratio
is a function of only the binary mass ratio.

The procedure for assigning planets is then as follows:
    * each selected star of type i gets a planet at rate `Λ_i`
    * the radii of planets are assigned independently of any host system
    property, as sampled from p_r(r) ~ r^δ for Model #3 or a δ function for
    Models #1 and #2.
    * a planet is detected when a) it transits, and and b) its host star is
    searchable.

The probability of transiting single stars in our model is assumed to be known,
and so it can be mostly corrected by the observer attempting to infer an
occurrence rate. The only bias is for secondaries, which can be smaller. This
effect is including when computing the transit probability.

For detected planets, apparent radii are computed according to analytic
formulae that account for both dilution and the misclassification of stellar
radii. We assume that the observer thinks they observe the primary.

The rates are then computed in bins of true planet radius and apparent planet
radius.

In a given radius bin, the true rate is found by counting the number of planets
that exist around selected stars of all types (singles, primaries,
secondaries), and dividing by the total number of these stars.

The apparent rates are found by counting the number of detected planets that
were found in an apparent radius bin, dividing by the geometric transit
probability for single stars, and dividing by the apparent total number of
stars.

----------
USAGE

Change parameters in the "inputs" section below. Run with Python3.X

----------
The following tests are implemented:

* γ == 0
* there are as many selected primaries as secondaries
* stars in the same system get the same mass ratio
* all mass ratios are > 0
* only things with detected planets have an apparent radius
* the sum of the true distribution's histogrammed rate densities is the true
  average occurrence rate, to three decimals

Model #1:
* completeness fractions are very close to 1/8
* numerical value of `X_Γ` at `r_p` matches analytic prediction

Model #2:
* numerical value of `X_Γ` at `r_p` matches analytic prediction

Models #2 and #3:
* primaries are all more complete than 1/8
* secondaries are all less complete than 1/8
