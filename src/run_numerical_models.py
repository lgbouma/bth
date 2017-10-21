'''
Wrapper to `numerical_models.py`.

Useful for running Model #3 with varying Λ_2 in parallel.
'''

import numpy as np
from multiprocessing import Pool

from numerical_models import numerical_transit_survey

if __name__ == '__main__':

    model_number = 3
    quickrun = False
    Λ_2_arr = np.arange(0., 0.5+0.05, 0.05)

    paramlist = [(quickrun, model_number, Λ_2) for Λ_2 in Λ_2_arr]

    with Pool(6) as p:
        p.starmap(
            numerical_transit_survey,
            paramlist
            )
