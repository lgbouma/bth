import pickle

global DATADIR
DATADIR = '/home/luke/local/GalaxiaData/kepler_synthesis/'

def convert_galaxia_output():
    '''
    Convert from ebf to pickle (latter is more generally usable, e.g. not in
    python 2.X)
    '''
    try:
        import ebf
    except ImportError:
        print('ImportError: ebf requires python 2.X, and manual installation.')

    # Keys described http://galaxia.sourceforge.net/Galaxia3pub.html#mozTocId124318
    # ['rad', 'exbv_solar', 'teff', 'sdss_g', 'mag2', 'mag1', 'mag0', 'sdss_r',
    # 'sdss_u', 'satid', 'vx', 'vy', 'vz', 'sdss_z', 'mtip', 'log', 'pz', 'px',
    # 'py', 'feh', 'exbv_schlegel', 'sdss_i', 'lum', 'exbv_schlegel_inf', 'mact',
    # 'glon', 'popid', 'glat', 'alpha', 'center', 'partid', 'age', 'grav', 'smass',
    # 'fieldid']

    datapath = DATADIR+'keplersynthesis.ebf'

    data = ebf.read(datapath)

    outd = {}

    for key in data.keys():

        outd[key] = data[key]

    pickle.dump(outd, open(DATADIR+'keplersynthesis.p', 'wb'))

if __name__ == '__main__':

    convert_galaxia_output()
