import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

catalog_id = pd.read_csv('/ocean/projects/phy210068p/hsu1/Ananke_datasets_training/stars_accretion_history_m12i_res7100_v2.csv',
                     usecols = ['id_stars'])

path = '/ocean/projects/phy210068p/hsu1/Ananke_datasets_training/AnankeDR3_data_reduced_m12i_lsr012.hdf5'

with h5py.File(path, 'r') as f:
    ananke_id = f['parentid'][:]

accretion_id = np.isin(ananke_id, catalog_id)
accretion = np.int32(accretion_id)
is_accreted = accretion

with h5py.File(path, 'a') as f:
#     del f['is_accreted']
    f.create_dataset('is_accreted', data=accretion)


import astropy.units as u
import astropy.coordinates as coord
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import h5py

import agama

import utilities as ut
import gizmo_analysis as gizmo

# Set unit of Agama to (mass, length, velocity) = (1 Msun, 1 kpc, 1 km/s)
agama.setUnits(mass=1, length=1, velocity=1)

plt.style.use('seaborn-colorblind')
mpl.rcParams.update({
    'font.size': 18,
    'figure.figsize': (8, 6),
    'figure.facecolor': 'w',
    'axes.linewidth': 2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# Read in DM potential and bar potential
pxr_DM = agama.Potential(file=
                         "/ocean/projects/phy210068p/hsu1/Fire_simulation/m12i_res7100/600.dark.axi_4.coef_mul_DR")
pxr_bar = agama.Potential(file=
                          "/ocean/projects/phy210068p/hsu1/Fire_simulation/m12i_res7100/600.bar.axi_4.coef_cylsp_DR")

# Combine bar and DM potential
potential = agama.Potential(pxr_DM, pxr_bar)

with h5py.File(path, 'r') as f:
    ra = f['ra'][:]
    dec = f['dec'][:]
    parallax = f['parallax'][:]
    pmra = f['pmra'][:]
    pmdec = f['pmdec'][:]
    rv = f['radial_velocity'][:]

store_rv = rv

null_mask = (np.isnan(rv))
non_null_mask = (~np.isnan(rv))
rv = rv[non_null_mask]
ra = ra[non_null_mask]
dec = dec[non_null_mask]
parallax = parallax[non_null_mask]
pmra = pmra[non_null_mask]
pmdec = pmdec[non_null_mask]

ra = ra * u.deg
dec = dec * u.deg
parallax = parallax * u.mas
rv = rv * u.km / u.s
pmra = pmra * u.mas / u.yr
pmdec = pmdec * u.mas / u.yr

dist = coord.Distance(parallax=parallax, allow_negative=True)

# Coord transformation
icrs = coord.ICRS(
    ra=ra, dec=dec, distance=dist, pm_ra_cosdec=pmra, pm_dec=pmdec, radial_velocity=rv)
gal = icrs.transform_to(coord.Galactocentric())

# Integrate and plot the orbit for 10 Billion years
posvel_accreted = np.stack([
    gal.x.to_value(u.kpc), 
    gal.y.to_value(u.kpc),
    gal.z.to_value(u.kpc),
    gal.v_x.to_value(u.km/u.s),
    gal.v_y.to_value(u.km/u.s),
    gal.v_z.to_value(u.km/u.s),   
], 1)

# Create an action finder from the potential to calculate the action
action_finder = agama.ActionFinder(potential)

Jr, Jz, Jphi = action_finder(posvel_accreted).T

new_Jr = np.empty(len(store_rv))
new_Jr[:] = np.nan
new_Jz = np.empty(len(store_rv))
new_Jz[:] = np.nan
new_Jphi = np.empty(len(store_rv))
new_Jphi[:] = np.nan

new_Jr[non_null_mask] = Jr
new_Jz[non_null_mask] = Jz
new_Jphi[non_null_mask] = Jphi

with h5py.File(path, 'a') as f:
#     del f['Jr']
#     del f['Jz']
#     del f['Jphi']
    f.create_dataset('Jr', data=new_Jr)
    f.create_dataset('Jz', data=new_Jz)
    f.create_dataset('Jphi', data=new_Jphi)