import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u
import astropy.coordinates as coord

# Read in the BPRP catalog for feh column

path0 = '/ocean/projects/phy210068p/hsu1/BPRP_spec/stellar_params_catalog_00.h5'
path1 = '/ocean/projects/phy210068p/hsu1/BPRP_spec/stellar_params_catalog_01.h5'
path2 = '/ocean/projects/phy210068p/hsu1/BPRP_spec/stellar_params_catalog_02.h5'
path3 = '/ocean/projects/phy210068p/hsu1/BPRP_spec/stellar_params_catalog_03.h5'
path4 = '/ocean/projects/phy210068p/hsu1/BPRP_spec/stellar_params_catalog_04.h5'
path5 = '/ocean/projects/phy210068p/hsu1/BPRP_spec/stellar_params_catalog_05.h5'
path6 = '/ocean/projects/phy210068p/hsu1/BPRP_spec/stellar_params_catalog_06.h5'
path7 = '/ocean/projects/phy210068p/hsu1/BPRP_spec/stellar_params_catalog_07.h5'
path8 = '/ocean/projects/phy210068p/hsu1/BPRP_spec/stellar_params_catalog_08.h5'
path9 = '/ocean/projects/phy210068p/hsu1/BPRP_spec/stellar_params_catalog_09.h5'

files = [path0, path1, path2, path3, path4, path5, path6, path7, path8, path9]
feh = []
BPRP_id = []

# Load in source_id and feh from BPRP catalog
for i in range(len(files)):
    with h5py.File(files[i], 'r') as f:
        feh = np.append(feh, f['stellar_params_est'][:,1])
        b_id = np.array(f['gdr3_source_id'][:], dtype=np.int64)
        BPRP_id = np.append(BPRP_id, b_id)

# Load in GaiaDR3 data with poe>10 cut only

path = '/ocean/projects/phy210068p/hsu1/Ananke_datasets_training/GaiaDR3_data_reduced.hdf5'
with h5py.File(path, 'r') as f:
    ra = f['ra'][:]
    dec = f['dec'][:]
    parallax = f['parallax'][:]
    pmra = f['pmra'][:]
    pmdec = f['pmdec'][:]
    rv = f['radial_velocity'][:]
    main_source_id = f['source_id'][:]

    # Create pandas dataframe for GaiaDR3 data and BPRP data

    gaia_df = pd.DataFrame(
    {
        #sort based on id
        'main_source_id': main_source_id,
        'parallax': parallax,
        'ra': ra,
        'dec': dec,
        'pmra': pmra,
        'pmdec': pmdec,
        'rv': rv
    },
#     index= main_source_id,
)

BPRP_df = pd.DataFrame(
    {
        'feh': feh,
        'BPRP_id': BPRP_id
    },
#     index=BPRP_id,
)

# Merge the two dataframes based on source_id

merged_df = gaia_df.merge(BPRP_df, left_on='main_source_id', right_on='BPRP_id')

# Save the merged dataframe as hdf5 file

ra = merged_df.ra
dec = merged_df.dec
parallax = merged_df.parallax
pmra = merged_df.pmra
pmdec = merged_df.pmdec
rv = merged_df.rv
feh = merged_df.feh
source_id = merged_df.main_source_id

with h5py.File('/ocean/projects/phy210068p/hsu1/Ananke_datasets_training/GaiaDR3_data_reduced_feh.hdf5', 'a') as f:
    f.create_dataset('ra', data=ra)
    f.create_dataset('dec', data=dec)
    f.create_dataset('parallax', data=parallax)
    f.create_dataset('pmra', data=pmra)
    f.create_dataset('pmdec', data=pmdec)
    f.create_dataset('radial_velocity', data=rv)
    f.create_dataset('feh', data=feh)
    f.create_dataset('source_id', data=source_id)
