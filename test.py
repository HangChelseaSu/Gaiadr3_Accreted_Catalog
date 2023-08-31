import h5py
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.coordinates as coord
import matplotlib.pyplot as plt
import matplotlib as mpl

path_phot = '/scratch/05328/tg846280/GAIADR3_data/temp_gaia_edr3_data/all_w_rv_phot.h5'
path_spec = '/scratch/05328/tg846280/GAIADR3_data/temp_gaia_edr3_data/all_w_rv_spec.h5'
df_phot = pd.read_hdf(path_phot)
df_spec = pd.read_hdf(path_spec)
# path_out = '/work2/08949/tg882489/stampede2/data/Ananke/GaiaDR3_data_reduced.hdf5'

# combine df_phot and df_spec based on source_id


df = pd.merge(df_phot, df_spec, on="source_id", how="inner")

#print 