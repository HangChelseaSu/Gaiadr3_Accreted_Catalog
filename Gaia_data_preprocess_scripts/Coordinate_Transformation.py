import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u
import astropy.coordinates as coord

# Read into equitorial coordinates from Gaia DR3 data
path = '/scratch/08949/tg882489/GaiaDR3_data_reduced.hdf5'

with h5py.File(path, 'r') as f:
    ra = f['ra'][:]
    dec = f['dec'][:]
    parallax = f['parallax'][:]
    pmra = f['pmra'][:]
    pmdec = f['pmdec'][:]
    rv = f['radial_velocity'][:]
    source_id = f['source_id'][:]
    poe = f['parallax_over_error'][:]
    # feh = f['feh'][:] #Optional
# Store a copy of rv
store_rv = rv

# Remove nan values based on rv
non_null_mask = (~np.isnan(rv))
source_id = source_id[non_null_mask==1]
ra = ra[non_null_mask==1]
dec = dec[non_null_mask==1]
parallax = parallax[non_null_mask==1]
pmra = pmra[non_null_mask==1]
pmdec = pmdec[non_null_mask==1]
rv = rv[non_null_mask==1]

# Keep a copy of source_id
keep_source_id = source_id

# Add units to the data
ra = ra * u.deg
dec = dec * u.deg
pmra = pmra * u.mas / u.yr
pmdec = pmdec * u.mas / u.yr
parallax = parallax * u.mas
rv = rv * u.km / u.s

dist = coord.Distance(parallax=parallax, allow_negative=True)

# Coord transformation to galactocentric and cylindrical 
icrs = coord.ICRS(
    ra=ra, dec=dec, distance=dist, pm_ra_cosdec=pmra, pm_dec=pmdec, radial_velocity=rv)
gal = icrs.transform_to(coord.Galactocentric())
x_gal = gal.x.to_value(u.kpc)
y_gal = gal.y.to_value(u.kpc)
z_gal = gal.z.to_value(u.kpc)
vx_gal = gal.v_x.to_value(u.km/u.s)
vy_gal = gal.v_y.to_value(u.km/u.s)
vz_gal = gal.v_z.to_value(u.km/u.s)  

icrs = coord.ICRS(
    ra=ra, dec=dec, distance=dist, pm_ra_cosdec=pmra, pm_dec=pmdec, radial_velocity=rv)
icrs.representation_type = 'cylindrical'

rho_cyl = icrs.rho.to_value(u.pc)
phi_cyl = icrs.phi.to_value(u.deg)
z_cyl = icrs.z.to_value(u.pc)
vrho_cyl = icrs.d_rho.to_value(u.mas * u.pc / (u.rad * u.yr))
vphi_cyl = icrs.d_phi.to_value(u.mas / u.yr)
vz_cyl = icrs.d_z.to_value(u.mas * u.pc / (u.rad * u.yr))

# Create new empty arrays for new coordinate data

new_x_gal = np.empty(len(store_rv))
new_x_gal[:] = np.nan
new_y_gal = np.empty(len(store_rv))
new_y_gal[:] = np.nan
new_z_gal = np.empty(len(store_rv))
new_z_gal[:] = np.nan
new_vx_gal = np.empty(len(store_rv))
new_vx_gal[:] = np.nan
new_vy_gal = np.empty(len(store_rv))
new_vy_gal[:] = np.nan
new_vz_gal = np.empty(len(store_rv))
new_vz_gal[:] = np.nan
new_rho_cyl = np.empty(len(store_rv))
new_rho_cyl[:] = np.nan
new_phi_cyl = np.empty(len(store_rv))
new_phi_cyl[:] = np.nan
new_z_cyl = np.empty(len(store_rv))
new_z_cyl[:] = np.nan
new_vrho_cyl = np.empty(len(store_rv))
new_vrho_cyl[:] = np.nan
new_vphi_cyl = np.empty(len(store_rv))
new_vphi_cyl[:] = np.nan
new_vz_cyl = np.empty(len(store_rv))

# Check the lengths of data from coordinate transformation

print('x_gal ', len(x_gal))
print('new_x_gal ', len(new_x_gal))
print('store_rv ', len(store_rv))
print('non_null_mask ', len(non_null_mask), ' ', np.sum(non_null_mask))

# Fill in the new arrays with the transformed data

new_x_gal[non_null_mask==1] = x_gal
new_y_gal[non_null_mask==1] = y_gal
new_z_gal[non_null_mask==1] = z_gal
new_vx_gal[non_null_mask==1] = vx_gal
new_vy_gal[non_null_mask==1] = vy_gal
new_vz_gal[non_null_mask==1] = vz_gal
new_rho_cyl[non_null_mask==1] = rho_cyl
new_phi_cyl[non_null_mask==1] = phi_cyl
new_z_cyl[non_null_mask==1] = z_cyl
new_vrho_cyl[non_null_mask==1] = vrho_cyl
new_vphi_cyl[non_null_mask==1] = vphi_cyl
new_vz_cyl[non_null_mask==1] = vz_cyl

# Save the new coordinate data to the GaiaDR3_data_reduced.hdf5 file

with h5py.File(path, 'a') as f:
    f.create_dataset('x_gal', data=new_x_gal)
    f.create_dataset('y_gal', data=new_y_gal)
    f.create_dataset('z_gal', data=new_z_gal)
    f.create_dataset('vx_gal', data=new_vx_gal)
    f.create_dataset('vy_gal', data=new_vy_gal)
    f.create_dataset('vz_gal', data=new_vz_gal)
    f.create_dataset('rho_cyl', data=new_rho_cyl)
    f.create_dataset('phi_cyl', data=new_phi_cyl)
    f.create_dataset('z_cyl', data=new_z_cyl)
    f.create_dataset('vrho_cyl', data=new_vrho_cyl)
    f.create_dataset('vphi_cyl', data=new_vphi_cyl)
    f.create_dataset('vz_cyl', data=new_vz_cyl)