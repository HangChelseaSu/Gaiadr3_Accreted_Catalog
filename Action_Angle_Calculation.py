import numpy as np
import pandas as pd
import astropy.units as u
import astropy.coordinates as coord
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py

import agama

import utilities as ut
import gizmo_analysis as gizmo

%matplotlib inline

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

# Read in simulation data
simulation_directory = '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m12i_res7100' #/output
part = gizmo.io.Read.read_snapshots('star', 'redshift', 0, simulation_directory)
host_pos = part['star'].prop('host.distance')
host_vel = part['star'].prop('host.velocity')
x = host_pos[:, 0]
y = host_pos[:, 1]
z = host_pos[:, 2]
vx = host_vel[:, 0]
vy = host_vel[:, 1]
vz = host_vel[:, 2]

# Extract accretion and all ids
csv = pd.read_csv('/scratch/08949/tg882489/data/Ananke/stars_accretion_history_m12i_res7100_v2.csv')
acc_id = csv.id_stars
all_id = part['star']['id'][:]
acc_mask = (np.isin(all_id, acc_id))

# Applying mask on data
x = x[acc_mask]
y = y[acc_mask]
z = z[acc_mask]
vx = vx[acc_mask]
vy = vy[acc_mask]
vz = vz[acc_mask]
posvel_accreted = np.stack([x, y, z, vx, vy, vz], 1)

# Read in DM potential and bar potential
pxr_DM = agama.Potential(file="/scratch/08949/tg882489/data/Ananke/m12i_res7100/600.dark.axi_4.coef_mul_DR")
pxr_bar = agama.Potential(file="/scratch/08949/tg882489/data/Ananke/m12i_res7100/600.bar.axi_4.coef_cylsp_DR")

# Combine bar and DM potential
potential = agama.Potential(pxr_DM, pxr_bar)

# Calculate actions
action_finder = agama.ActionFinder(potential)
Jr, Jz, Jphi = action_finder(posvel_accreted).T

# Plot the actions
hist_range = ((0, 150), (0, 150), (-100, 100))
def plot_action(Jr, Jz, Jphi, hist_range, bins=100, norm=None):
    ''' Plot the 2d distribution of the actions'''
    
    if norm is None:
        norm = mpl.colors.LogNorm(vmin=1, vmax=100)    
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].hist2d(
        Jphi, Jr, bins=(bins, bins), range=(hist_range[2], hist_range[0]), norm=norm)
    axes[1].hist2d(
        Jphi, Jz, bins=(bins, bins), range=(hist_range[2], hist_range[1]), norm=norm)
    axes[2].hist2d(
        Jr, Jz, bins=(bins, bins), range=(hist_range[0], hist_range[1]), norm=norm)

    axes[0].set_xlabel(r' $J_\phi [\mathrm{kpc} \, \mathrm{km} \mathrm{s}^{-1}]$')
    axes[1].set_xlabel(r' $J_\phi [\mathrm{kpc} \, \mathrm{km} \mathrm{s}^{-1}]$')
    axes[2].set_xlabel(r' $J_r [\mathrm{kpc}  \, \mathrm{km} \mathrm{s}^{-1}]$')
    axes[0].set_ylabel(r' $J_r [\mathrm{kpc}  \, \mathrm{km} \mathrm{s}^{-1}]$')
    axes[1].set_ylabel(r' $J_z [\mathrm{kpc}  \, \mathrm{km} \mathrm{s}^{-1}]$')
    axes[2].set_ylabel(r' $J_z [\mathrm{kpc}  \, \mathrm{km} \mathrm{s}^{-1}]$')
    fig.tight_layout()
    
    return fig, axes

plot_action(Jr, Jz, Jphi, hist_range, bins=100, norm=None)

