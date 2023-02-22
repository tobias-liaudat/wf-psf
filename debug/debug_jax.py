
import numpy as np
import jax.numpy as jnp
import jax as jax
from wf_psf.jax_wf_psf import jaxSimPSFToolkit
from wf_psf.SimPSFToolkit import SimPSFToolkit
from wf_psf.utils import zernike_generator

import matplotlib.pyplot as plt

# Telescope parameters
telescope_params = {
    'oversampling_rate': 3., # dimensionless
    'pupil_diameter': 1024,  # In [pix]
    'tel_focal_length': 24.5, # In [m]
    'tel_diameter': 1.2, # In [m]
    'pix_sampling': 12, # In [um]
    'WFE_dim': 256, # In [WFE pix] -> Modelling choice.
    'n_zernikes': 45, # Order of the Zernike polynomial
    'output_Q': 3, # Choice of the output pixel resolution. Only integer values admitted by the current implementation.
    'output_dim': 32, # Output image dimension in [pix]
    # 'max_order': 45, # Max zernike order
    'LP_filter_length': 2, # Filter size for smoothing obscurations
}

# SED parameters
SED_params = {
    'n_bins': 8, # Number of bins to use
    'interp_pts_per_bin': 0, #  Number of points to interpolate in between SED values. It can be 0, 1 or 2.
    'SED_sigma': 0, # Standard deviation of the multiplicative SED Gaussian noise.
    'SED_interp_kind': 'linear', # SED interpolation kind. To augment the number of bins. Options are `'cubic'` or `'linear'`.
    'extrapolate': True, # SED interpolation mode. Default mode uses extrapolation.
    'interp_kind': 'cubic', # Interpolation kind for the small adjustment for the feasible wavelengths.
}


# Generate pupil obsc
obscurations = SimPSFToolkit.generate_pupil_obscurations(
    N_pix=telescope_params['WFE_dim'], N_filter=telescope_params['LP_filter_length']
)
# Generate Zernike polynomials
zernike_maps = jnp.nan_to_num(
    jnp.array(zernike_generator(telescope_params['n_zernikes'], telescope_params['WFE_dim'])),
    nan=0.
)
# Generate pupil mask
pupil_mask = ~np.isnan(zernike_maps[0])


# Draw random zernike coefficients
# key = jax.random.PRNGKey(0)
# z_coeffs = jax.random.normal(key, (telescope_params['n_zernikes'],))

# Load Zernike coefficients from simulated FOV
dataset = np.load(
    '/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/data/coherent_euclid_dataset/train_Euclid_res_500_TrainStars_id_001.npy',
    allow_pickle=True
)[()]
star_id = 0

z_coeffs = dataset['zernike_coef'][star_id, 0:telescope_params['n_zernikes'], 0]
SEDs = dataset['SEDs'][star_id, :, :]

# Compute OPD
opd = jaxSimPSFToolkit.calculate_opd_from_zernikes(z_coeffs, zernike_maps)

# vmax = np.max(np.abs(opd))
# plt.figure()
# plt.imshow(opd, cmap='seismic', vmax=vmax, vmin=-vmax);plt.colorbar()
# plt.show()

# Compute a monochromatic PSF at a given wavelength
lambda_obs = 0.8
mono_psf = jaxSimPSFToolkit.generate_mono_PSF(opd, pupil_mask, obscurations, telescope_params, lambda_obs, SED_lambda_norm=1)


# plt.figure()
# plt.subplot(311)
# plt.imshow(mono_psf, cmap='gist_stern');plt.colorbar()
# plt.subplot(312)
# plt.imshow(dataset['stars'][0,:,:], cmap='gist_stern');plt.colorbar()
# plt.subplot(313)
# plt.imshow(dataset['stars'][0,:,:]-mono_psf, cmap='gist_stern');plt.colorbar()
# plt.show()


# Compute a polychromatic PSF
poly_psf = jaxSimPSFToolkit.generate_poly_PSF(opd, SEDs, pupil_mask, obscurations, SED_params, telescope_params)


plt.figure()
plt.subplot(311)
plt.imshow(poly_psf, cmap='gist_stern');plt.colorbar()
plt.subplot(312)
plt.imshow(dataset['stars'][0,:,:], cmap='gist_stern');plt.colorbar()
plt.subplot(313)
plt.imshow(dataset['stars'][0,:,:]-poly_psf, cmap='gist_stern');plt.colorbar()
plt.show()


print('Good bye!')
