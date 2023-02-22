
import jax.numpy as jnp
from jax import vmap
import numpy as np

from wf_psf.SimPSFToolkit import SimPSFToolkit
from wf_psf.jax_wf_psf.jaxSEDs import calc_SED_wave_values, feasible_wavelength, feasible_N
from wf_psf.utils import zernike_generator
from flax import linen as nn




# Generate pupil obsc
obscurations = SimPSFToolkit.generate_pupil_obscurations(N_pix=1024, N_filter=3)

# Telescope parameters
telescope_params = {
    'oversampling_rate': 3., # dimensionless
    'pupil_diameter': 1024,  # In [pix]
    'tel_focal_length': 24.5, # In [m]
    'tel_diameter': 1.2, # In [m]
    'pix_sampling': 12, # In [um]
    'WFE_dim': 256, # In [WFE pix] -> Modelling choice.
    'n_zernikes': 21, # Order of the Zernike polynomial
    'output_Q': 1, # Choice of the output pixel resolution
    'output_dim': 64, # Output image dimension in [pix]
    'max_order': 45, # Max zernike order
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

# Generate Zernike polynomials
zernike_maps = zernike_generator(telescope_params['n_zernikes'], telescope_params['WFE_dim'])
# Generate pupil mask
pupil_mask = ~np.isnan(zernike_maps[0])


# generate_mono_PSF

# generate_poly_PSF
# 
# 
#   generate_mono_PSF
#       calculate_opd
#       diffract_phase
#           feasible_wavelength
#               feasible_N
#           feasible_N
#           fft_diffract
#               resize (cv2) -> replace with tf.keras.layers.AveragePooling2D() -> use flax (nn.avg_pool())


# Need a function to generate Zernike coefficients


def downsample_op(imgs, output_Q):
    """ Downsample opertarion. Only supports integer downsampling.
    
    imgs : dimensions (batch, window dims..., features)
    output_Q : int
    """
    if output_Q != 1:
        imgs = nn.avg_pool(
            inputs=jnp.expand_dims(imgs, 2),
            window_shape=(int(output_Q), int(output_Q)),
            strides=(int(output_Q), int(output_Q)),
            padding='VALID',
            count_include_pad=True # Should it be false?
        )

        imgs = jnp.squeeze(imgs, axis=2)

    return imgs

def fft_diffract(wf, output_Q, output_dim=64):
    # Perform the FFT-based diffraction operation
    fft_wf = jnp.fft.fftshift(jnp.fft.fft2(wf))
    psf = jnp.abs(fft_wf)**2

    # Calculate crop dimensions
    if output_dim * output_Q < psf.shape[0]:
        start = int(psf.shape[0] // 2 - (output_dim * output_Q) // 2)
        stop = int(psf.shape[0] // 2 + (output_dim * output_Q) // 2)
    else:
        start = int(0)
        stop = int(psf.shape[0])

    # Crop psf
    psf = psf.at[start:stop, start:stop].get()

    # Downsample 
    psf = downsample_op(psf, output_Q)

    return psf




def diffract_phase(opd, pupil_mask, obscurations, telescope_params, lambda_obs=0.8):
    """Diffract the phase map."""

    # Calculate the feasible lambda closest to lambda_obs
    possible_lambda = feasible_wavelength(telescope_params, lambda_obs)

    # Save wavelength
    lambda_obs = possible_lambda

    # Calculate the required N for the input lambda_obs
    possible_N = feasible_N(telescope_params, lambda_obs)
    
    # Generate the full phase and
    # Add zeros to the phase to have the correct fourier sampling
    start = (possible_N // 2 - telescope_params['WFE_dim'] // 2).astype(int)
    stop = (possible_N // 2 + telescope_params['WFE_dim'] // 2).astype(int)

    phase = jnp.zeros((int(possible_N), int(possible_N)), dtype=jnp.complex64)
    # phase[start:stop,start:stop][pupil_mask] = jnp.exp(2j * jnp.pi * opd[pupil_mask] / lambda_obs)
    phase = phase.at[start:stop,start:stop].set(jnp.exp(2j * jnp.pi * opd / lambda_obs))
    # Project obscurations to the phase
    # phase[start:stop, start:stop] *= obscurations
    phase = phase.at[start:stop, start:stop].multiply(obscurations)

    # FFT-diffract the phase (wavefront) and then crop to desired dimension
    psf = fft_diffract(
        wf=phase, output_Q=telescope_params['output_Q'], output_dim=telescope_params['output_dim']
    )

    # Normalize psf
    psf = psf /  jnp.sum(psf)

    return psf


def calculate_opd_from_zernikes(z_coeffs, zernike_maps):
    """Calculate the OPD from the Zernike coefficients.
    
    z_coeffs: (n,) array of Zernike coefficients
    zernike_maps: (n, wfe_dim, wfe_dim) Zernike maps
    """

    # Create the phase with the Zernike basis
    # opd = 0
    # for it in range(len(z_coeffs)):
    #     opd += zernike_maps[it] * z_coeffs[it]

    assert z_coeffs.shape[0] == zernike_maps.shape[0]

    # return wavefront
    return jnp.einsum('ijk,i->jk', zernike_maps, z_coeffs)


def generate_mono_PSF(opd, pupil_mask, obscurations, telescope_params, lambda_obs, SED_lambda_norm=1):
    """Generate monochromatic PSF.

    Optional to weight by the `SED_norm`.
    """

    # Calculate the OPD from the Zernike coefficients
    # opd = calculate_opd(z_coeffs, zernike_maps)

    # Apply the diffraction operator using the opd (optical path differences)
    mono_psf = diffract_phase(opd, pupil_mask, obscurations, telescope_params, lambda_obs)

    return mono_psf * SED_lambda_norm


def generate_poly_PSF(opd, SED, pupil_mask, obscurations, SED_params, telescope_params):
    """Generate polychromatic PSF with a specific SED.

    The wavelength space will be the Euclid VIS instrument band:
    [550,900]nm and will be sample in ``n_bins``.

    """
    # Calculate the feasible values of wavelength and the corresponding
    # SED interpolated values
    feasible_wv, SED_norm = calc_SED_wave_values(SED, SED_params, telescope_params)

    # Return polychromatic PSF
    return vmap(
        generate_mono_PSF, in_axes=(None, None, None, None, 0, 0), out_axes=(0)
    )(opd, pupil_mask, obscurations, telescope_params, lambda_obs=feasible_wv, SED_lambda_norm=SED_norm)


