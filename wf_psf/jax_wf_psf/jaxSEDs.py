
import numpy as np
import jax.numpy as jnp
from wf_psf.SimPSFToolkit import SimPSFToolkit
import scipy.interpolate as sinterp


#   calc_SED_wave_values
#       gen_SED_sampler
#           filter_SED
#           SED_gen_noise
#           interp_SED
#               gen_SED_interp
#       feasible_wavelength

def feasible_N(telescope_params, lambda_obs):
    """Calculate the feasible N for a lambda_obs diffraction.

    Input wavelength must be in [um].
    """
    # Calculate the required N for the input lambda_obs
    req_N = (telescope_params['oversampling_rate'] * telescope_params['pupil_diameter'] * lambda_obs *
                telescope_params['tel_focal_length']) / (
                    telescope_params['tel_diameter'] * telescope_params['pix_sampling']
                )
    # Recalculate the req_N into a possible value (a pair integer)
    possible_N = jnp.floor((req_N // 2) * 2)

    return possible_N

def feasible_wavelength(telescope_params, lambda_obs):
    """Calculate closest fesible wavelength to target wavelength.

    Input wavelength must be in [um].
    """
    # Calculate a feasible N for the input lambda_obs
    possible_N = feasible_N(telescope_params, lambda_obs)

    # Recalculate the corresponding the wavelength
    possible_lambda = (possible_N * telescope_params['tel_diameter'] * telescope_params['pix_sampling']) / (
        telescope_params['pupil_diameter'] * telescope_params['oversampling_rate'] * telescope_params['tel_focal_length']
    )

    return possible_lambda

def interp_SED(SED_filt, SED_params):
    """Interpolate the binned SED.

    Returns a ('n_bins')x('n_points'+1) point SED and wvlength vector.

    Parameters
    ----------
    SED_filt: np.ndarray
        The filtered SED. In the first column it contains the wavelength positions. In the second column the SED value for each bin. 
    SED_params: dict
        interp_pts_per_bin: int
            Number of points to add in each of the filtered SED bins. 
            It can only be 0, 1, 2 or 3. 

    """
    # Generate interpolation function from the binned SED
    _, SED_interpolator = SimPSFToolkit.gen_SED_interp(SED_filt, SED_params['n_bins'], SED_params['SED_interp_kind'])
    wv_step = SED_filt[1, 0] - SED_filt[0, 0]

    # Regenerate the wavelenght points
    if SED_params['interp_pts_per_bin'] == 1:
        if SED_params['extrapolate']:
            # Add points at the border of each bin : *--o--*--o--*--o--*--o--*
            SED = np.zeros((SED_params['n_bins'] * 2 + 1, 3))
            # Set wavelength points then interpolate
            SED[1::2, 0] = SED_filt[:, 0]
            SED[2::2, 0] = SED_filt[:, 0] + wv_step / 2
            SED[0, 0] = SED_filt[0, 0] - wv_step / 2
            SED[:, 1] = SED_interpolator(SED[:, 0])
            # Set weigths for new bins (borders have half the bin size)
            SED[:, 2] = np.ones(SED_params['n_bins'] * 2 + 1)
            SED[0, 2], SED[-1, 2] = 0.5, 0.5
            # Apply weights to bins
            SED[:, 1] *= SED[:, 2]
        else:
            # Add points at the border of each bin with no extrapolation: ---o--*--o--*--o--*--o---
            SED = np.zeros((SED_params['n_bins'] * 2 - 1, 3))
            # Set wavelength points then interpolate
            SED[::2, 0] = SED_filt[:, 0]
            SED[1::2, 0] = SED_filt[1:, 0] - wv_step / 2
            SED[:, 1] = SED_interpolator(SED[:, 0])
            # Set weigths for new bins (borders have half the bin size)
            SED[:, 2] = np.ones(SED_params['n_bins'] * 2 - 1)
            SED[0, 2], SED[-1, 2] = 1.5, 1.5
            # Apply weights to bins
            SED[:, 1] *= SED[:, 2]
    elif SED_params['interp_pts_per_bin'] == 2:
        if SED_params['extrapolate']:
            # Add 2 points per bin: -*-o-*-*-o-*-*-o-*-*-o-*-
            SED = np.zeros((SED_params['n_bins'] * 3, 3))
            SED[1::3, 0] = SED_filt[:, 0]
            SED[::3, 0] = SED_filt[:, 0] - wv_step / 3
            SED[2::3, 0] = SED_filt[:, 0] + wv_step / 3
            SED[:, 1] = SED_interpolator(SED[:, 0])
            # Set weigths for new bins (borders have half the bin size)
            SED[:, 2] = np.ones(SED_params['n_bins'] * 3)
            # Apply weights to bins
            SED[:, 1] *= SED[:, 2]
        else:
            # Add 2 points per bin with no extrapolation: ---o-*-*-o-*-*-o-*-*-o---
            SED = np.zeros((SED_params['n_bins'] * 3 - 2, 3))
            SED[::3, 0] = SED_filt[:, 0]
            SED[1::3, 0] = SED_filt[1:, 0] - 2 * wv_step / 3
            SED[2::3, 0] = SED_filt[1:, 0] - wv_step / 3
            SED[:, 1] = SED_interpolator(SED[:, 0])
            # Set weigths for new bins (borders have half the bin size)
            SED[:, 2] = np.ones(SED_params['n_bins'] * 3 - 2)
            SED[0, 2], SED[-1, 2] = 2, 2
            # Apply weights to bins
            SED[:, 1] *= SED[:, 2]
    elif SED_params['interp_pts_per_bin'] == 3:
        if SED_params['extrapolate']:
            # Add 3 points inside each bin :  *-*-o-*-*-*-o-*-*-*-o-*-*-*-o-*-*
            SED = np.zeros((SED_params['n_bins'] * 4 + 1, 3))
            # Set wavelength points then interpolate
            SED[4::4, 0] = SED_filt[:, 0] + wv_step / 2
            SED[0, 0] = SED_filt[0, 0] - wv_step / 2
            SED[1::4, 0] = SED_filt[:, 0] - wv_step / 4
            SED[2::4, 0] = SED_filt[:, 0]
            SED[3::4, 0] = SED_filt[:, 0] + wv_step / 4
            # Evaluate interpolator at new points
            SED[:, 1] = SED_interpolator(SED[:, 0])
            # Set weigths for new bins (borders have half the bin size)
            SED[:, 2] = np.ones(SED_params['n_bins'] * 4 + 1)
            SED[0, 2], SED[-1, 2] = 0.5, 0.5
            # Apply weights to bins
            SED[:, 1] *= SED[:, 2]
    else:
        SED = SED_filt

    # Normalize SED
    SED[:, 1] = SED[:, 1] / np.sum(SED[:, 1])

    return SED


def gen_SED_sampler(SED, SED_params):
    """Generate SED sampler.
    
    Returns the sampler and the wavelengths in [nm]
    """
    # Integrate SED into n_bins
    SED_filt = SimPSFToolkit.filter_SED(SED, SED_params['n_bins'])

    # Add noise. Scale sigma for each bin. Normalise the SED.
    #SED_filt[:,1] = SED_filt[:,1] + self.SED_gen_noise(len(SED_filt), self.SED_sigma)/len(SED_filt) # Here we assume 1/N as the mean bin value
    SED_filt[:, 1] += np.multiply(
        SED_filt[:, 1], SimPSFToolkit.SED_gen_noise(len(SED_filt), SED_params['SED_sigma'])
    )
    SED_filt[:, 1] = SED_filt[:, 1] / np.sum(SED_filt[:, 1])

    # Add inside-bin points - Interpolate
    SED_filt = interp_SED(SED_filt, SED_params)

    # Add weights if not present
    if SED_filt.shape[1] == 2:
        weights = np.ones((SED_filt.shape[0], 1))
        SED_filt = np.hstack((SED_filt, weights))

    # Interpolate the unweighted SED
    SED_sampler = sinterp.interp1d(
        SED_filt[:, 0],
        SED_filt[:, 1] / SED_filt[:, 2],
        kind=SED_params['interp_kind'],
        bounds_error=False,
        fill_value="extrapolate"
    )

    return SED_filt[:, 0], SED_sampler, SED_filt[:, 2]


def calc_SED_wave_values(SED, SED_params, telescope_params):
    """Calculate feasible wavelength and SED values.

    Feasable so that the padding number N is integer.
    """
    # Generate SED interpolator and wavelength array (use new sampler method)
    wvlength, SED_interp, weights = gen_SED_sampler(SED, SED_params)

    # Convert wavelength from [nm] to [um]
    wvlength_um = wvlength / 1e3

    # Calculate feasible wavelengths (in [um])
    feasible_wv = np.array([feasible_wavelength(telescope_params, _wv) for _wv in wvlength_um])

    # Interpolate and normalize SED
    SED_norm = SED_interp(feasible_wv * 1e3)  # Interpolation is done in [nm]
    SED_norm *= weights  # Weight by the relative size of the bins, then normalise.
    SED_norm /= np.sum(SED_norm)

    return feasible_wv, SED_norm

