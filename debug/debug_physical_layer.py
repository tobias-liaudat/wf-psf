import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import wf_psf as wf

import numpy as np
import tensorflow as tf
from wf_psf.tf_layers import TF_poly_Z_field, TF_zernike_OPD, TF_batch_poly_PSF
from wf_psf.tf_layers import TF_NP_poly_OPD, TF_batch_mono_PSF

# [1] Define the physical layer
class TF_physical_layer(tf.keras.layers.Layer):
    """ Store and calculate the zernike coefficients for a given position.

    This layer gives the Zernike contribution of the physical layer.
    It is fixed and not trainable.

    Parameters
    ----------
    obs_pos: tensor(n_stars, 2)
        Observed positions of the `n_stars` in the dataset. The indexing of the
        positions has to correspond to the indexing in the `zks_prior`.
    n_zernikes: int
        Number of Zernike polynomials
    zks_prior: Tensor (n_stars, n_zernikes)
        Zernike coefficients for each position

    """

    def __init__(
        self, 
        obs_pos,
        zks_prior,
        name='TF_physical_layer',
    ):
        super().__init__(name=name)
        self.obs_pos = obs_pos
        self.zks_prior = zks_prior


    def call(self, positions):
        """ Calculate the prior zernike coefficients for a given position.

        The position polynomial matrix and the coefficients should be
        set before calling this function.

        Parameters
        ----------
        positions: Tensor(batch, 2)
            First element is x-axis, second is y-axis.

        Returns
        -------
        zernikes_coeffs: Tensor(batch, n_zernikes, 1, 1)
        """

        def calc_index(idx_pos):
            return tf.where(tf.equal(self.obs_pos, idx_pos))[0, 0]

        # Calculate the indices of the input batch
        indices = tf.map_fn(calc_index, positions, fn_output_signature=tf.int64)
        # Recover the prior zernikes from the batch indexes
        batch_zks = tf.gather(self.zks_prior, indices=indices, axis=0, batch_dims=0)

        return batch_zks[:, :, tf.newaxis, tf.newaxis]



class TF_physical_poly_field(tf.keras.Model):
    """ PSF field forward model with a physical layer

    WaveDiff-original with a physical layer

    Parameters
    ----------
    zernike_maps: Tensor(n_batch, opd_dim, opd_dim)
        Zernike polynomial maps.
    obscurations: Tensor(opd_dim, opd_dim)
        Predefined obscurations of the phase.
    batch_size: int
        Batch size
    obs_pos: Tensor(n_stars, 2)
        The positions of all the stars
    zks_prior: Tensor(n_stars, n_zks)
        The Zernike coeffients of the prior for all the stars
    output_Q: float
        Oversampling used. This should match the oversampling Q used to generate
        the diffraction zero padding that is found in the input `packed_SEDs`.
        We call this other Q the `input_Q`.
        In that case, we replicate the original sampling of the model used to
        calculate the input `packed_SEDs`.
        The final oversampling of the generated PSFs with respect to the
        original instrument sampling depend on the division `input_Q/output_Q`.
        It is not recommended to use `output_Q < 1`.
        Although it works with float values it is better to use integer values.
    d_max_nonparam: int
        Maximum degree of the polynomial for the non-parametric variations.
    l2_param: float
        Parameter going with the l2 loss on the opd. If it is `0.` the loss
        is not added. Default is `0.`.
    output_dim: int
        Output dimension of the PSF stamps.
    n_zks_param: int
        Order of the Zernike polynomial for the parametric model.
    d_max: int
        Maximum degree of the polynomial for the Zernike coefficient variations.
    x_lims: [float, float]
        Limits for the x coordinate of the PSF field.
    y_lims: [float, float]
        Limits for the x coordinate of the PSF field.
    coeff_mat: Tensor or None
        Initialization of the coefficient matrix defining the parametric psf
        field model.

    """

    def __init__(
        self,
        zernike_maps,
        obscurations,
        batch_size,
        obs_pos,
        zks_prior,
        output_Q,
        d_max_nonparam=3,
        l2_param=0.,
        output_dim=64,
        n_zks_param=45,
        d_max=2,
        x_lims=[0, 1e3],
        y_lims=[0, 1e3],
        coeff_mat=None,
        name='TF_physical_poly_field'
    ):
        super(TF_physical_poly_field, self).__init__()

        # Inputs: oversampling used
        self.output_Q = output_Q
        self.n_zks_total = tf.shape(zernike_maps)[0].numpy()

        # Inputs: TF_poly_Z_field
        self.n_zks_param = n_zks_param
        self.d_max = d_max
        self.x_lims = x_lims
        self.y_lims = y_lims

        # Inputs: TF_poly_Z_field
        self.obs_pos = obs_pos
        self.zks_prior = zks_prior
        self.n_zks_prior = tf.shape(zks_prior)[1].numpy()

        # Inputs: TF_NP_poly_OPD
        self.d_max_nonparam = d_max_nonparam
        self.opd_dim = tf.shape(zernike_maps)[1].numpy()

        # Check if the Zernike maps are enough
        if (self.n_zks_prior > self.n_zks_total) or (self.n_zks_param > self.n_zks_total):
            raise ValueError('The number of Zernike maps is not enough.')

        # Inputs: TF_zernike_OPD
        # They are not stored as they are memory-intensive
        # zernike_maps =[]

        # Inputs: TF_batch_poly_PSF
        self.batch_size = batch_size
        self.obscurations = obscurations
        self.output_dim = output_dim

        # Inputs: Loss
        self.l2_param = l2_param

        # Initialize the first layer
        self.tf_poly_Z_field = TF_poly_Z_field(
            x_lims=self.x_lims,
            y_lims=self.y_lims,
            n_zernikes=self.n_zks_param,
            d_max=self.d_max,
        )
        # Initialize the physical layer
        self.tf_physical_layer = TF_physical_layer(
            self.obs_pos,
            self.zks_prior,
        )
        # Initialize the zernike to OPD layer
        self.tf_zernike_OPD = TF_zernike_OPD(zernike_maps=zernike_maps)

        # Initialize the non-parametric layer
        self.tf_np_poly_opd = TF_NP_poly_OPD(
            x_lims=self.x_lims,
            y_lims=self.y_lims,
            d_max=self.d_max_nonparam,
            opd_dim=self.opd_dim,
        )
        # Initialize the batch opd to batch polychromatic PSF layer
        self.tf_batch_poly_PSF = TF_batch_poly_PSF(
            obscurations=self.obscurations,
            output_Q=self.output_Q,
            output_dim=self.output_dim,
        )
        # Initialize the model parameters with non-default value
        if coeff_mat is not None:
            self.assign_coeff_matrix(coeff_mat)


    def get_coeff_matrix(self):
        """ Get coefficient matrix."""
        return self.tf_poly_Z_field.get_coeff_matrix()

    def assign_coeff_matrix(self, coeff_mat):
        """ Assign coefficient matrix."""
        self.tf_poly_Z_field.assign_coeff_matrix(coeff_mat)

    def set_zero_nonparam(self):
        """ Set to zero the non-parametric part."""
        self.tf_np_poly_opd.set_alpha_zero()

    def set_nonzero_nonparam(self):
        """ Set to non-zero the non-parametric part."""
        self.tf_np_poly_opd.set_alpha_identity()

    def set_trainable_layers(self, param_bool=True, nonparam_bool=True):
        """ Set the layers to be trainable or not."""
        self.tf_np_poly_opd.trainable = nonparam_bool
        self.tf_poly_Z_field.trainable = param_bool

    def set_output_Q(self, output_Q, output_dim=None):
        """ Set the value of the output_Q parameter.
        Useful for generating/predicting PSFs at a different sampling wrt the
        observation sampling.
        """
        self.output_Q = output_Q
        if output_dim is not None:
            self.output_dim = output_dim

        # Reinitialize the PSF batch poly generator
        self.tf_batch_poly_PSF = TF_batch_poly_PSF(
            obscurations=self.obscurations, output_Q=self.output_Q, output_dim=self.output_dim
        )

    def zks_pad(self, zk_param, zk_prior):
        """ Pad the zernike coefficients with zeros to have the same length.

        Pad them to have `n_zks_total` length.

        Parameters
        ----------
        zk_param: Tensor(batch, n_zks_param, 1, 1)
            Zernike coefficients for the parametric part
        zk_prior: Tensor(batch, n_zks_prior, 1, 1)
            Zernike coefficients for the prior part

        Returns
        -------
        zk_param: Tensor(batch, n_zks_total, 1, 1)
            Zernike coefficients for the parametric part
        zk_prior: Tensor(batch, n_zks_total, 1, 1)
            Zernike coefficients for the prior part

        """
        # Calculate the number of zernikes to pad
        pad_num = tf.cast(self.n_zks_total - self.n_zks_param, dtype=tf.int32)
        # Pad the zernike coefficients
        padding = [
            (0, 0),
            (0, pad_num),
            (0, 0),
            (0, 0),
        ]
        padded_zk_param = tf.pad(zk_param, padding)

        # Calculate the number of zernikes to pad
        pad_num = tf.cast(self.n_zks_total - self.n_zks_prior, dtype=tf.int32)
        # Pad the zernike coefficients
        padding = [
            (0, 0),
            (0, pad_num),
            (0, 0),
            (0, 0),
        ]
        padded_zk_prior = tf.pad(zk_prior, padding)

        return padded_zk_param, padded_zk_prior

    def predict_mono_psfs(self, input_positions, lambda_obs, phase_N):
        """ Predict a set of monochromatic PSF at desired positions.

        input_positions: Tensor(batch_dim x 2)

        lambda_obs: float
            Observed wavelength in um.

        phase_N: int
            Required wavefront dimension. Should be calculated with as:
            ``simPSF_np = wf.SimPSFToolkit(...)``
            ``phase_N = simPSF_np.feasible_N(lambda_obs)``
        """

        # Initialise the monochromatic PSF batch calculator
        tf_batch_mono_psf = TF_batch_mono_PSF(
            obscurations=self.obscurations, output_Q=self.output_Q, output_dim=self.output_dim
        )
        # Set the lambda_obs and the phase_N parameters
        tf_batch_mono_psf.set_lambda_phaseN(phase_N, lambda_obs)

        # Compute zernikes from parametric model and physical layer
        zks_coeffs = self.compute_zernikes(input_positions)
        # Propagate to obtain the OPD
        param_opd_maps = self.tf_zernike_OPD(zks_coeffs)
        # Calculate the non parametric part
        nonparam_opd_maps = self.tf_np_poly_opd(input_positions)
        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)

        # Compute the monochromatic PSFs
        mono_psf_batch = tf_batch_mono_psf(opd_maps)

        return mono_psf_batch

    def predict_opd(self, input_positions):
        """ Predict the OPD at some positions.

        Parameters
        ----------
        input_positions: Tensor(batch_dim x 2)
            Positions to predict the OPD.

        Returns
        -------
        opd_maps : Tensor [batch x opd_dim x opd_dim]
            OPD at requested positions.

        """
        # Compute zernikes from parametric model and physical layer
        zks_coeffs = self.compute_zernikes(input_positions)
        # Propagate to obtain the OPD
        param_opd_maps = self.tf_zernike_OPD(zks_coeffs)
        # Calculate the non parametric part
        nonparam_opd_maps = self.tf_np_poly_opd(input_positions)
        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)

        return opd_maps

    def compute_zernikes(self,input_positions):
        """ Compute Zernike coefficients at a batch of positions

        This includes the parametric model and the physical layer

        Parameters
        ----------
        input_positions: Tensor(batch_dim, 2)
            Positions to compute the Zernikes.

        Returns
        -------
        zks_coeffs : Tensor [batch, n_zks_total, 1, 1]
            Zernikes at requested positions

        """
        # Calculate parametric part
        zks_params = self.tf_poly_Z_field(input_positions)
        # Calculate the physical layer
        zks_prior = self.tf_physical_layer(input_positions)
        # Pad and sum the zernike coefficients
        padded_zk_param, padded_zk_prior = self.zks_pad(zks_params, zks_prior)
        zks_coeffs = tf.math.add(padded_zk_param, padded_zk_prior)

        return zks_coeffs

    def call(self, inputs):
        """Define the PSF field forward model.

        [1] From positions to Zernike coefficients
        [2] From Zernike coefficients to OPD maps
        [3] From OPD maps and SED info to polychromatic PSFs

        OPD: Optical Path Differences
        """
        # Unpack inputs
        input_positions = inputs[0]
        packed_SEDs = inputs[1]

        # Compute zernikes from parametric model and physical layer
        zks_coeffs = self.compute_zernikes(input_positions)
        # Propagate to obtain the OPD
        param_opd_maps = self.tf_zernike_OPD(zks_coeffs)
        # Add l2 loss on the parametric OPD
        self.add_loss(self.l2_param * tf.math.reduce_sum(tf.math.square(param_opd_maps)))
        # Calculate the non parametric part
        nonparam_opd_maps = self.tf_np_poly_opd(input_positions)
        # Add the estimations
        opd_maps = tf.math.add(param_opd_maps, nonparam_opd_maps)
        # Compute the polychromatic PSFs
        poly_psfs = self.tf_batch_poly_PSF([opd_maps, packed_SEDs])

        return poly_psfs


# [2] Test the model
import tensorflow_addons as tfa

import wf_psf.SimPSFToolkit as SimPSFToolkit
import wf_psf.utils as wf_utils
import wf_psf.tf_mccd_psf_field as tf_mccd_psf_field
import wf_psf.tf_psf_field as tf_psf_field
import wf_psf.metrics as wf_metrics
import wf_psf.train_utils as wf_train_utils


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import seaborn as sns

args = {
    'model':  'poly_physical',
    'id_name':  '_poly_physical_NoSFE_10nm_2k',
    'base_path':  '/Users/tliaudat/Documents/PhD/codes/WF_PSF/euclid_results/tmp_outputs/',
    'log_folder':  "log-files/",
    'model_folder':  "chkp/",
    'optim_hist_folder':  "optim-hist/",
    'plots_folder': "plots/" ,
    'dataset_folder':  '/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-euclid/Euclid-sims/processed_data/',
    'metric_base_path':  '/Users/tliaudat/Documents/PhD/codes/WF_PSF/euclid_results/tmp_outputs/',
    'chkp_save_path':  '/Users/tliaudat/Documents/PhD/codes/WF_PSF/euclid_results/tmp_outputs/',
    'train_dataset_file':  'train_NoSFE_err_10nm_id_01.npy',
    'test_dataset_file':  'test_NoSFE_id_01.npy',
    'n_zernikes':  66,
    'pupil_diameter':  256,
    'n_bins_lda':  20,
    'output_q':  3.,
    'oversampling_rate':  3.,
    'output_dim':  32,
    'd_max':  2,
    'd_max_nonparam':  5,
    'x_lims':  [0, 1e3],
    'y_lims':  [0, 1e3],
    'graph_features': 10,
    'l1_rate':  1e-8,
    'use_sample_weights':  True,
    'batch_size':  32,
    'l_rate_param':  [0.01,0.004],
    'l_rate_non_param':  [0.1,0.06],
    'n_epochs_param':  [15,15],
    'n_epochs_non_param':  [100,50],
    'total_cycles':  2,
    'saved_model_type':  'checkpoint',
    'saved_cycle':  'cycle2',
    'gt_n_zernikes':  66,
    'eval_batch_size':  16,
    'l2_param':  0.,
    'base_id_name':  '_poly_physical_NoSFE_10nm_',
    'suffix_id_name':  '2k',
    'star_numbers':  2000,
}





# Define model run id
run_id_name = args['model'] + args['id_name']

# Define paths
log_save_file = args['base_path'] + args['log_folder']
model_save_file = args['base_path'] + args['model_folder']
optim_hist_file = args['base_path'] + args['optim_hist_folder']
saving_optim_hist = dict()

# # Save output prints to logfile
# old_stdout = sys.stdout
# log_file = open(log_save_file + run_id_name + '_output.log', 'w')
# sys.stdout = log_file
# print('Starting the log file.')

# Print GPU and tensorflow info
device_name = tf.test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))
print('tf_version: ' + str(tf.__version__))

## Prepare the inputs
# Generate Zernike maps
zernikes = wf_utils.zernike_generator(
    n_zernikes=args['n_zernikes'], wfe_dim=args['pupil_diameter']
)
# Now as cubes
np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))
for it in range(len(zernikes)):
    np_zernike_cube[it, :, :] = zernikes[it]
np_zernike_cube[np.isnan(np_zernike_cube)] = 0
tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)
print('Zernike cube:')
print(tf_zernike_cube.shape)

## Load the dictionaries
train_dataset = np.load(
    args['dataset_folder'] + args['train_dataset_file'], allow_pickle=True
)[()]
# train_stars = train_dataset['stars']
# noisy_train_stars = train_dataset['noisy_stars']
# train_pos = train_dataset['positions']
train_SEDs = train_dataset['SEDs']
# train_zernike_coef = train_dataset['zernike_coef']
# train_C_poly = train_dataset['C_poly']
train_parameters = train_dataset['parameters']

test_dataset = np.load(
    args['dataset_folder'] + args['test_dataset_file'], allow_pickle=True
)[()]
# test_stars = test_dataset['stars']
# test_pos = test_dataset['positions']
test_SEDs = test_dataset['SEDs']
# test_zernike_coef = test_dataset['zernike_coef']

# Convert to tensor
tf_noisy_train_stars = tf.convert_to_tensor(train_dataset['noisy_stars'], dtype=tf.float32)
# tf_train_stars = tf.convert_to_tensor(train_dataset['stars'], dtype=tf.float32)
tf_train_pos = tf.convert_to_tensor(train_dataset['positions'], dtype=tf.float32)
tf_test_stars = tf.convert_to_tensor(test_dataset['stars'], dtype=tf.float32)
tf_test_pos = tf.convert_to_tensor(test_dataset['positions'], dtype=tf.float32)

tf_zernike_prior = tf.convert_to_tensor(test_dataset['zernike_prior'], dtype=tf.float32)


print('Dataset parameters:')
print(train_parameters)

## Generate initializations
# Prepare np input
simPSF_np = SimPSFToolkit(
    zernikes,
    max_order=args['n_zernikes'],
    pupil_diameter=args['pupil_diameter'],
    output_dim=args['output_dim'],
    oversampling_rate=args['oversampling_rate'],
    output_Q=args['output_q']
)
simPSF_np.gen_random_Z_coeffs(max_order=args['n_zernikes'])
z_coeffs = simPSF_np.normalize_zernikes(simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms)
simPSF_np.set_z_coeffs(z_coeffs)
simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)
# Obscurations
obscurations = simPSF_np.generate_pupil_obscurations(N_pix=args['pupil_diameter'], N_filter=2)
tf_obscurations = tf.convert_to_tensor(obscurations, dtype=tf.complex64)
# Initialize the SED data list
packed_SED_data = [
    wf_utils.generate_packed_elems(_sed, simPSF_np, n_bins=args['n_bins_lda'])
    for _sed in train_SEDs
]

# Prepare the inputs for the training
tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])

inputs = [tf_train_pos, tf_packed_SED_data]

# Select the observed stars (noisy or noiseless)
outputs = tf_noisy_train_stars
# outputs = tf_train_stars

## Prepare validation data inputs
val_SEDs = test_SEDs
tf_val_pos = tf_test_pos
tf_val_stars = tf_test_stars

# Initialize the SED data list
val_packed_SED_data = [
    wf_utils.generate_packed_elems(_sed, simPSF_np, n_bins=args['n_bins_lda'])
    for _sed in val_SEDs
]

# Prepare the inputs for the validation
tf_val_packed_SED_data = tf.convert_to_tensor(val_packed_SED_data, dtype=tf.float32)
tf_val_packed_SED_data = tf.transpose(tf_val_packed_SED_data, perm=[0, 2, 1])

# Prepare input validation tuple
val_x_inputs = [tf_val_pos, tf_val_packed_SED_data]
val_y_inputs = tf_val_stars
val_data = (val_x_inputs, val_y_inputs)

tf_semiparam_field = TF_physical_poly_field(
    zernike_maps=tf_zernike_cube,
    obscurations=tf_obscurations,
    batch_size=args['batch_size'],
    obs_pos=tf_train_pos,
    zks_prior=tf_zernike_prior,
    output_Q=args['output_q'],
    d_max_nonparam=args['d_max_nonparam'],
    l2_param=args['l2_param'],
    output_dim=args['output_dim'],
    n_zks_param=args['n_zernikes'],
    d_max=args['d_max'],
    x_lims=args['x_lims'],
    y_lims=args['y_lims']
)

# # Model Training
# Prepare the saving callback
# Prepare to save the model as a callback
filepath_chkp_callback = args['chkp_save_path'] + 'chkp_callback_' + run_id_name + '_cycle1'
model_chkp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath_chkp_callback,
    monitor='mean_squared_error',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    save_freq='epoch',
    options=None
)

# Prepare the optimisers
param_optim = tfa.optimizers.RectifiedAdam(lr=args['l_rate_param'][0])
non_param_optim = tfa.optimizers.RectifiedAdam(lr=args['l_rate_non_param'][0])


tf_semiparam_field, hist_param, hist_non_param = wf_train_utils.general_train_cycle(
    tf_semiparam_field,
    inputs=inputs,
    outputs=outputs,
    val_data=val_data,
    batch_size=args['batch_size'],
    l_rate_param=args['l_rate_param'][0],
    l_rate_non_param=args['l_rate_non_param'][0],
    n_epochs_param=args['n_epochs_param'][0],
    n_epochs_non_param=args['n_epochs_non_param'][0],
    param_optim=param_optim,
    non_param_optim=non_param_optim,
    param_loss=None,
    non_param_loss=None,
    param_metrics=None,
    non_param_metrics=None,
    param_callback=None,
    non_param_callback=None,
    general_callback=[model_chkp_callback],
    first_run=True,
    use_sample_weights=args['use_sample_weights'],
    verbose=1
)

end = 1

