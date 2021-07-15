import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import galsim as gs
from wf_psf.utils import generate_packed_elems
from wf_psf.tf_psf_field import build_PSF_model


def compute_poly_metric(tf_semiparam_field, GT_tf_semiparam_field, simPSF_np,
                        tf_pos, tf_SEDs, n_bins_lda=20, batch_size=16):
    """ Calculate metrics for polychromatic reconstructions.

    The ``tf_semiparam_field`` should be the model to evaluate, and the 
    ``GT_tf_semiparam_field`` should be loaded with the ground truth PSF field.

    Relative values returned in [%] (so multiplied by 100).

    Parameters
    ----------
    tf_semiparam_field: PSF field object
        Trained model to evaluate.
    GT_tf_semiparam_field: PSF field object
        Ground truth model to produce GT observations at any position
        and wavelength.
    simPSF_np: PSF simulator object
        Simulation object to be used by ``generate_packed_elems`` function.
    tf_pos: Tensor or numpy.ndarray [batch x 2] floats
        Positions to evaluate the model.
    tf_SEDs: numpy.ndarray [batch x SED_samples x 2]
        SED samples for the corresponding positions.
    n_bins_lda: int
        Number of wavelength bins to use for the polychromatic PSF.
    batch_size: int
        Batch size for the PSF calcualtions.

    Returns
    -------
    rmse: float
        RMSE value.
    rel_rmse: float
        Relative RMSE value. Values in %.
    std_rmse: float
        Sstandard deviation of RMSEs.
    std_rel_rmse: float
        Standard deviation of relative RMSEs. Values in %.

    """
    # Generate SED data list
    packed_SED_data = [generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                            for _sed in tf_SEDs]
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    pred_inputs = [tf_pos , tf_packed_SED_data]

    # Model prediction
    preds = tf_semiparam_field.predict(x=pred_inputs, batch_size=batch_size)

    # GT model prediction
    GT_preds = GT_tf_semiparam_field.predict(x=pred_inputs, batch_size=batch_size)

    # Calculate residuals
    residuals = np.sqrt(np.mean((GT_preds - preds)**2, axis=(1,2)))
    GT_star_mean = np.sqrt(np.mean((GT_preds)**2, axis=(1,2)))

    # RMSE calculations
    rmse = np.mean(residuals)
    rel_rmse = 100. * np.mean(residuals/GT_star_mean)

    # STD calculations
    std_rmse = np.std(residuals)
    std_rel_rmse = 100. * np.std(residuals/GT_star_mean)

    # Print RMSE values
    print('Absolute RMSE:\t %.4e \t +/- %.4e' % (rmse, std_rmse))
    print('Relative RMSE:\t %.4e %% \t +/- %.4e %%' % (rel_rmse, std_rel_rmse))

    return rmse, rel_rmse, std_rmse, std_rel_rmse


def compute_mono_metric(tf_semiparam_field, GT_tf_semiparam_field, simPSF_np, tf_pos, lambda_list):
    """ Calculate metrics for monochromatic reconstructions.

    The ``tf_semiparam_field`` should be the model to evaluate, and the 
    ``GT_tf_semiparam_field`` should be loaded with the ground truth PSF field.

    Relative values returned in [%] (so multiplied by 100).

    Parameters
    ----------
    tf_semiparam_field: PSF field object
        Trained model to evaluate.
    GT_tf_semiparam_field: PSF field object
        Ground truth model to produce GT observations at any position
        and wavelength.
    simPSF_np: PSF simulator object
        Simulation object capable of calculating ``phase_N`` values from 
        wavelength values.
    tf_pos: list of floats [batch x 2]
        Positions to evaluate the model.
    lambda_list: list of floats [batch]
        List of wavelength values in [um] to evaluate the model.

    Returns
    -------
    rmse_lda: list of float
        List of RMSE as a function of wavelength.
    rel_rmse_lda: list of float
        List of relative RMSE as a function of wavelength. Values in %.
    std_rmse_lda: list of float
        List of standard deviation of RMSEs as a function of wavelength.
    std_rel_rmse_lda: list of float
        List of standard deviation of relative RMSEs as a function of
        wavelength. Values in %.

    """
    # Initialise lists
    rmse_lda = []
    rel_rmse_lda = []
    std_rmse_lda = []
    std_rel_rmse_lda = []

    for it in range(len(lambda_list)):
        # Set the lambda (wavelength) and the required wavefront N
        lambda_obs = lambda_list[it]
        phase_N = simPSF_np.feasible_N(lambda_obs)


        # Estimate the monochromatic PSFs
        GT_mono_psf = GT_tf_semiparam_field.predict_mono_psfs(
            input_positions=tf_pos,
            lambda_obs=lambda_obs,
            phase_N=phase_N)
        
        model_mono_psf = tf_semiparam_field.predict_mono_psfs(
            input_positions=tf_pos,
            lambda_obs=lambda_obs,
            phase_N=phase_N)
        
        # Calculate residuals
        residuals = np.sqrt(np.mean((GT_mono_psf - model_mono_psf)**2, axis=(1,2)))
        GT_star_mean = np.sqrt(np.mean((GT_mono_psf)**2, axis=(1,2)))

        # RMSE calculations
        rmse_lda.append(np.mean(residuals))
        rel_rmse_lda.append(100. * np.mean(residuals/GT_star_mean))

        # STD calculations
        std_rmse_lda.append(np.std(residuals))
        std_rel_rmse_lda.append(100. * np.std(residuals/GT_star_mean))

    return rmse_lda, rel_rmse_lda, std_rmse_lda, std_rel_rmse_lda

def compute_opd_metrics(tf_semiparam_field, GT_tf_semiparam_field, pos,
                        batch_size=16):
    """ Compute the OPD metrics.
    
    Need to handle a batch size to avoid Out-Of-Memory errors with
    the GPUs. This is specially due to the fact that the OPD maps
    have a higher dimensionality than the observed PSFs.
    
    Parameters
    ----------
    tf_semiparam_field: PSF field object
        Trained model to evaluate.
    GT_tf_semiparam_field: PSF field object
        Ground truth model to produce GT observations at any position
        and wavelength.
    pos: numpy.ndarray [batch x 2]
        Positions at where to predict the OPD maps.
    batch_size: int
        Batch size to process the OPD calculations.

    Returns
    -------
    rmse: float
        Absolute RMSE value.
    rel_rmse: float
        Relative RMSE value.
    rmse_std: float
        Absolute RMSE standard deviation.
    rel_rmse_std: float
        Relative RMSE standard deviation.

    """
    # Get OPD obscurations
    np_obscurations = np.real(GT_tf_semiparam_field.obscurations.numpy())
    # Define total number of samples
    n_samples = pos.shape[0]

    # Initialise batch variables
    opd_batch = None
    GT_opd_batch = None
    counter = 0
    # Initialise result lists
    rmse_vals = np.zeros(n_samples)
    rel_rmse_vals = np.zeros(n_samples)
    
    
    while counter < n_samples:
        # Calculate the batch end element
        if counter + batch_size <= n_samples:
            end_sample = counter + batch_size
        else:
            end_sample = n_samples

        # Define the batch positions
        batch_pos = pos[counter:end_sample, :]

        # We calculate a batch of OPDs
        opd_batch = tf_semiparam_field.predict_opd(batch_pos)
        GT_opd_batch = GT_tf_semiparam_field.predict_opd(batch_pos)

        # Compute RMSE with the obscured the OPD
        res_opd = np.sqrt(np.mean(
            ((GT_opd_batch.numpy() - opd_batch.numpy()) * np_obscurations)**2,
            axis=(1,2)))
        GT_opd_mean = np.sqrt(np.mean(
            (GT_opd_batch.numpy() * np_obscurations)**2,
            axis=(1,2)))

        # RMSE calculations
        rmse_vals[counter:end_sample] = res_opd
        rel_rmse_vals[counter:end_sample] = 100. * (res_opd/GT_opd_mean)

        # Add the results to the lists
        counter += batch_size

    # Calculate final values
    rmse = np.mean(rmse_vals)
    rel_rmse = np.mean(rel_rmse_vals)
    rmse_std = np.std(rmse_vals)
    rel_rmse_std = np.std(rel_rmse_vals)

    # Print RMSE values
    print('Absolute RMSE:\t %.4e \t +/- %.4e' % (rmse, rmse_std))
    print('Relative RMSE:\t %.4e %% \t +/- %.4e %%' % (rel_rmse, rel_rmse_std))

    return rmse, rel_rmse, rmse_std, rel_rmse_std


def compute_metrics(tf_semiparam_field, simPSF_np, test_SEDs, train_SEDs,
                    tf_test_pos, tf_train_pos, tf_test_stars, tf_train_stars,
                    n_bins_lda, batch_size=16):
    # Generate SED data list
    test_packed_SED_data = [generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                            for _sed in test_SEDs]

    tf_test_packed_SED_data = tf.convert_to_tensor(test_packed_SED_data, dtype=tf.float32)
    tf_test_packed_SED_data = tf.transpose(tf_test_packed_SED_data, perm=[0, 2, 1])
    test_pred_inputs = [tf_test_pos , tf_test_packed_SED_data]
    test_predictions = tf_semiparam_field.predict(x=test_pred_inputs, batch_size=batch_size)


    # Initialize the SED data list
    packed_SED_data = [generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                    for _sed in train_SEDs]
    # First estimate the stars for the observations
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    inputs = [tf_train_pos, tf_packed_SED_data]
    train_predictions = tf_semiparam_field.predict(x=inputs, batch_size=batch_size)

    # Calculate RMSE values
    test_res = np.sqrt(np.mean((tf_test_stars - test_predictions)**2))
    train_res = np.sqrt(np.mean((tf_train_stars - train_predictions)**2))

    # Calculate relative RMSE values
    relative_test_res = test_res / np.sqrt(np.mean((tf_test_stars)**2))
    relative_train_res = train_res / np.sqrt(np.mean((tf_train_stars)**2))

    # Print RMSE values
    print('Test stars absolute RMSE:\t %.4e'%test_res)
    print('Training stars absolute RMSE:\t %.4e'%train_res)

    # Print RMSE values
    print('Test stars relative RMSE:\t %.4e %%'%(relative_test_res*100.))
    print('Training stars relative RMSE:\t %.4e %%'%(relative_train_res*100.))

    return test_res, train_res


def compute_opd_metrics_mccd(tf_semiparam_field, GT_tf_semiparam_field, test_pos,
                        train_pos):
    """ Compute the OPD metrics. """

    np_obscurations = np.real(tf_semiparam_field.obscurations.numpy())

    ## For test positions
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(test_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred =  tf_semiparam_field.tf_NP_mccd_OPD.predict(test_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(test_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate absolute RMSE values
    test_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Calculate relative RMSE values
    relative_test_opd_rmse = test_opd_rmse / np.sqrt(np.mean((GT_opd_maps.numpy()*np_obscurations)**2))

    # Print RMSE values
    print('Test stars absolute OPD RMSE:\t %.4e'%test_opd_rmse)
    print('Test stars relative OPD RMSE:\t %.4e %%\n'%(relative_test_opd_rmse*100.))


    ## For train part
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(train_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred =  tf_semiparam_field.tf_NP_mccd_OPD.predict(train_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(train_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    train_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Calculate relative RMSE values
    relative_train_opd_rmse = train_opd_rmse / np.sqrt(np.mean((GT_opd_maps.numpy()*np_obscurations)**2))


    # Print RMSE values
    print('Train stars absolute OPD RMSE:\t %.4e'%train_opd_rmse)
    print('Train stars relative OPD RMSE:\t %.4e %%\n'%(relative_train_opd_rmse*100.))

    return test_opd_rmse, train_opd_rmse

def compute_opd_metrics_polymodel(tf_semiparam_field, GT_tf_semiparam_field,
                                  test_pos, train_pos):
    """ Compute the OPD metrics. """

    np_obscurations = np.real(tf_semiparam_field.obscurations.numpy())

    ## For test positions
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(test_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred =  tf_semiparam_field.tf_np_poly_opd(test_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(test_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    test_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Calculate relative RMSE values
    relative_test_opd_rmse = test_opd_rmse / np.sqrt(np.mean((GT_opd_maps.numpy()*np_obscurations)**2))

    # Print RMSE values
    print('Test stars OPD RMSE:\t %.4e'%test_opd_rmse)
    print('Test stars relative OPD RMSE:\t %.4e %%\n'%(relative_test_opd_rmse*100.))

    ## For train part
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(train_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred =  tf_semiparam_field.tf_np_poly_opd(train_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(train_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    train_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Calculate relative RMSE values
    relative_train_opd_rmse = train_opd_rmse / np.sqrt(np.mean((GT_opd_maps.numpy()*np_obscurations)**2))

    # Pritn RMSE values
    print('Train stars OPD RMSE:\t %.4e'%train_opd_rmse)
    print('Train stars relative OPD RMSE:\t %.4e %%\n'%(relative_train_opd_rmse*100.))

    return test_opd_rmse, train_opd_rmse

def compute_opd_metrics_param_model(tf_semiparam_field, GT_tf_semiparam_field,
                                    test_pos, train_pos):
    """ Compute the OPD metrics. """

    np_obscurations = np.real(tf_semiparam_field.obscurations.numpy())

    ## For test positions
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(test_pos)
    opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(test_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate absolute RMSE values
    test_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Calculate relative RMSE values
    relative_test_opd_rmse = test_opd_rmse / np.sqrt(np.mean((GT_opd_maps.numpy()*np_obscurations)**2))

    # Print RMSE values
    print('Test stars absolute OPD RMSE:\t %.4e'%test_opd_rmse)
    print('Test stars relative OPD RMSE:\t %.4e %%\n'%(relative_test_opd_rmse*100.))

    ## For train part
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(train_pos)
    opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(train_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    train_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Calculate relative RMSE values
    relative_train_opd_rmse = train_opd_rmse / np.sqrt(np.mean((GT_opd_maps.numpy()*np_obscurations)**2))


    # Print RMSE values
    print('Train stars absolute OPD RMSE:\t %.4e'%train_opd_rmse)
    print('Train stars relative OPD RMSE:\t %.4e %%\n'%(relative_train_opd_rmse*100.))

    return test_opd_rmse, train_opd_rmse

def compute_one_opd_rmse(GT_tf_semiparam_field, tf_semiparam_field, pos, is_poly=False):
    """ Compute the OPD map for one position!. """

    np_obscurations = np.real(tf_semiparam_field.obscurations.numpy())

    tf_pos = tf.convert_to_tensor(pos, dtype=tf.float32)

    ## For test positions
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(tf_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    if is_poly == False:
        NP_opd_pred =  tf_semiparam_field.tf_NP_mccd_OPD.predict(tf_pos)
    else:
        NP_opd_pred =  tf_semiparam_field.tf_np_poly_opd(tf_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(tf_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    opd_rmse = np.sqrt(np.mean(res_opd**2))

    return opd_rmse

def plot_function(mesh_pos, residual, tf_train_pos, tf_test_pos, title='Error'):
    vmax = np.max(residual)
    vmin = np.min(residual)

    plt.figure(figsize=(12,8))
    plt.scatter(mesh_pos[:,0], mesh_pos[:,1], s=100, c=residual.reshape(-1,1), cmap='viridis', marker='s', vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.scatter(tf_train_pos[:,0], tf_train_pos[:,1], c='k', marker='*', s=10, label='Train stars')
    plt.scatter(tf_test_pos[:,0], tf_test_pos[:,1], c='r', marker='*', s=10, label='Test stars')
    plt.title(title)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.show()

def plot_residual_maps(GT_tf_semiparam_field, tf_semiparam_field, simPSF_np, train_SEDs,
                       tf_train_pos, tf_test_pos, n_bins_lda=20, n_points_per_dim=30,
                       is_poly=False):

    # Recover teh grid limits
    x_lims = tf_semiparam_field.x_lims
    y_lims = tf_semiparam_field.y_lims

    # Generate mesh of testing positions
    x = np.linspace(x_lims[0], x_lims[1], n_points_per_dim)
    y = np.linspace(y_lims[0], y_lims[1], n_points_per_dim)
    x_pos, y_pos = np.meshgrid(x, y)

    mesh_pos = np.concatenate((x_pos.flatten().reshape(-1,1), y_pos.flatten().reshape(-1,1)), axis=1)
    tf_mesh_pos = tf.convert_to_tensor(mesh_pos, dtype=tf.float32)

    # Testing the positions
    rec_x_pos = mesh_pos[:,0].reshape(x_pos.shape)
    rec_y_pos = mesh_pos[:,1].reshape(y_pos.shape)

    # Get random SED from the training catalog
    SED_random_integers = np.random.choice(np.arange(train_SEDs.shape[0]), size=mesh_pos.shape[0], replace=True)
    # Build the SED catalog for the testing mesh
    mesh_SEDs = np.array([train_SEDs[_id,:,:] for _id in SED_random_integers])


    # Generate SED data list
    mesh_packed_SED_data = [generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                            for _sed in mesh_SEDs]

    # Generate inputs
    tf_mesh_packed_SED_data = tf.convert_to_tensor(mesh_packed_SED_data, dtype=tf.float32)
    tf_mesh_packed_SED_data = tf.transpose(tf_mesh_packed_SED_data, perm=[0, 2, 1])
    mesh_pred_inputs = [tf_mesh_pos , tf_mesh_packed_SED_data]


    # Predict mesh stars
    model_mesh_preds = tf_semiparam_field.predict(x=mesh_pred_inputs, batch_size=16)
    GT_mesh_preds = GT_tf_semiparam_field.predict(x=mesh_pred_inputs, batch_size=16)

    # Calculate pixel RMSE for each star
    pix_rmse = np.array([np.sqrt(np.mean((_GT_pred-_model_pred)**2))
                            for _GT_pred, _model_pred  in zip(GT_mesh_preds, model_mesh_preds)])

    relative_pix_rmse = np.array([np.sqrt(np.mean((_GT_pred-_model_pred)**2))/np.sqrt(np.mean((_GT_pred)**2))
                                    for _GT_pred, _model_pred  in zip(GT_mesh_preds, model_mesh_preds)])

    # Plot absolute pixel error
    plot_function(mesh_pos, pix_rmse, tf_train_pos, tf_test_pos, title='Absolute pixel error')
    # Plot relative pixel error
    plot_function(mesh_pos, relative_pix_rmse, tf_train_pos, tf_test_pos, title='Relative pixel error')

    # Compute OPD errors
    opd_rmse = np.array([compute_one_opd_rmse(GT_tf_semiparam_field, tf_semiparam_field, _pos.reshape(1,-1), is_poly) for _pos in mesh_pos])

    # Plot absolute pixel error
    plot_function(mesh_pos, opd_rmse, tf_train_pos, tf_test_pos, title='Absolute OPD error')

def plot_imgs(mat, cmap = 'gist_stern', figsize=(20,20)):
    """ Function to plot 2D images of a tensor.
    The Tensor is (batch,xdim,ydim) and the matrix of subplots is
    chosen "intelligently".
    """
    def dp(n, left): # returns tuple (cost, [factors])
        memo = {}
        if (n, left) in memo: return memo[(n, left)]

        if left == 1:
            return (n, [n])

        i = 2
        best = n
        bestTuple = [n]
        while i * i <= n:
            if n % i == 0:
                rem = dp(n / i, left - 1)
                if rem[0] + i < best:
                    best = rem[0] + i
                    bestTuple = [i] + rem[1]
            i += 1

        memo[(n, left)] = (best, bestTuple)
        return memo[(n, left)]


    n_images = mat.shape[0]
    row_col = dp(n_images, 2)[1]
    row_n = int(row_col[0])
    col_n = int(row_col[1])

    plt.figure(figsize=figsize)
    idx = 0

    for _i in range(row_n):
        for _j in range(col_n):

            plt.subplot(row_n,col_n,idx+1)
            plt.imshow(mat[idx,:,:], cmap=cmap);plt.colorbar()
            plt.title('matrix id %d'%idx)

            idx += 1

    plt.show()

def compute_shape_metrics(tf_semiparam_field, GT_tf_semiparam_field, simPSF_np, SEDs,
                    tf_pos, n_bins_lda, output_Q=1, output_dim=64, batch_size=16):
    """ Compute the pixel, shape and size RMSE of a PSF model.

    This is done at a specific sampling and output image dimension.
    It is done for polychromatic PSFs so SEDs are needed.
    """
    # Save original output_Q and output_dim
    original_out_Q = tf_semiparam_field.output_Q
    original_out_dim = tf_semiparam_field.output_dim
    GT_original_out_Q = GT_tf_semiparam_field.output_Q
    GT_original_out_dim = GT_tf_semiparam_field.output_dim

    # Set the required output_Q and output_dim parameters in the models
    tf_semiparam_field.set_output_Q(output_Q=output_Q, output_dim=output_dim)
    GT_tf_semiparam_field.set_output_Q(output_Q=output_Q, output_dim=output_dim)

    # Need to compile the models again
    tf_semiparam_field = build_PSF_model(tf_semiparam_field)
    GT_tf_semiparam_field = build_PSF_model(GT_tf_semiparam_field)


    # Generate SED data list
    packed_SED_data = [generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                            for _sed in SEDs]

    # Prepare inputs
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    pred_inputs = [tf_pos , tf_packed_SED_data]

    # PSF model
    predictions = tf_semiparam_field.predict(x=pred_inputs, batch_size=batch_size)

    # Ground Truth model
    GT_predictions = GT_tf_semiparam_field.predict(x=pred_inputs, batch_size=batch_size)


    # Calculate absolute RMSE values
    pixel_residual = np.sqrt(np.mean((GT_predictions - predictions)**2))

    # Calculate relative RMSE values
    relative_pixel_residual = pixel_residual / np.sqrt(np.mean((GT_predictions)**2))

    # Print pixel RMSE values
    print('\nPixel star absolute RMSE:\t %.4e'%pixel_residual)
    print('Pixel star relative RMSE:\t %.4e %%\n'%(relative_pixel_residual*100.))

    # Measure shapes of the reconstructions
    pred_moments = [gs.hsm.FindAdaptiveMom(gs.Image(_pred), strict=False) for _pred in predictions]

    # Measure shapes of the reconstructions
    GT_pred_moments = [gs.hsm.FindAdaptiveMom(gs.Image(_pred), strict=False) for _pred in GT_predictions]

    pred_e1_HSM, pred_e2_HSM, pred_R2_HSM = [], [], []
    GT_pred_e1_HSM, GT_pred_e2_HSM, GT_pred_R2_HSM = [], [], []

    for it in range(len(GT_pred_moments)):
        if pred_moments[it].moments_status == 0 and GT_pred_moments[it].moments_status == 0:

            pred_e1_HSM.append(pred_moments[it].observed_shape.g1)
            pred_e2_HSM.append(pred_moments[it].observed_shape.g2)
            pred_R2_HSM.append(2*(pred_moments[it].moments_sigma**2))

            GT_pred_e1_HSM.append(GT_pred_moments[it].observed_shape.g1)
            GT_pred_e2_HSM.append(GT_pred_moments[it].observed_shape.g2)
            GT_pred_R2_HSM.append(2*(GT_pred_moments[it].moments_sigma**2))


    pred_e1_HSM = np.array(pred_e1_HSM)
    pred_e2_HSM = np.array(pred_e2_HSM)
    pred_R2_HSM = np.array(pred_R2_HSM)

    GT_pred_e1_HSM = np.array(GT_pred_e1_HSM)
    GT_pred_e2_HSM = np.array(GT_pred_e2_HSM)
    GT_pred_R2_HSM = np.array(GT_pred_R2_HSM)

    # Print shape/size errors
    print('sigma(e1) RMSE = %.4e'%np.sqrt(np.mean((GT_pred_e1_HSM - pred_e1_HSM)**2)))
    print('sigma(e2) RMSE = %.4e'%np.sqrt(np.mean((GT_pred_e2_HSM - pred_e2_HSM)**2)))
    print('sigma(R2)/<R2>  = %.4e \n'%(np.sqrt(np.mean((GT_pred_R2_HSM - pred_R2_HSM)**2))/np.mean(GT_pred_R2_HSM)) )

    # Print relative shape/size errors
    print('relative sigma(e1) RMSE = %.4e %%'%(100.* np.sqrt(np.mean((GT_pred_e1_HSM - pred_e1_HSM)**2)) / np.sqrt(np.mean((GT_pred_e1_HSM)**2)) ))
    print('relative sigma(e2) RMSE = %.4e %%'%(100.* np.sqrt(np.mean((GT_pred_e2_HSM - pred_e2_HSM)**2)) / np.sqrt(np.mean((GT_pred_e2_HSM)**2)) ))

    # Print number of stars
    print('Total number of stars: \t\t%d'%(len(GT_pred_moments)))
    print('Problematic number of stars: \t%d'%(len(GT_pred_moments) - GT_pred_e1_HSM.shape[0]))

    # Re-et the original output_Q and output_dim parameters in the models
    tf_semiparam_field.set_output_Q(output_Q=original_out_Q, output_dim=original_out_dim)
    GT_tf_semiparam_field.set_output_Q(output_Q=GT_original_out_Q, output_dim=GT_original_out_dim)

    # Need to compile the models again
    tf_semiparam_field = build_PSF_model(tf_semiparam_field)
    GT_tf_semiparam_field = build_PSF_model(GT_tf_semiparam_field)

    return predictions, GT_predictions
