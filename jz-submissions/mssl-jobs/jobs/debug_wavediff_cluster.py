

import numpy as np
import wf_psf as wf
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

args = {
    'id_name': '_wf_PR_9_cycles_v1' ,
    'suffix_id_name': 'v1' ,
    'base_id_name': '_wf_PR_9_cycles_' ,
    'eval_only_param': False ,
    'total_cycles': 2 ,
    'saved_cycle': 'cycle2' ,
    'reset_dd_features': True ,
    'eval_only_param': False ,
    'project_dd_features': True ,
    'd_max': 2 ,
    'n_zernikes': 45 ,
    'save_all_cycles': True ,
    'n_bins_lda': 8 ,
    'output_dim': 32,
    'batch_size': 32,
    'oversampling_rate': 3.,
    'output_q': 3.,
    'sed_sigma': 0.,
    'x_lims': [0, 1e3],
    'y_lims': [0, 1e3],
    'sed_interp_kind': 'linear',
    'interp_pts_per_bin': 0,
    'extrapolate': True,
    'opt_stars_rel_pix_rmse': True ,
    'pupil_diameter': 256 ,
    # 'n_epochs_param_multi_cycle': "0" ,
    # 'n_epochs_non_param_multi_cycle': "3" ,
    # 'l_rate_non_param_multi_cycle': "0.1" ,
    # 'l_rate_param_multi_cycle': "0" ,
    'l_rate_param': [0.04, 0.04],
    'l_rate_non_param': [0.1, 0.04],
    'n_epochs_param': [4, 4],
    'n_epochs_non_param': [4, 4],
    'model': 'poly' ,
    'model_eval': 'poly' ,
    'cycle_def': 'complete' ,
    'gt_n_zernikes': 45 ,
    'd_max_nonparam': 5 ,
    'saved_model_type': 'checkpoint' ,
    'use_sample_weights': True ,
    'l2_param': 0. ,
    'interpolation_type': 'none' ,
    'eval_batch_size': 16 ,
    'train_opt': True ,
    'eval_opt': True ,
    'plot_opt': True ,
    # 'dataset_folder': '/disk/xray0/tl3/datasets/wf-phase-retrieval/data/' ,
    # 'test_dataset_file': 'test_Euclid_res_id_010_8_bins.npy' ,
    # 'train_dataset_file': 'train_Euclid_res_2000_TrainStars_id_010_8_bins.npy' ,
    'dataset_folder': '/disk/xray0/tl3/repos/wf-psf/data/coherent_euclid_dataset/' ,
    'test_dataset_file': 'test_Euclid_res_id_001.npy' ,
    'train_dataset_file': 'train_Euclid_res_200_TrainStars_id_001.npy' ,
    'base_path': '/disk/xray0/tl3/outputs/wf-phase-retrieval/wf-outputs/' ,
    'metric_base_path': '/disk/xray0/tl3/outputs/wf-phase-retrieval/wf-outputs/metrics/' ,
    'chkp_save_path': '/disk/xray0/tl3/outputs/wf-phase-retrieval/wf-outputs/chkp/9_cycles/' ,
    'plots_folder': 'plots/9_cycles/' ,
    'model_folder': 'chkp/9_cycles/' ,
    'log_folder': 'log-files/' ,
    'optim_hist_folder': 'optim-hist/' ,
    'star_numbers': 1 ,
}


args = wf.utils.load_multi_cycle_params_click(args)
print('Training...')
wf.script_utils.train_model(**args)
print('Finish training...')
