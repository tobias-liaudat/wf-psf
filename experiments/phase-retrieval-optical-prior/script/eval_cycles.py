# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import matplotlib.pyplot as plt

import wf_psf as wf



# %%

args = {
    'id_name': '_wf_PR_12_cycles_02nm_v1_' ,
    'suffix_id_name': '_' ,
    'base_id_name': '_wf_PR_12_cycles_02nm_v1' ,
    'eval_only_param': False ,
    'total_cycles': 12 ,
    'saved_cycle': 'cycle12' ,
    'reset_dd_features': True ,
    'eval_only_param': False ,
    'project_dd_features': True ,
    'd_max': 2 ,
    'n_zernikes': 66 ,
    'save_all_cycles': True ,
    'n_bins_lda': 20,
    'n_bins_gt': 20,
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
    'eval_mono_metric_rmse': False,
    'eval_opd_metric_rmse': True,
    'eval_train_shape_sr_metric_rmse': True,
    'pupil_diameter': 256 ,
    'n_epochs_param_multi_cycle': "0" ,
    'n_epochs_non_param_multi_cycle': "75" ,
    'l_rate_non_param_multi_cycle': "0.1" ,
    'l_rate_param_multi_cycle': "0" ,
    'l_rate_param': None,
    'l_rate_non_param': None,
    'n_epochs_param': None,
    'n_epochs_non_param': None,
    'model': 'poly_physical' ,
    'model_eval': 'physical' ,
    'cycle_def': 'complete' ,
    'gt_n_zernikes': 66 ,
    'd_max_nonparam': 5 ,
    'saved_model_type': 'external' ,
    'use_sample_weights': True ,
    'l2_param': 0. ,
    'interpolation_type': 'none' ,
    'eval_batch_size': 16 ,
    'train_opt': True ,
    'eval_opt': True ,
    'plot_opt': True ,
    'dataset_folder': '/disk/xray0/tl3/datasets/wf-phase-retrieval/euclid_data_sims/data/',
    'test_dataset_file': 'test_SFE_02nm_id_11.npy',
    'train_dataset_file': 'train_SFE_err_02nm_id_11.npy',
    'base_path': '/disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/',
    'metric_base_path': '/disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/metrics/',
    'chkp_save_path': '/disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/chkp/',
    'plots_folder': 'plots/' ,
    'model_folder': 'chkp/' ,
    'log_folder': 'log-files/' ,
    'optim_hist_folder': 'optim-hist/' ,
    'star_numbers': 1 ,
}



# %%
base_path = '/disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/'
log_folder = 'log-files/'
metric_base_path = '/disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/metrics/'
base_chkp_path = '/disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/chkp/chkp_callback_poly_physical_wf_PR_12_cycles_02nm_v1__'


# base_id_name = '_wf_PR_12_cycles_v1_'
# eval_cycle = 'cycle1'

args['base_path'] = base_path
args['log_folder'] = log_folder
args['metric_base_path'] = metric_base_path




# %%
base_id_name_list = [
    '_wf_PR_12_cycles_02nm_v2',
    '_wf_PR_12_cycles_10nm_v2',
]
base_chkp_path_list = [
    '/disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/chkp/chkp_callback_poly_physical_wf_PR_12_cycles_02nm_v2__',
    '/disk/xray0/tl3/outputs/wf-phase-retrieval-euclid-exp/wf-outputs/chkp/chkp_callback_poly_physical_wf_PR_12_cycles_10nm_v2__',
]
eval_cycle_base_id_list = [5,5]# [12,12] # [16]# [12,18]# [12,12,12,18]

# Iterate over the base_ids
for base_id, base_chkp_path, total_cycles in zip(base_id_name_list, base_chkp_path_list, eval_cycle_base_id_list):
    # Define the list of cycles
    eval_cycle_list = np.arange(1,total_cycles+1)

    # if base_id == '_wf_PR_12_cycles_v1_Zk60_':
    #     args['n_zernikes'] = 60
    # else:
    #     args['n_zernikes'] = 45

    # Iterate over the cycles
    for eval_cycle in eval_cycle_list:

        args['chkp_save_path'] = base_chkp_path + 'cycle' + str(eval_cycle)
        args['id_name'] = base_id + 'cycle' + str(eval_cycle)

        print(args['id_name'])
        # print(args['chkp_save_path'])

        # Process args
        args_2 = wf.utils.load_multi_cycle_params_click(args)
        # Run evaluations
        wf.script_utils.evaluate_model(**args_2)




# %%

str(np.arange(1,13)[2])


# %%






# %%


# %%


# %%


# %%



