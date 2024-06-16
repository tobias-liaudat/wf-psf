# %%

import numpy as np
import matplotlib.pyplot as plt

import wf_psf as wf



# %%

args = {
    'id_name': '_wf_PR_NewPRoj_12_cycles_v0' ,
    'suffix_id_name': '_' ,
    'base_id_name': '_wf_PR_NewPRoj_12_cycles_' ,
    'random_seed' : 5000,
    'eval_only_param': False ,
    'total_cycles': 12 ,
    'saved_cycle': 'cycle12' ,
    'reset_dd_features': True ,
    'eval_only_param': False ,
    'project_dd_features': True ,
    'd_max': 2 ,
    'n_zernikes': 45 ,
    'save_all_cycles': True ,
    'n_bins_lda': 8,
    'n_bins_gt': 8,
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
    'model': 'poly' ,
    'model_eval': 'poly' ,
    'cycle_def': 'complete' ,
    'gt_n_zernikes': 45 ,
    'd_max_nonparam': 5 ,
    'saved_model_type': 'external' ,
    'use_sample_weights': True ,
    'l2_param': 0. ,
    'interpolation_type': 'none' ,
    'eval_batch_size': 16 ,
    'train_opt': True ,
    'eval_opt': True ,
    'plot_opt': True ,
    'dataset_folder': '/gpfswork/rech/ynx/ulx23va/wfv2/dataset_pr/data/',
    'test_dataset_file': 'test_Euclid_res_id_010_8_bins.npy',
    'train_dataset_file': 'train_Euclid_res_2000_TrainStars_id_010_8_bins.npy',
    'base_path': '/gpfsstore/rech/ynx/ulx23va/wfv2/phase-retrieval-PR/output_v1/',
    'metric_base_path': '/gpfsstore/rech/ynx/ulx23va/wfv2/phase-retrieval-PR/output_v1/metrics/',
    'chkp_save_path': '/gpfsstore/rech/ynx/ulx23va/wfv2/phase-retrieval-PR/output_v1/chkp/',
    'plots_folder': 'plots/' ,
    'model_folder': 'chkp/' ,
    'log_folder': 'log-files/' ,
    'optim_hist_folder': 'optim-hist/' ,
    'star_numbers': 1 ,
}





# %%
base_chkp_path = '/gpfsstore/rech/ynx/ulx23va/wfv2/phase-retrieval-PR/output_v1/chkp/'

random_seed_list = [
    5000,
    5100,
    5200,
    5300
]

chkp_path_format = base_chkp_path + f"chkp_callback_poly_wf_PR_NewPRoj_12_cycles_v%d_cycle%d"
id_name_format = f"_wf_PR_NewPRoj_12_cycles_v%d"


n_cycles = 12
n_rep = 4

for it_cycle in range(1, n_cycles + 1):


    for it_rep in range(n_rep):

        
        id_name = id_name_format%(it_rep)
        chkp_path = chkp_path_format%(it_rep, it_cycle)

        print(id_name)
        print(chkp_path)

        args['chkp_save_path'] = chkp_path
        args['id_name'] = id_name
        args['random_seed'] = random_seed_list[it_rep]

        # Process args
        args_2 = wf.utils.load_multi_cycle_params_click(args)
        # Run evaluations
        wf.script_utils.evaluate_model(**args_2)




