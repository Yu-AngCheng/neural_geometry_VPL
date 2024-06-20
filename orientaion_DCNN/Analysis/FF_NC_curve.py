import numpy as np
import pickle

ref_list = [35, 55, 125, 145]
contrast_levels = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
noise_levels = [0.005, 1, 5, 10, 15, 30, 50, 75]
layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
CW_CCW = ["CW", "CCW"]
checkpoints = ["/LFI_Component_" + str(x) + ".pkl"
               for x in ['pre', 'VPL_49', 'VPL_99', 'VPL_149' ,'VPL_199',
                         'VPL_249', 'VPL_299', 'VPL_349', 'VPL_399',
                         'VPL_449', 'post']]
N_checkpoints = len(checkpoints)
N_contrast_levels = len(contrast_levels)
N_noise_levels = len(noise_levels)
N_ref = len(ref_list)
N_layers = len(layers)
N_CW_CCW = len(CW_CCW)

FF_curve = [[[[[[None for _ in range(N_CW_CCW)]  for _ in range(N_noise_levels)] for _ in range(N_contrast_levels)]
             for _ in range(N_checkpoints)] for _ in range(N_ref)] for _ in range(N_layers)]
NC_curve = [[[[[[None for _ in range(N_CW_CCW)] for _ in range(N_noise_levels)] for _ in range(N_contrast_levels)]
             for _ in range(N_checkpoints)] for _ in range(N_ref)]for _ in range(N_layers)]

for i_ori, ref_ori in enumerate(ref_list):

    weightspath = "../Results/ref_" + str(ref_ori) + "_sf_40_dtheta_1_results_0"
    with open(weightspath+ "/LFI_Component_pre.pkl", 'rb') as f1:
        LFI_Component_pre = pickle.load(f1)
    with open(weightspath+ "/LFI_Component_post.pkl", 'rb') as f2:
        LFI_Component_post = pickle.load(f2)
    for tp in range(N_checkpoints):
        with open(weightspath + checkpoints[tp], 'rb') as f:
            LFI_Component = pickle.load(f)
        for i_layer in range(N_layers):
            for i_contrast, contrast in enumerate(contrast_levels):
                for i_noise, noise in enumerate(noise_levels):
                    for i_CW_CCW in range(N_CW_CCW):
                        idx = np.logical_and(LFI_Component_pre[i_layer][i_contrast][i_noise][i_CW_CCW]['fr'] > 0.001,
                                             LFI_Component_post[i_layer][i_contrast][i_noise][i_CW_CCW]['fr'] > 0.001)
                        fr_tp1 = LFI_Component[i_layer][i_contrast][i_noise][i_CW_CCW]['fr'][idx]
                        df_tp1 = LFI_Component[i_layer][i_contrast][i_noise][i_CW_CCW]['df'][idx]
                        variance_tp1 = LFI_Component[i_layer][i_contrast][i_noise][i_CW_CCW]['variance'][idx]
                        corr_tp1 = LFI_Component[i_layer][i_contrast][i_noise][i_CW_CCW]['corr'][idx, :][:, idx]
                        cov_tp1 = LFI_Component[i_layer][i_contrast][i_noise][i_CW_CCW]['cov'][idx, :][:, idx]
                        FanoFactor_median_tp1 = np.nanmedian(variance_tp1 / fr_tp1)
                        upper_corr_tp1 = corr_tp1[np.triu_indices(n=corr_tp1.shape[0], k=1)]
                        NoiseCorrelation_median_tp1 = np.nanmedian(upper_corr_tp1)
                        FF_curve[i_layer][i_ori][tp][i_contrast][i_noise][i_CW_CCW] = FanoFactor_median_tp1
                        NC_curve[i_layer][i_ori][tp][i_contrast][i_noise][i_CW_CCW] = NoiseCorrelation_median_tp1

with open("../Results/FF_NC_curve.pkl", 'wb') as g:
    pickle.dump({"FF_curve":np.array(FF_curve), "NC_curve":np.array(NC_curve)}, g)

