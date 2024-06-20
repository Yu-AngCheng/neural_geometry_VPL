import numpy as np
import pickle
from scipy.stats import sem
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax
from scipy.optimize import curve_fit

ref_list = [35, 55, 125, 145]
rep_list = np.arange(0, 10, 1)
contrast_levels = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
noise_levels = [0.005, 1, 5, 10, 15, 30, 50, 75]
N_neuron = [96, 256, 384, 384, 256]
layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
CW_CCW = ["CW", "CCW"]
N_CW_CCW = len(CW_CCW)
N_layers = len(layers)

FR_pre = [[None for _ in range(len(ref_list))] for _ in range(N_layers)]
FR_post = [[None for _ in range(len(ref_list))] for _ in range(N_layers)]
for i_ref, ref_ori in enumerate(ref_list):
    dir = "../Results_all_directions/ref_"+str(ref_ori)+"_sf_40_dtheta_1_results_0/"
    file1 = "FiringRate_Pre_all_directions.pkl"
    file2 = "FiringRate_Post_all_directions.pkl"
    tmp1 = np.load(dir+file1, allow_pickle=True)["FiringRate_Pre_all_directions"]
    tmp2 = np.load(dir+file2, allow_pickle=True)["FiringRate_Post_all_directions"]
    for i_layer in range(N_layers):
        tmp_pre = gaussian_filter1d(
            tmp1[i_layer], sigma=1.5, axis=2, order=0, mode="wrap")
        tmp_post = gaussian_filter1d(
            tmp2[i_layer], sigma=1.5, axis=2, order=0, mode="wrap")
        FR_pre[i_layer][i_ref] = tmp_pre.mean(axis=3)
        FR_post[i_layer][i_ref] = tmp_post.mean(axis=3)
for i_layer in range(N_layers):
    FR_pre[i_layer] = np.array(FR_pre[i_layer])
    FR_post[i_layer] = np.array(FR_post[i_layer])

TuningSimilarity_Pre = [[[[[None for _ in range(N_CW_CCW)] for _ in range(len(noise_levels))] for _ in range(len(contrast_levels))]for _ in range(len(ref_list))] for _ in range(N_layers)]
TuningSimilarity_Post = [[[[[None for _ in range(N_CW_CCW)] for _ in range(len(noise_levels))] for _ in range(len(contrast_levels))]for _ in range(len(ref_list))] for _ in range(N_layers)]
NoiseCorrelation_Pre = [[[[[None for _ in range(N_CW_CCW)] for _ in range(len(noise_levels))] for _ in range(len(contrast_levels))]for _ in range(len(ref_list))] for _ in range(N_layers)]
NoiseCorrelation_Post = [[[[[None for _ in range(N_CW_CCW)] for _ in range(len(noise_levels))] for _ in range(len(contrast_levels))]for _ in range(len(ref_list))] for _ in range(N_layers)]

for i_ref, ref_ori in enumerate(ref_list):

    dir = "../Results/ref_" + str(ref_ori) + "_sf_40_dtheta_1_results_0/"
    file1 = "LFI_Component_pre.pkl"
    file2 = "LFI_Component_post.pkl"
    LFI_Component_tp1 = np.load(dir + file1, allow_pickle=True)
    LFI_Component_tp2 = np.load(dir + file2, allow_pickle=True)
    for i_layer in range(N_layers):
        for i_contrast, _ in enumerate(contrast_levels):
            for i_noise, _ in enumerate(noise_levels):
                for i_CW_CCW in range(N_CW_CCW):

                    idx1 = LFI_Component_tp1[i_layer][i_contrast][i_noise][i_CW_CCW]['fr'] > 0.001
                    idx2 = LFI_Component_tp2[i_layer][i_contrast][i_noise][i_CW_CCW]['fr'] > 0.001
                    idx3 = FR_pre[i_layer][i_ref][i_contrast][i_noise].mean(axis=0)> 0.001
                    idx4 = FR_post[i_layer][i_ref][i_contrast][i_noise].mean(axis=0) > 0.001
                    idx = np.logical_and(np.logical_and(idx3, idx4), np.logical_and(idx1, idx2))

                    FR_pre_effective = FR_pre[i_layer][i_ref][i_contrast][i_noise][:, idx]
                    FR_post_effective = FR_post[i_layer][i_ref][i_contrast][i_noise][:, idx]
                    for i_neuron in range(idx.sum()):
                        FR_pre_effective[:, i_neuron] = gaussian_filter1d(FR_pre_effective[:, i_neuron],
                                                                          sigma=10, axis=0, order=0, mode="wrap")

                        FR_pre_effective[:, i_neuron] = (FR_pre_effective[:, i_neuron]) / np.max(FR_pre_effective[:, i_neuron])
                        FR_post_effective[:, i_neuron] = gaussian_filter1d(FR_post_effective[:, i_neuron],
                                                                           sigma=10, axis=0, order=0, mode="wrap")
                        FR_post_effective[:, i_neuron] = ((FR_post_effective[:, i_neuron]) / (np.max(FR_post_effective[:, i_neuron])))

                    noisecorr_tp1 = LFI_Component_tp1[i_layer][i_contrast][i_noise][i_CW_CCW]['corr'][idx, :][:, idx]
                    noisecorr_tp2 = LFI_Component_tp1[i_layer][i_contrast][i_noise][i_CW_CCW]['corr'][idx, :][:, idx]
                    signalcorr_tp1 = np.corrcoef(FR_pre_effective.T)
                    signalcorr_tp2 = np.corrcoef(FR_post_effective.T)
                    TuningSimilarity_Pre[i_layer][i_ref][i_contrast][i_noise][i_CW_CCW] = signalcorr_tp1[np.triu_indices(n=signalcorr_tp1.shape[0], k=1)]
                    TuningSimilarity_Post[i_layer][i_ref][i_contrast][i_noise][i_CW_CCW] = signalcorr_tp2[np.triu_indices(n=signalcorr_tp2.shape[0], k=1)]
                    NoiseCorrelation_Pre[i_layer][i_ref][i_contrast][i_noise][i_CW_CCW] = noisecorr_tp1[np.triu_indices(n=noisecorr_tp1.shape[0], k=1)]
                    NoiseCorrelation_Post[i_layer][i_ref][i_contrast][i_noise][i_CW_CCW] = noisecorr_tp2[np.triu_indices(n=noisecorr_tp2.shape[0], k=1)]

    TuningSimilarity_Pre[i_layer] = np.array(TuningSimilarity_Pre[i_layer])
    TuningSimilarity_Post[i_layer] = np.array(TuningSimilarity_Post[i_layer])
    NoiseCorrelation_Pre[i_layer] = np.array(NoiseCorrelation_Pre[i_layer])
    NoiseCorrelation_Post[i_layer] = np.array(NoiseCorrelation_Post[i_layer])


Signal_Noise_Correlation = {'TuningSimilarity_Pre': TuningSimilarity_Pre, 'TuningSimilarity_Post': TuningSimilarity_Post,
                            'NoiseCorrelation_Pre': NoiseCorrelation_Pre, 'NoiseCorrelation_Post': NoiseCorrelation_Post}
with open("../Results/Signal_Noise_Correlation.pkl", 'wb') as g:
    pickle.dump(Signal_Noise_Correlation, g)
