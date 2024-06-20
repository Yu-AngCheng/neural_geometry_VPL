import numpy as np
import pickle
from scipy.stats import sem
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax
from scipy.optimize import curve_fit


def gaussian_function(x, a, mu, sigma, d):
    return a * np.exp(-np.deg2rad(x - mu)**2 / (2 * np.deg2rad(sigma) ** 2)) + d

ref_list = [35, 55, 125, 145]
rep_list = np.arange(0, 10, 1)
contrast_levels = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
noise_levels = [0.005, 1, 5, 10, 15, 30, 50, 75]
N_neuron = [96, 256, 384, 384, 256]
layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
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
        index = tmp_pre.argmax(axis=2)
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                for k in range(index.shape[2]):
                    for l in range(index.shape[3]):
                        tmp_pre[i,j, :, k,l] = np.roll(tmp_pre[i,j, :, k,l], 90 - index[i][j][k][l], axis=0)
        tmp_post = gaussian_filter1d(
            tmp2[i_layer], sigma=1.5, axis=2, order=0, mode="wrap")
        index = tmp_post.argmax(axis=2)
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                for k in range(index.shape[2]):
                    for l in range(index.shape[3]):
                        tmp_post[i, j, :, k, l] = np.roll(tmp_post[i, j, :, k, l], 90 - index[i][j][k][l], axis=0)
        FR_pre[i_layer][i_ref] = tmp_pre.mean(axis=3)
        FR_post[i_layer][i_ref] = tmp_post.mean(axis=3)
for i_layer in range(N_layers):
    FR_pre[i_layer] = np.array(FR_pre[i_layer])
    FR_post[i_layer] = np.array(FR_post[i_layer])
del tmp1, tmp2

Tuning_Pre = [[[[None for _ in range(len(noise_levels))] for _ in range(len(contrast_levels))]for _ in range(len(ref_list))] for _ in range(N_layers)]
Tuning_Post = [[[[None for _ in range(len(noise_levels))] for _ in range(len(contrast_levels))]for _ in range(len(ref_list))] for _ in range(N_layers)]
for i_layer in range(N_layers):
    for i_ref, _ in enumerate(ref_list):
            for i_contrast, _ in enumerate(contrast_levels):
                for i_noise, _ in enumerate(noise_levels):
                    idx1 = FR_pre[i_layer][i_ref][i_contrast][i_noise].mean(axis=0)> 0.001
                    idx2 = FR_post[i_layer][i_ref][i_contrast][i_noise].mean(axis=0) > 0.001
                    idx = np.logical_and(idx1, idx2)
                    FR_pre_effective = FR_pre[i_layer][i_ref][i_contrast][i_noise][:, idx]
                    FR_post_effective = FR_post[i_layer][i_ref][i_contrast][i_noise][:, idx]
                    for i_neuron in range(idx.sum()):
                        FR_pre_effective[:, i_neuron] = gaussian_filter1d(FR_pre_effective[:,i_neuron],
                                                                           sigma=10, axis=0, order=0, mode="wrap")
                        index = FR_pre_effective[:,i_neuron].argmax()
                        FR_pre_effective[:,i_neuron] = np.roll(FR_pre_effective[:,i_neuron], 90-index, axis=0)
                        FR_pre_effective[:, i_neuron] = (FR_pre_effective[:, i_neuron])  / np.max(FR_pre_effective[:, i_neuron])
                        FR_post_effective[:,i_neuron] = gaussian_filter1d(FR_post_effective[:,i_neuron],
                                                                           sigma=10, axis=0, order=0, mode="wrap")
                        index = FR_post_effective[:,i_neuron].argmax()
                        FR_post_effective[:,i_neuron] = np.roll(FR_post_effective[:,i_neuron], 90-index, axis=0)
                        FR_post_effective[:, i_neuron] = ((FR_post_effective[:, i_neuron]) / (np.max(FR_post_effective[:, i_neuron])))
                    Tuning_Pre[i_layer][i_ref][i_contrast][i_noise] = FR_pre_effective.mean(axis=1)
                    Tuning_Post[i_layer][i_ref][i_contrast][i_noise] = FR_post_effective.mean(axis=1)
    Tuning_Pre[i_layer] = np.array(Tuning_Pre[i_layer])
    Tuning_Post[i_layer] = np.array(Tuning_Post[i_layer])
for i_layer in range(N_layers):
    for i_ref in range(len(ref_list)):
        for i_contrast, _ in enumerate(contrast_levels):
            for i_noise, _ in enumerate(noise_levels):
                popt, _ = curve_fit(gaussian_function, np.arange(-90, 90, 1), Tuning_Pre[i_layer][i_ref][i_contrast][i_noise])
                Tuning_Pre[i_layer][i_ref][i_contrast][i_noise] =gaussian_function(np.arange(-90,90,1), *popt)
                popt, _ = curve_fit(gaussian_function, np.arange(-90, 90, 1), Tuning_Post[i_layer][i_ref][i_contrast][i_noise])
                Tuning_Post[i_layer][i_ref][i_contrast][i_noise] =gaussian_function(np.arange(-90,90,1), *popt)
                Tuning_Pre[i_layer][i_ref][i_contrast][i_noise] = ((Tuning_Pre[i_layer][i_ref][i_contrast][i_noise] -
                                                                    Tuning_Pre[i_layer][i_ref][i_contrast][i_noise].min()) /
                                                                    (Tuning_Pre[i_layer][i_ref][i_contrast][i_noise].max() -
                                                                     Tuning_Pre[i_layer][i_ref][i_contrast][i_noise].min()))
                Tuning_Post[i_layer][i_ref][i_contrast][i_noise] = ((Tuning_Post[i_layer][i_ref][i_contrast][i_noise] -
                                                                     Tuning_Post[i_layer][i_ref][i_contrast][i_noise].min()) /
                                                                    (Tuning_Post[i_layer][i_ref][i_contrast][i_noise].max() -
                                                                     Tuning_Post[i_layer][i_ref][i_contrast][i_noise].min()))
Tuning = {"Tuning_Pre": Tuning_Pre, "Tuning_Post": Tuning_Post}
with open("../Results/Orientation_Tuning.pkl", 'wb') as g:
    pickle.dump(Tuning, g)
