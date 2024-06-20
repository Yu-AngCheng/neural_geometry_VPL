import numpy as np
import pickle
from scipy.stats import sem
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax

ref_list = [35, 55, 125, 145]
rep_list = np.arange(0, 10, 1)
contrast_levels = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
noise_levels = [0.005, 1, 5, 10, 15, 30, 50, 75]
N_neuron = [96, 256, 384, 384, 256]
layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
N_layers = len(layers)
FR_pre = [[None for _ in range(len(ref_list))] for _ in range(N_layers)]
FR_post = [[None for _ in range(len(ref_list))] for _ in range(N_layers)]
Variance_pre = [[None for _ in range(len(ref_list))] for _ in range(N_layers)]
Variance_post = [[None for _ in range(len(ref_list))] for _ in range(N_layers)]

for i_ref, ref_ori in enumerate(ref_list):
    dir = "../Results_all_directions/ref_"+str(ref_ori)+"_sf_40_dtheta_1_results_0/"
    file1 = "FiringRate_Pre_all_directions.pkl"
    file2 = "FiringRate_Post_all_directions.pkl"
    tmp1 = np.load(dir+file1, allow_pickle=True)["FiringRate_Pre_all_directions"]
    tmp2 = np.load(dir+file2, allow_pickle=True)["FiringRate_Post_all_directions"]
    for i_layer in range(N_layers):
        tmp_pre = gaussian_filter1d(
            tmp1[i_layer], sigma=1, axis=2, order=0, mode="wrap")
        index = tmp_pre.argmax(axis=2)
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                for k in range(index.shape[2]):
                    for l in range(index.shape[3]):
                        tmp_pre[i,j, :, k,l] = np.roll(tmp_pre[i,j, :, k,l], 90 - index[i][j][k][l], axis=0)
        tmp_post = gaussian_filter1d(
            tmp2[i_layer], sigma=1, axis=2, order=0, mode="wrap")
        index = tmp_post.argmax(axis=2)
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                for k in range(index.shape[2]):
                    for l in range(index.shape[3]):
                        tmp_post[i, j, :, k, l] = np.roll(tmp_post[i, j, :, k, l], 90 - index[i][j][k][l], axis=0)
        Variance_pre[i_layer][i_ref] = np.var(tmp_pre, axis=3)
        Variance_post[i_layer][i_ref] = np.var(tmp_post, axis=3)
        FR_pre[i_layer][i_ref] = np.mean(tmp_pre, axis=3)
        FR_post[i_layer][i_ref] = np.mean(tmp_post, axis=3)


for i_layer in range(N_layers):
    FR_pre[i_layer] = np.array(FR_pre[i_layer])
    FR_post[i_layer] = np.array(FR_post[i_layer])
    Variance_pre[i_layer] = np.array(Variance_pre[i_layer])
    Variance_post[i_layer] = np.array(Variance_post[i_layer])
del tmp1, tmp2


FanoFactor_property_Pre = [[[[None for _ in range(len(noise_levels))] for _ in range(len(contrast_levels))]for _ in range(len(ref_list))] for _ in range(N_layers)]
FanoFactor_property_Post = [[[[None for _ in range(len(noise_levels))] for _ in range(len(contrast_levels))]for _ in range(len(ref_list))] for _ in range(N_layers)]

for i_layer in range(N_layers):
    for i_ref, _ in enumerate(ref_list):
            for i_contrast, _ in enumerate(contrast_levels):
                for i_noise, _ in enumerate(noise_levels):
                    idx = np.logical_and(FR_pre[i_layer][i_ref][i_contrast][i_noise].min(axis=0)> 0.001,
                                         FR_post[i_layer][i_ref][i_contrast][i_noise].min(axis=0) > 0.001)
                    FR_pre_effective = FR_pre[i_layer][i_ref][i_contrast][i_noise][:, idx]
                    FR_post_effective = FR_post[i_layer][i_ref][i_contrast][i_noise][:, idx]
                    Variance_pre_effective = Variance_pre[i_layer][i_ref][i_contrast][i_noise][:, idx]
                    Variance_post_effective = Variance_post[i_layer][i_ref][i_contrast][i_noise][:, idx]

                    FanoFactor_property_Pre[i_layer][i_ref][i_contrast][i_noise] = np.mean(Variance_pre_effective/FR_pre_effective,axis=1)
                    FanoFactor_property_Post[i_layer][i_ref][i_contrast][i_noise] = np.mean(Variance_post_effective/FR_post_effective,axis=1)
    FanoFactor_property_Pre[i_layer] = np.array(FanoFactor_property_Pre[i_layer])
    FanoFactor_property_Post[i_layer] = np.array(FanoFactor_property_Post[i_layer])

Tuning = {"FanoFactor_property_Pre": FanoFactor_property_Pre, "FanoFactor_property_Post": FanoFactor_property_Post}
with open("../Results/FanoFactor_property.pkl", 'wb') as g:
    pickle.dump(Tuning, g)
