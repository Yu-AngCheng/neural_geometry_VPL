import pickle
import numpy as np
from copy import deepcopy
from tqdm import tqdm

def LFI_calculator_full(df, cov, target_sep):
    temp_LFI = np.squeeze(
        df.reshape(1, -1) @
        np.linalg.inv(cov) @
        df.reshape(-1, 1) /
        df.shape[0])
    return temp_LFI / (np.deg2rad(target_sep) ** 2)

def PCA_LFI_calculator(df_pre, df_post, cov_pre, cov_post):

    eig_val_pre, eig_vec_pre = np.linalg.eigh(cov_pre)
    eig_pairs_pre = [(np.abs(eig_val_pre[i]), eig_vec_pre[:, i]) for i in range(len(eig_val_pre))]
    eig_pairs_pre.sort(reverse=True)
    eig_val_post, eig_vec_post = np.linalg.eigh(cov_post)
    eig_pairs_post = [(np.abs(eig_val_post[i]), eig_vec_post[:, i]) for i in range(len(eig_val_post))]
    eig_pairs_post.sort(reverse=True)

    assert df_pre.size == df_post.size
    N = df_pre.size

    df_abs_pre = np.linalg.norm(df_pre)
    unit_df_pre = df_pre / np.linalg.norm(df_pre)
    df_abs_post = np.linalg.norm(df_post)
    unit_df_post = df_post / np.linalg.norm(df_post)
    lambda_sum_pre = eig_val_pre.sum()
    lambda_sum_post = eig_val_post.sum()
    unit_eig_val_pre = eig_val_pre / eig_val_pre.sum()
    unit_eig_val_post = eig_val_post / eig_val_post.sum()


    temp0 = df_abs_pre ** 2 / (N * lambda_sum_pre) * (
            unit_df_pre.reshape(1, -1) @
            eig_vec_pre @ np.diag(1 / unit_eig_val_pre) @ eig_vec_pre.T
            @ unit_df_pre.reshape(-1, 1))
    temp1 = df_abs_post ** 2 / (N * lambda_sum_pre) * (
            unit_df_pre.reshape(1, -1) @
            eig_vec_pre @ np.diag(1 / unit_eig_val_pre) @ eig_vec_pre.T
            @ unit_df_pre.reshape(-1, 1))
    temp2 = df_abs_post ** 2 / (N * lambda_sum_post) * (
            unit_df_pre.reshape(1, -1) @
            eig_vec_pre @ np.diag(1 / unit_eig_val_pre) @ eig_vec_pre.T
            @ unit_df_pre.reshape(-1, 1))
    temp3 = df_abs_post ** 2 / (N * lambda_sum_post) * (
            unit_df_pre.reshape(1, -1) @
            eig_vec_post @ np.diag(1 / unit_eig_val_post) @ eig_vec_post.T
            @ unit_df_pre.reshape(-1, 1))
    temp4 = df_abs_post ** 2 / (N * lambda_sum_post) * (
            unit_df_post.reshape(1, -1) @
            eig_vec_post @ np.diag(1 / unit_eig_val_post) @ eig_vec_post.T
            @ unit_df_post.reshape(-1, 1))

    temp0 = temp0.flatten()
    temp1 = temp1.flatten()
    temp2 = temp2.flatten()
    temp3 = temp3.flatten()
    temp4 = temp4.flatten()
    temp = np.hstack(
        (np.log10(temp1) - np.log10(temp0),
         np.log10(temp2) - np.log10(temp0),
         np.log10(temp3) - np.log10(temp0),
         np.log10(temp4) - np.log10(temp0)))

    return temp


def wrap(x):
    if x > 90:
        return np.array(180 - x)
    else:
        return np.array(x)


def angle_calculator(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product) / np.pi * 180
    return angle


ref_list = [35, 55, 125, 145]
rep_list = np.arange(0, 10, 1)
contrast_levels = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
noise_levels = [0.005, 1, 5, 10, 15, 30, 50, 75]
layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
CW_CCW = ["CW", "CCW"]
eigenpicked = [50, 150, 250, 250, 25]
N_ref = len(ref_list)
N_rep = len(rep_list)
N_contrast = len(contrast_levels)
N_noise = len(noise_levels)
N_layers = len(layers)
N_CW_CCW = len(CW_CCW)

df_abs_tp1 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
df_abs_tp2 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
variance_median_tp1 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
variance_median_tp2 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
FanoFactor_median_tp1 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
FanoFactor_median_tp2 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
LFI_tp1 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
LFI_tp2 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
Decoding_tp1 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
Decoding_tp2 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
NoiseCorrelation_median_tp1 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
NoiseCorrelation_median_tp2 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
rotation_tp1_tp2 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
PCA_LFI = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW, 4], np.nan)
N_Neuron = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
PCA_rotation = [[[[[[[] for _ in range(N_CW_CCW)] for _ in range(N_noise)] for _ in range(N_contrast)] for _ in range(N_ref)] for _ in range(N_rep)] for _ in range(N_layers)]
PC_abs_pre = [[[[[[[] for _ in range(N_CW_CCW)] for _ in range(N_noise)] for _ in range(N_contrast)] for _ in range(N_ref)] for _ in range(N_rep)] for _ in range(N_layers)]
PC_abs_post = [[[[[[[] for _ in range(N_CW_CCW)] for _ in range(N_noise)] for _ in range(N_contrast)] for _ in range(N_ref)] for _ in range(N_rep)] for _ in range(N_layers)]


with tqdm(total=35200) as pbar:
    for i_ref, ref_angle in enumerate(ref_list):
            for i_rep, rep in enumerate(rep_list):
                weightspath = "../Results/" + "ref_" + str(ref_angle) + "_sf_40_dtheta_1_results_" + str(rep)
                LFI_Component_tp1_path = weightspath + "/LFI_Component_pre.pkl"
                LFI_Component_tp2_path = weightspath + "/LFI_Component_post.pkl"
                with open(LFI_Component_tp1_path, 'rb') as f1:
                    LFI_Component_tp1 = pickle.load(f1)
                with open(LFI_Component_tp2_path, 'rb') as f2:
                    LFI_Component_tp2 = pickle.load(f2)
                for i_layer in range(N_layers):
                    for i_contrast in range(N_contrast):
                        for i_noise in range(N_noise):
                            for i_CW_CCW in range(N_CW_CCW):
                                LFI_Component_tp1_temp = deepcopy(LFI_Component_tp1[i_layer][i_contrast][i_noise][i_CW_CCW])
                                LFI_Component_tp2_temp = deepcopy(LFI_Component_tp2[i_layer][i_contrast][i_noise][i_CW_CCW])
                                idx = np.logical_and(LFI_Component_tp1_temp['fr'] > 0.001,
                                                     LFI_Component_tp2_temp['fr'] > 0.001)

                                N_Neuron[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = idx.sum()
                                Decoding_tp1[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = LFI_Component_tp1_temp['decoding']
                                Decoding_tp2[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = LFI_Component_tp2_temp['decoding']
                                fr_tp1 = LFI_Component_tp1_temp['fr'][idx]
                                fr_tp2 = LFI_Component_tp2_temp['fr'][idx]
                                df_tp1 = LFI_Component_tp1_temp['df'][idx]
                                df_tp2 = LFI_Component_tp2_temp['df'][idx]
                                variance_tp1 = LFI_Component_tp1_temp['variance'][idx]
                                variance_tp2 = LFI_Component_tp2_temp['variance'][idx]
                                corr_tp1 = LFI_Component_tp1_temp['corr'][idx, :][:, idx]
                                corr_tp2 = LFI_Component_tp2_temp['corr'][idx, :][:, idx]
                                cov_tp1 = LFI_Component_tp1_temp['cov'][idx, :][:, idx]
                                cov_tp2 = LFI_Component_tp2_temp['cov'][idx, :][:, idx]

                                variance_median_tp1[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = np.median(variance_tp1)
                                variance_median_tp2[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = np.median(variance_tp2)
                                FanoFactor_median_tp1[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = np.median(variance_tp1 / fr_tp1)
                                FanoFactor_median_tp2[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = np.median(variance_tp2 / fr_tp2)
                                df_abs_tp1[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = np.linalg.norm(df_tp1, axis=0)
                                df_abs_tp2[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = np.linalg.norm(df_tp2, axis=0)
                                rotation_tp1_tp2[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = wrap(angle_calculator(df_tp1, df_tp2))
                                upper_corr_tp1 = corr_tp1[np.triu_indices(n=corr_tp1.shape[0], k=1)]
                                upper_corr_tp2 = corr_tp2[np.triu_indices(n=corr_tp2.shape[0], k=1)]
                                NoiseCorrelation_median_tp1[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = np.median(upper_corr_tp1)
                                NoiseCorrelation_median_tp2[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = np.median(upper_corr_tp2)
                                LFI_tp1[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = LFI_calculator_full(df_tp1, cov_tp1,
                                                                                              target_sep=1)
                                LFI_tp2[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = LFI_calculator_full(df_tp2, cov_tp2,
                                                                                              target_sep=1)
                                PCA_LFI[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW,:] = PCA_LFI_calculator(df_tp1, df_tp2, cov_tp1, cov_tp2)
                                pbar.update(1)

Metrics_All = {'df_abs_tp1': df_abs_tp1,
               'df_abs_tp2': df_abs_tp2,
               'variance_median_tp1': variance_median_tp1,
               'variance_median_tp2': variance_median_tp2,
               'FanoFactor_median_tp1': FanoFactor_median_tp1,
               'FanoFactor_median_tp2': FanoFactor_median_tp2,
               'NoiseCorrelation_median_tp1': NoiseCorrelation_median_tp1,
               'NoiseCorrelation_median_tp2': NoiseCorrelation_median_tp2,
               'LFI_tp1': LFI_tp1,
               'LFI_tp2': LFI_tp2,
               'rotation_tp1_tp2': rotation_tp1_tp2,
               'PCA_LFI': PCA_LFI,
               'Decoding_tp1':Decoding_tp1,
               'Decoding_tp2':Decoding_tp2,
               'N_Neuron': N_Neuron,
               'PCA_rotation': PCA_rotation,
               'PC_abs_pre': PC_abs_pre,
               'PC_abs_post': PC_abs_post}
Metrics_All_name = "../Results/Metrics_All_Pre_Post_new.pkl"
with open(Metrics_All_name, 'wb') as g:
    pickle.dump(Metrics_All, g)