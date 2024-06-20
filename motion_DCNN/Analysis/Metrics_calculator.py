import numpy as np
import pickle
import os
from copy import deepcopy
from PCA_LFI_calculator import PCA_LFI_calculator


def LFI_calculator_full(df, cov, target_sep):
    temp_LFI = np.squeeze(
        df.reshape(1, -1) @
        np.linalg.inv(cov) @
        df.reshape(-1, 1))
    return temp_LFI / (np.deg2rad(target_sep) ** 2) / df.shape[0]


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


def Metrics_calculator(ref_direction, target_sep, repetition, tp_str_1, tp_str_2):
    assert isinstance(tp_str_1, str)
    assert isinstance(tp_str_2, str)

    weightspath = "../Results/" + "ref_direction_" + str(ref_direction) + \
                  "_target_sep_" + str(target_sep) + "_results_" + \
                  str(repetition)
    LFI_Component_tp1_path = weightspath + "/LFI_Component_" + tp_str_1 + ".pkl"
    LFI_Component_tp2_path = weightspath + "/LFI_Component_" + tp_str_2 + ".pkl"
    with open(LFI_Component_tp1_path, 'rb') as f1:
        LFI_Component_tp1 = pickle.load(f1)
    with open(LFI_Component_tp2_path, 'rb') as f2:
        LFI_Component_tp2 = pickle.load(f2)

    coherence_levels = [0.0884, 0.1250, 0.1768, 0.2500, 0.3536, 0.5000, 0.7071, 1]
    layers = ["conv1", "conv2", "conv3a", "conv3b", "conv4a", "conv4b"]
    CW_CCW = ["CW", "CCW"]
    eigenpicked = [50, 100, 200, 200, 500, 200]
    N_coherence_levels = len(coherence_levels)
    N_layers = len(layers)
    N_CW_CCW = len(CW_CCW)

    df_abs_tp1 = np.full([N_coherence_levels, N_layers, N_CW_CCW], np.nan)
    df_abs_tp2 = np.full([N_coherence_levels, N_layers, N_CW_CCW], np.nan)
    variance_median_tp1 = np.full([N_coherence_levels, N_layers, N_CW_CCW], np.nan)
    variance_median_tp2 = np.full([N_coherence_levels, N_layers, N_CW_CCW], np.nan)
    FanoFactor_median_tp1 = np.full([N_coherence_levels, N_layers, N_CW_CCW], np.nan)
    FanoFactor_median_tp2 = np.full([N_coherence_levels, N_layers, N_CW_CCW], np.nan)
    LFI_tp1 = np.full([N_coherence_levels, N_layers, N_CW_CCW], np.nan)
    LFI_tp2 = np.full([N_coherence_levels, N_layers, N_CW_CCW], np.nan)
    NoiseCorrelation_median_tp1 = np.full([N_coherence_levels, N_layers, N_CW_CCW], np.nan)
    NoiseCorrelation_median_tp2 = np.full([N_coherence_levels, N_layers, N_CW_CCW], np.nan)
    rotation_tp1_tp2 = np.full([N_coherence_levels, N_layers, N_CW_CCW], np.nan)
    PCA_LFI = [[[None for _ in range(N_CW_CCW)] for _ in range(N_layers)] for _ in range(N_coherence_levels)]
    PCA_rotation = [[[[] for _ in range(N_CW_CCW)] for _ in range(N_layers)] for _ in range(N_coherence_levels)]
    PC_abs_pre = [[[[] for _ in range(N_CW_CCW)] for _ in range(N_layers)] for _ in range(N_coherence_levels)]
    PC_abs_post = [[[[] for _ in range(N_CW_CCW)] for _ in range(N_layers)] for _ in range(N_coherence_levels)]
    N_Neuron = np.full([N_coherence_levels, N_layers, N_CW_CCW], np.nan)
    Decoding_tp1 = np.full([N_coherence_levels, N_layers, N_CW_CCW], np.nan)
    Decoding_tp2 = np.full([N_coherence_levels, N_layers, N_CW_CCW], np.nan)

    for i_coherence in range(N_coherence_levels):
        for i_layer in range(N_layers):
            for i_CW_CCW in range(N_CW_CCW):
                LFI_Component_tp1_temp = deepcopy(LFI_Component_tp1[i_coherence][i_layer][i_CW_CCW])
                LFI_Component_tp2_temp = deepcopy(LFI_Component_tp2[i_coherence][i_layer][i_CW_CCW])
                idx = np.logical_and(LFI_Component_tp1_temp['fr'] > 0.001, LFI_Component_tp2_temp['fr'] > 0.001)

                N_Neuron[i_coherence][i_layer][i_CW_CCW] = idx.sum()
                Decoding_tp1[i_coherence][i_layer][i_CW_CCW] = LFI_Component_tp1_temp['decoding']
                Decoding_tp2[i_coherence][i_layer][i_CW_CCW] = LFI_Component_tp2_temp['decoding']
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

                variance_median_tp1[i_coherence][i_layer][i_CW_CCW] = np.median(variance_tp1)
                variance_median_tp2[i_coherence][i_layer][i_CW_CCW] = np.median(variance_tp2)
                FanoFactor_median_tp1[i_coherence][i_layer][i_CW_CCW] = np.median(variance_tp1 / fr_tp1)
                FanoFactor_median_tp2[i_coherence][i_layer][i_CW_CCW] = np.median(variance_tp2 / fr_tp2)
                df_abs_tp1[i_coherence][i_layer][i_CW_CCW] = np.linalg.norm(df_tp1, axis=0)
                df_abs_tp2[i_coherence][i_layer][i_CW_CCW] = np.linalg.norm(df_tp2, axis=0)
                rotation_tp1_tp2[i_coherence][i_layer][i_CW_CCW] = wrap(angle_calculator(df_tp1, df_tp2))
                upper_corr_tp1 = corr_tp1[np.triu_indices(n=corr_tp1.shape[0], k=1)]
                upper_corr_tp2 = corr_tp2[np.triu_indices(n=corr_tp2.shape[0], k=1)]
                NoiseCorrelation_median_tp1[i_coherence][i_layer][i_CW_CCW] = np.median(upper_corr_tp1)
                NoiseCorrelation_median_tp2[i_coherence][i_layer][i_CW_CCW] = np.median(upper_corr_tp2)
                LFI_tp1[i_coherence][i_layer][i_CW_CCW] = LFI_calculator_full(df_tp1, cov_tp1, target_sep=target_sep)
                LFI_tp2[i_coherence][i_layer][i_CW_CCW] = LFI_calculator_full(df_tp2, cov_tp2, target_sep=target_sep)
                PCA_LFI[i_coherence][i_layer][i_CW_CCW], eig_pairs_pre, eig_pairs_post = PCA_LFI_calculator(df_tp1, df_tp2,
                                                                             cov_tp1, cov_tp2)
                assert len(eig_pairs_pre) == len(eig_pairs_post)
                assert len(eig_pairs_pre) >= eigenpicked[i_layer]
                for i in range(eigenpicked[i_layer]):
                    PCA_rotation[i_coherence][i_layer][i_CW_CCW].append(wrap(angle_calculator(eig_pairs_pre[i][1], eig_pairs_post[i][1])))
                    PC_abs_pre[i_coherence][i_layer][i_CW_CCW].append(eig_pairs_pre[i][0])
                    PC_abs_post[i_coherence][i_layer][i_CW_CCW].append(eig_pairs_post[i][0])

    PCA_LFI = np.array(PCA_LFI)
    PCA_rotation = np.array(PCA_rotation)
    PC_abs_pre = np.array(PC_abs_pre)
    PC_abs_post = np.array(PC_abs_post)

    Metrics = {'df_abs_tp1': df_abs_tp1,
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
               'N_Neuron': N_Neuron,
               'Decoding_tp1': Decoding_tp1,
               'Decoding_tp2': Decoding_tp2,
               'PCA_rotation': PCA_rotation,
               'PC_abs_pre': PC_abs_pre,
               'PC_abs_post': PC_abs_post
               }
    Metrics_Component_name = weightspath + "/Metrics_Component_" + tp_str_1 + \
                             "_" + tp_str_2 + ".pkl"
    with open(Metrics_Component_name, 'wb') as g:
        pickle.dump(Metrics, g)
