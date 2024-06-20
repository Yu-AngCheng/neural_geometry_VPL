from preprocess import preprocess
from LFI_Component_calculator import LFI_Component_calculator
import numpy as np
from scipy.stats import ttest_rel

from PCA_LFI_calculator import PCA_LFI_calculator
import os
import pickle


def angle_calculator(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product) / np.pi * 180
    return angle

def wrap(x):
    if x > 90:
        return np.array(180 - x)
    else:
        return np.array(x)

def Metrics():
    subjects = 22
    regions = 7
    alpha = 0.05

    voxel_response = preprocess()
    LFI_Component = LFI_Component_calculator(voxel_response)

    df_abs_pre = np.full([subjects, regions], np.nan)
    df_abs_post = np.full([subjects, regions], np.nan)
    variance_pre = np.full([subjects, regions], np.nan)
    variance_post = np.full([subjects, regions], np.nan)
    LFI_pre = np.full([subjects, regions], np.nan)
    LFI_post = np.full([subjects, regions], np.nan)
    correlation_pre = np.full([subjects, regions], np.nan)
    correlation_post = np.full([subjects, regions], np.nan)
    decoding_acc_pre = np.full([subjects, regions], np.nan)
    decoding_acc_post = np.full([subjects, regions], np.nan)
    rotation = np.full([subjects, regions], np.nan)
    signal_strength_pre = np.full([subjects, regions], np.nan)
    signal_strength_post = np.full([subjects, regions], np.nan)
    noise_fluctuations_pre = np.full([subjects, regions], np.nan)
    noise_fluctuations_post = np.full([subjects, regions], np.nan)
    PCA_LFI = [[None for _ in range(regions)] for _ in range(subjects)]
    PCA_rotation = [[[] for _ in range(regions)] for _ in range(subjects)]
    PC_abs_pre = [[[] for _ in range(regions)] for _ in range(subjects)]
    PC_abs_post = [[[] for _ in range(regions)] for _ in range(subjects)]


    for i_sub in range(subjects):
        for i_reg in range(regions):
            df_abs_pre[i_sub][i_reg] = LFI_Component[0][i_sub][i_reg]['df_abs']
            df_abs_post[i_sub][i_reg] = LFI_Component[1][i_sub][i_reg]['df_abs']
            LFI_pre[i_sub][i_reg] = LFI_Component[0][i_sub][i_reg]['LFI']
            LFI_post[i_sub][i_reg] = LFI_Component[1][i_sub][i_reg]['LFI']
            decoding_acc_pre[i_sub][i_reg] = LFI_Component[0][i_sub][i_reg]['decoding_acc']
            decoding_acc_post[i_sub][i_reg] = LFI_Component[1][i_sub][i_reg]['decoding_acc']
            correlation_pre[i_sub][i_reg] = np.median(LFI_Component[0][i_sub][i_reg]['upper_corr'])
            correlation_post[i_sub][i_reg] = np.median(LFI_Component[1][i_sub][i_reg]['upper_corr'])
            variance_pre[i_sub][i_reg] = np.median(LFI_Component[0][i_sub][i_reg]['variance'])
            variance_post[i_sub][i_reg] = np.median(LFI_Component[1][i_sub][i_reg]['variance'])
            signal_strength_pre[i_sub][i_reg] = LFI_Component[0][i_sub][i_reg]['signal_strength']
            signal_strength_post[i_sub][i_reg] = LFI_Component[1][i_sub][i_reg]['signal_strength']
            noise_fluctuations_pre[i_sub][i_reg] = LFI_Component[0][i_sub][i_reg]['noise_fluctuation']
            noise_fluctuations_post[i_sub][i_reg] = LFI_Component[1][i_sub][i_reg]['noise_fluctuation']
            rotation[i_sub][i_reg] = wrap(angle_calculator(LFI_Component[0][i_sub][i_reg]['df'],
                                                           LFI_Component[1][i_sub][i_reg]['df']))
            PCA_LFI[i_sub][i_reg], eig_pairs_pre, eig_pairs_post = PCA_LFI_calculator(
                LFI_Component[0][i_sub][i_reg]['df'],
                LFI_Component[1][i_sub][i_reg]['df'],
                LFI_Component[0][i_sub][i_reg]['cov'],
                LFI_Component[1][i_sub][i_reg]['cov'])

            assert len(eig_pairs_pre) == len(eig_pairs_post)
            for i in range(len(eig_pairs_pre)):
                PCA_rotation[i_sub][i_reg].append(wrap(angle_calculator(eig_pairs_pre[i][1], eig_pairs_post[i][1])))
                PC_abs_pre[i_sub][i_reg].append(eig_pairs_pre[i][0])
                PC_abs_post[i_sub][i_reg].append(eig_pairs_post[i][0])

    PCA_LFI = np.array(PCA_LFI)
    Metrics = {'df_abs_pre': df_abs_pre, 'df_abs_post': df_abs_post, 'LFI_pre': LFI_pre, 'LFI_post': LFI_post,
               'signal_strength_pre': signal_strength_pre, 'signal_strength_post': signal_strength_post,
               'noise_fluctuations_pre': noise_fluctuations_pre, 'noise_fluctuations_post': noise_fluctuations_post,
               'decoding_acc_pre': decoding_acc_pre, 'decoding_acc_post': decoding_acc_post,
               'correlation_pre': correlation_pre, 'correlation_post': correlation_post,
               'variance_pre': variance_pre, 'variance_post': variance_post, 'rotation': rotation, 'PCA_LFI': PCA_LFI,
               "PCA_rotation": PCA_rotation, "PC_abs_pre": PC_abs_pre, "PC_abs_post": PC_abs_post}

    with open('Metrics_fMRI.pkl', 'wb') as f:
        pickle.dump(Metrics, f)