import os
from Metrics_calculator import Metrics_calculator
# from MotionNN.Analysis.LFI_Component_calculator import LFI_Component_calculator
import pickle
import numpy as np


def Metrics_Stats():
    direction_list = [45, -45, 135, -135]
    repetition_list = np.arange(0, 10, 1)
    N_directions = len(direction_list)
    N_repetitions = len(repetition_list)
    df_abs_tp1 = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    df_abs_tp2 = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    variance_median_tp1 = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    variance_median_tp2 = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    FanoFactor_median_tp1 = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    FanoFactor_median_tp2 = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    LFI_tp1 = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    LFI_tp2 = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    Decoding_tp1 = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    Decoding_tp2 = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    NoiseCorrelation_median_tp1 = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    NoiseCorrelation_median_tp2 = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    rotation_tp1_tp2 = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    PCA_LFI = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    N_Neuron = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    PCA_rotation = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    PC_abs_pre = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]
    PC_abs_post = [[None for _ in range(N_directions)] for _ in range(N_repetitions)]

    for i_direction, ref_direction in enumerate(direction_list):
        for target_sep in [4]:
            for i_repetition, repetition in enumerate(repetition_list):

                weightspath = "../Results/" + "ref_direction_" + str(ref_direction) + \
                              "_target_sep_" + str(target_sep) + "_results_" + \
                              str(repetition)
                # LFI_Component_calculator(ref_direction=ref_direction, target_sep=target_sep,
                #                          repetition=repetition, tp_str="Pre")
                # LFI_Component_calculator(ref_direction=ref_direction, target_sep=target_sep,
                #                          repetition=repetition, tp_str="Post")
                Metrics_calculator(ref_direction=ref_direction, target_sep=target_sep, repetition=repetition,
                                   tp_str_1="Pre", tp_str_2="Post")

                with open(weightspath + "/Metrics_Component_Pre_Post_" + ".pkl", 'rb') as f:
                    Metrics_temp = pickle.load(f)

                df_abs_tp1[i_repetition][i_direction] = Metrics_temp['df_abs_tp1']
                df_abs_tp2[i_repetition][i_direction] = Metrics_temp['df_abs_tp2']
                variance_median_tp1[i_repetition][i_direction] = Metrics_temp['variance_median_tp1']
                variance_median_tp2[i_repetition][i_direction] = Metrics_temp['variance_median_tp2']
                FanoFactor_median_tp1[i_repetition][i_direction] = Metrics_temp['FanoFactor_median_tp1']
                FanoFactor_median_tp2[i_repetition][i_direction] = Metrics_temp['FanoFactor_median_tp2']
                LFI_tp1[i_repetition][i_direction] = Metrics_temp['LFI_tp1']
                LFI_tp2[i_repetition][i_direction] = Metrics_temp['LFI_tp2']
                NoiseCorrelation_median_tp1[i_repetition][i_direction] = Metrics_temp['NoiseCorrelation_median_tp1']
                NoiseCorrelation_median_tp2[i_repetition][i_direction] = Metrics_temp['NoiseCorrelation_median_tp2']
                rotation_tp1_tp2[i_repetition][i_direction] = Metrics_temp['rotation_tp1_tp2']
                PCA_LFI[i_repetition][i_direction] = Metrics_temp['PCA_LFI']
                PCA_rotation[i_repetition][i_direction] = Metrics_temp['PCA_rotation']
                PC_abs_pre[i_repetition][i_direction] = Metrics_temp['PC_abs_pre']
                PC_abs_post[i_repetition][i_direction] = Metrics_temp['PC_abs_post']
                Decoding_tp1[i_repetition][i_direction] = Metrics_temp['Decoding_tp1']
                Decoding_tp2[i_repetition][i_direction] = Metrics_temp['Decoding_tp2']
                N_Neuron[i_repetition][i_direction] = Metrics_temp['N_Neuron']

    # the shape below are (10, 4, 8, 4, 2), (repetitions, directions, coherence, layers, CW_CCW)
    df_abs_tp1 = np.array(df_abs_tp1)
    df_abs_tp2 = np.array(df_abs_tp2)
    variance_median_tp1 = np.array(variance_median_tp1)
    variance_median_tp2 = np.array(variance_median_tp2)
    FanoFactor_median_tp1 = np.array(FanoFactor_median_tp1)
    FanoFactor_median_tp2 = np.array(FanoFactor_median_tp2)
    LFI_tp1 = np.array(LFI_tp1)
    LFI_tp2 = np.array(LFI_tp2)
    NoiseCorrelation_median_tp1 = np.array(NoiseCorrelation_median_tp1)
    NoiseCorrelation_median_tp2 = np.array(NoiseCorrelation_median_tp2)
    rotation_tp1_tp2 = np.array(rotation_tp1_tp2)
    PCA_LFI = np.array(PCA_LFI)
    PCA_rotation = np.array(PCA_rotation)
    PC_abs_pre = np.array(PC_abs_pre)
    PC_abs_post = np.array(PC_abs_post)
    N_Neuron = np.array(N_Neuron)
    Decoding_tp1 = np.array(Decoding_tp1)
    Decoding_tp2 = np.array(Decoding_tp2)

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
                   'PC_abs_post': PC_abs_post
                   }
    Metrics_All_name = "../Results/Metrics_All_Pre_Post_" + ".pkl"
    with open(Metrics_All_name, 'wb') as g:
        pickle.dump(Metrics_All, g)


if __name__ == "__main__":
    Metrics_Stats()