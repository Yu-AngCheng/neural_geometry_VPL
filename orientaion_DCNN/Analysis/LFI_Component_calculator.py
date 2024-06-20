import numpy as np
import warnings
import pickle
from tqdm import tqdm
from copy import deepcopy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import argparse


def LFI_Component_calculator(ref_angle, sf, target_sep, repetition, tp_str):
    assert isinstance(tp_str, str)
    weightspath = "../Results/" + "ref_" + str(ref_angle) + "_sf_" + \
                  str(sf) + "_dtheta_" + str(target_sep) + "_results_" + \
                  str(repetition)
    FiringRate_path = weightspath + '/FR_' + tp_str + '.pkl'
    with open(FiringRate_path, 'rb') as f:
        FiringRate = pickle.load(f)

    contrast_levels = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    noise_levels = [0.005, 1, 5, 10, 15, 30, 50, 75]
    layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]
    CW_CCW = ["CW", "CCW"]
    N_contrast = len(contrast_levels)
    N_noise = len(noise_levels)
    N_layers = len(layers)
    N_CW_CCW = len(CW_CCW)

    LFI_Component = [[[[None for _ in range(N_CW_CCW)] for _ in range(N_noise)] for _ in range(N_contrast)] for _ in range(N_layers)]

    with tqdm(total=880) as pbar:
        for i_layer in range(N_layers):
            for i_contrast in range(N_contrast):
                for i_noise in range(N_noise):
                    for i_CW_CCW in range(N_CW_CCW):
                        X_All = deepcopy(FiringRate["FR_target"][i_layer][i_contrast][i_noise][i_CW_CCW]).T
                        Y_All = deepcopy(FiringRate["FR_ref"][i_layer][i_contrast][i_noise][i_CW_CCW]).T

                        Data_All = np.concatenate((X_All.T, Y_All.T), axis=0)
                        Target_ALL = np.concatenate((np.zeros(X_All.shape[1]), np.ones(Y_All.shape[1])), axis=0)
                        clf = make_pipeline(StandardScaler(), SVC(kernel='linear', cache_size=1000))
                        X_train, X_test, y_train, y_test = train_test_split(Data_All, Target_ALL,
                                                                            test_size=0.5, shuffle=True)
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        scores = metrics.accuracy_score(y_test, y_pred)

                        fr_X = X_All.mean(axis=1)
                        fr_Y = Y_All.mean(axis=1)
                        fr = (fr_X + fr_Y) / 2
                        fr_df = (fr_X - fr_Y) / target_sep
                        fr_cov_X = np.cov(X_All)
                        fr_cov_Y = np.cov(Y_All)
                        fr_cov = (fr_cov_X + fr_cov_Y) / 2
                        fr_variance = np.diag(fr_cov)
                        fr_sigma = np.sqrt(fr_variance)

                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
                            fr_corr = (fr_cov /
                                       fr_sigma[:, None] /
                                       fr_sigma[None, :])

                        # fr_sigma = np.diag(fr_sigma) # just to turn the array into a matrix

                        # fr_upper_corr = fr_corr[np.triu_indices(n=fr_corr.shape[0], k=1)]
                        # fr_df_abs = np.linalg.norm(fr_df, axis=0)
                        # fr_LFI = LFI_calculator(fr_df, fr_cov)

                        temp_LFI_Component = dict()
                        temp_LFI_Component['df'] = fr_df
                        temp_LFI_Component['fr'] = fr
                        temp_LFI_Component['cov'] = fr_cov
                        temp_LFI_Component['variance'] = fr_variance
                        temp_LFI_Component['corr'] = fr_corr
                        temp_LFI_Component['decoding'] = scores
                        # temp_LFI_Component['sigma'] = fr_sigma
                        # temp_LFI_Component['df_abs'] = fr_df_abs
                        # temp_LFI_Component['upper_corr'] = fr_upper_corr
                        # temp_LFI_Component['LFI'] = fr_LFI
                        LFI_Component[i_layer][i_contrast][i_noise][i_CW_CCW] = temp_LFI_Component
                        pbar.update(1)
                        del X_All, Y_All, fr_X, fr_Y, fr_df, fr_cov_X, fr_cov_Y, fr_cov, \
                            fr_variance, fr_sigma, fr_corr, fr, scores, temp_LFI_Component

    LFI_Component_name = weightspath + "/LFI_Component_"+tp_str+".pkl"
    with open(LFI_Component_name, 'wb') as g:
        pickle.dump(LFI_Component, g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Denoise simplified')
    parser.add_argument('--repetition', '-r', type=int)
    args = parser.parse_args()
    for ref_angle in [35, 55, 125, 145]:
        for sf in [40]:
            for target_sep in [1]:
                LFI_Component_calculator(ref_angle, sf, target_sep, args.repetition, tp_str="pre")
                LFI_Component_calculator(ref_angle, sf, target_sep, args.repetition, tp_str="post")
