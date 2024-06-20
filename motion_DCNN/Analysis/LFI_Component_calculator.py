import numpy as np
import pickle
import os
from copy import deepcopy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

def LFI_Component_calculator(ref_direction, target_sep, repetition, tp_str):
    assert isinstance(tp_str, str)

    weightspath = "../Results/" + "ref_direction_" + str(ref_direction) + \
                  "_target_sep_" + str(target_sep) + "_results_" + \
                  str(repetition)
    FiringRate_path = weightspath + '/FiringRate_' + tp_str + '.pkl'
    with open(FiringRate_path, 'rb') as f:
        FiringRate = pickle.load(f)

    coherence_levels = [0.0884, 0.1250, 0.1768, 0.2500, 0.3536, 0.5000, 0.7071, 1]
    layers = ["conv1", "conv2", "conv3a", "conv3b", "conv4a", "conv4b"]
    CW_CCW = ["CW", "CCW"]
    N_coherence_levels = len(coherence_levels)
    N_layers = len(layers)
    N_CW_CCW = len(CW_CCW)

    LFI_Component = [[[None for _ in range(N_CW_CCW)] for _ in range(N_layers)] for _ in range(N_coherence_levels)]
    FiringRate_X = [[[None for _ in range(N_CW_CCW)] for _ in range(N_layers)] for _ in range(N_coherence_levels)]
    FiringRate_Y = [[[None for _ in range(N_CW_CCW)] for _ in range(N_layers)] for _ in range(N_coherence_levels)]

    for i_coherence in range(N_coherence_levels):
        for i_layer in range(N_layers):
            for i_CW_CCW in range(N_CW_CCW):
                FiringRate_X_temp = FiringRate['FiringRate_X'][i_coherence][i_layer][i_CW_CCW]
                FiringRate_Y_temp = FiringRate['FiringRate_Y'][i_coherence][i_layer][i_CW_CCW]
                FiringRate_X[i_coherence][i_layer][i_CW_CCW] = deepcopy(FiringRate_X_temp.mean(axis=1))
                FiringRate_Y[i_coherence][i_layer][i_CW_CCW] = deepcopy(FiringRate_Y_temp.mean(axis=1))

    del FiringRate

    for i_coherence in range(N_coherence_levels):
        for i_layer in range(N_layers):
            for i_CW_CCW in range(N_CW_CCW):
                X_All = deepcopy(FiringRate_X[i_coherence][i_layer][i_CW_CCW])
                Y_All = deepcopy(FiringRate_Y[i_coherence][i_layer][i_CW_CCW])

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
                fr_corr = (fr_cov /
                           fr_sigma[:, None] /
                           fr_sigma[None, :])
                fr_sigma = np.diag(fr_sigma) # just to turn the array into a matrix

                temp_LFI_Component = dict()
                temp_LFI_Component['df'] = fr_df
                temp_LFI_Component['fr'] = fr
                temp_LFI_Component['cov'] = fr_cov
                temp_LFI_Component['variance'] = fr_variance
                # temp_LFI_Component['sigma'] = fr_sigma
                temp_LFI_Component['corr'] = fr_corr
                temp_LFI_Component['decoding'] = scores
                # temp_LFI_Component['df_abs'] = fr_df_abs
                # temp_LFI_Component['upper_corr'] = fr_upper_corr
                # temp_LFI_Component['LFI'] = fr_LFI
                LFI_Component[i_coherence][i_layer][i_CW_CCW] = temp_LFI_Component

                del X_All, Y_All, fr_X, fr_Y, fr_df, fr_cov_X, fr_cov_Y, fr_cov, \
                    fr_variance, fr_sigma, fr_corr, fr, scores, temp_LFI_Component

    LFI_Component_name = weightspath + "/LFI_Component_"+tp_str+".pkl"
    with open(LFI_Component_name, 'wb') as g:
        pickle.dump(LFI_Component, g)

if __name__ == "__main__":
    LFI_Component_calculator(ref_direction=-35, target_sep=5, repetition=0, tp_str="Pre")
    LFI_Component_calculator(ref_direction=-35, target_sep=5, repetition=0, tp_str="Post")
