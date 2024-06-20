import numpy as np
import pickle
from tqdm import tqdm
from copy import deepcopy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics


if __name__ == "__main__":

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
    Decoding_tp1 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)
    Decoding_tp2 = np.full([N_ref, N_rep, N_layers, N_contrast, N_noise, N_CW_CCW], np.nan)

    for i_ref, ref_angle in enumerate(ref_list):
        for i_rep, rep in enumerate(rep_list):
            weightspath = "../Results/" + "ref_" + str(ref_angle) + "_sf_40_dtheta_1_results_" + str(rep)
            FiringRate_path = weightspath + '/FR_pre.pkl'
            with open(FiringRate_path, 'rb') as f:
                FiringRate = pickle.load(f)
            for i_layer in range(N_layers):
                for i_contrast in range(N_contrast):
                    for i_noise in range(N_noise):
                        for i_CW_CCW in range(N_CW_CCW):
                            X_All = deepcopy(FiringRate["FR_target"][i_layer][i_contrast][i_noise][i_CW_CCW]).T
                            Y_All = deepcopy(FiringRate["FR_ref"][i_layer][i_contrast][i_noise][i_CW_CCW]).T

                            Data_All = np.concatenate((X_All.T, Y_All.T), axis=0)
                            Target_ALL = np.concatenate((np.zeros(X_All.shape[1]), np.ones(Y_All.shape[1])), axis=0)
                            clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
                            X_train, X_test, y_train, y_test = train_test_split(Data_All, Target_ALL,
                                                                                test_size=0.5, shuffle=True)
                            clf.fit(X_train, y_train)
                            y_pred = clf.predict(X_test)
                            scores = metrics.accuracy_score(y_test, y_pred)
                            Decoding_tp1[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = scores
            FiringRate_path = weightspath + '/FR_post.pkl'
            with open(FiringRate_path, 'rb') as f:
                FiringRate = pickle.load(f)
            for i_layer in range(N_layers):
                for i_contrast in range(N_contrast):
                    for i_noise in range(N_noise):
                        for i_CW_CCW in range(N_CW_CCW):
                            X_All = deepcopy(FiringRate["FR_target"][i_layer][i_contrast][i_noise][i_CW_CCW]).T
                            Y_All = deepcopy(FiringRate["FR_ref"][i_layer][i_contrast][i_noise][i_CW_CCW]).T

                            Data_All = np.concatenate((X_All.T, Y_All.T), axis=0)
                            Target_ALL = np.concatenate((np.zeros(X_All.shape[1]), np.ones(Y_All.shape[1])), axis=0)
                            clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
                            X_train, X_test, y_train, y_test = train_test_split(Data_All, Target_ALL,
                                                                                test_size=0.5, shuffle=True)
                            clf.fit(X_train, y_train)
                            y_pred = clf.predict(X_test)
                            scores = metrics.accuracy_score(y_test, y_pred)
                            Decoding_tp2[i_ref, i_rep, i_layer, i_contrast, i_noise, i_CW_CCW] = scores

    Decoding_supp = {"Decoding_tp1": Decoding_tp1, "Decoding_tp2": Decoding_tp2}
    with open("../Results/Decoding_supp.pkl", 'wb') as f:
        pickle.dump(Decoding_supp, f)


