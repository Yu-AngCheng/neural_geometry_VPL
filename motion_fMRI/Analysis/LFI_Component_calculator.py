import numpy as np
from copy import deepcopy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

subjects = 22
regions = 7
trials = 120
smalltr = 30
tps = 2


def LFI_calculator(df, cov):
    temp_LFI = np.squeeze(
        df.reshape(1, -1) @
        np.linalg.inv(cov) @
        df.reshape(-1, 1))
    return (temp_LFI) / df.shape[0]


def Decoding(data, label, groups):
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    cv = LeaveOneOut()
    scores = cross_val_score(clf, data, label, groups=groups,
                             scoring='accuracy', cv=cv, n_jobs=-1)
    score = np.array(scores).mean()
    return score


def LFI_Component_calculator(voxel_response):

    LFI_Component = [[[None for _ in range(regions)] for _ in range(subjects)] for _ in range(tps)]

    for t in range(tps):
        for i_sub in range(subjects):
            for i_reg in range(regions):
                X_All = deepcopy(voxel_response[t][0][i_sub][i_reg])
                Y_All = deepcopy(voxel_response[t][1][i_sub][i_reg])

                data = np.concatenate((X_All.T, Y_All.T), axis=0)
                label = np.concatenate((np.zeros(trials), np.ones(trials)), axis=0)
                groups = np.concatenate((np.zeros(smalltr), np.ones(smalltr),
                                         2 * np.ones(smalltr), 3 * np.ones(smalltr),
                                         np.zeros(smalltr), np.ones(smalltr),
                                         2 * np.ones(smalltr), 3 * np.ones(smalltr)
                                         ), axis=0)

                beta_decoding_acc = Decoding(data, label, groups)

                clf = LinearDiscriminantAnalysis()
                Data_projected = clf.fit_transform(data, label)
                X_projected = Data_projected[label == 0] / np.linalg.norm(clf.scalings_)
                Y_projected = Data_projected[label == 1] / np.linalg.norm(clf.scalings_)
                signal_strength = np.abs(np.mean(X_projected) - np.mean(Y_projected))
                noise_fluctuation = np.sqrt(np.var(X_projected) + np.var(Y_projected))

                beta_X = X_All.mean(axis=1)
                beta_Y = Y_All.mean(axis=1)
                beta_df = (beta_X - beta_Y) / np.deg2rad(90)
                beta_cov_X = np.cov(X_All)
                beta_cov_Y = np.cov(Y_All)
                beta_cov = (beta_cov_X + beta_cov_Y) / 2
                beta_variance = np.diag(beta_cov)
                beta_sigma = np.sqrt(beta_variance)
                beta_corr = (beta_cov /
                             beta_sigma[:, None] /
                             beta_sigma[None, :])
                beta = (beta_X + beta_Y) / 2
                beta_sigma = np.diag(beta_sigma)
                beta_upper_corr = beta_corr[np.triu_indices(n=beta_corr.shape[0], k=1)]
                beta_df_abs = np.linalg.norm(beta_df, axis=0)
                beta_LFI = LFI_calculator(beta_df, beta_cov)

                temp_LFI_Component = dict()
                temp_LFI_Component['df'] = beta_df
                temp_LFI_Component['sigma'] = beta_sigma
                temp_LFI_Component['corr'] = beta_corr
                temp_LFI_Component['cov'] = beta_cov
                temp_LFI_Component['beta'] = beta
                temp_LFI_Component['df_abs'] = beta_df_abs
                temp_LFI_Component['upper_corr'] = beta_upper_corr
                temp_LFI_Component['variance'] = beta_variance
                temp_LFI_Component['LFI'] = beta_LFI
                temp_LFI_Component['decoding_acc'] = beta_decoding_acc
                temp_LFI_Component['signal_strength'] = signal_strength
                temp_LFI_Component['noise_fluctuation'] = noise_fluctuation

                LFI_Component[t][i_sub][i_reg] = temp_LFI_Component

    return LFI_Component
