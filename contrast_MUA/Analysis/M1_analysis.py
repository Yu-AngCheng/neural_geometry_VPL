import numpy as np
from scipy.io import loadmat
from PCA_LFI_calculator import PCA_LFI_calculator
import pickle
import matplotlib.pyplot as plt
import pymc as pm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

import multiprocessing as mp

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

def LFI_calculator(df, cov):
    temp_LFI = np.squeeze(
        df.reshape(1, -1) @
        np.linalg.inv(cov) @
        df.reshape(-1, 1) /
        df.shape[0])
    return temp_LFI

def mu_cov_estimate(params):
    data, T = params
    dim, n_trial = data.shape
    with pm.Model() as model:
        # Define priors
        mu = pm.Normal('mu', mu=1.5, sigma=3, shape=dim)
        chol, _, _ = pm.LKJCholeskyCov("chol_cov", n=dim, eta=1, sd_dist=pm.Exponential.dist(3))
        cov = pm.Deterministic('cov', chol.dot(chol.T))
        eps = pm.MvNormal('eps', mu=np.zeros(dim), chol=chol, shape=(n_trial, dim))
        lam = pm.Deterministic('lam', np.exp(mu + eps)*T)
        Y_obs = pm.Poisson('Y_obs', mu=lam, observed=data.T*T)
        mean_field = pm.fit(n=100000, obj_optimizer=pm.adam(learning_rate=1e-2))
        trace = mean_field.sample(10000)

    mu_estimate = np.mean(trace.posterior['mu'], axis=(0,1))
    cov_estimate = np.mean(trace.posterior['cov'], axis=(0,1))
    return np.array(mu_estimate), np.array(cov_estimate)

def Decoding(X_All, Y_All):
    data = np.concatenate((X_All.T, Y_All.T), axis=0)
    label = np.concatenate((np.zeros(X_All.shape[1]), np.ones(Y_All.shape[1])), axis=0)
    clf = make_pipeline(StandardScaler(), SVC(kernel='linear', cache_size=1000))
    scores = cross_val_score(clf, data, label, scoring='accuracy', cv=10, n_jobs=-1)
    score = np.array(scores).mean()
    return score


if __name__ == '__main__':

    M1_s1 = loadmat('M1_clear_30_130.mat')['M1_stimuli1']
    M1_s2 = loadmat('M1_clear_30_130.mat')['M1_stimuli2']
    twindow = loadmat('M1_clear_30_130.mat')['twindow'].squeeze()
    M1_behavior = loadmat('M1/M1_behavior.mat')['ACC']

    contrast_levels = [10, 15, 20, 25, 27, 28, 29, 31, 32, 33, 35, 40, 50, 60]
    N_contrast = len(contrast_levels)
    days = [307, 308, 311, 313, 314,
            318, 320, 321, 329, 330,
            331, 332, 333, 334, 335, 336,
            337, 338, 339, 340, 341]
    channels = [1, 2, 3, 4, 13, 14, 18, 20, 22, 24, 33, 34, 36, 37, 38, 40, 42, 49, 50, 51, 52, 53, 54, 55, 57, 59, 60]

    pre_tp = days[:4]
    post_tp = days[-4:]

    M1_s1_pre = [[[] for _ in range(len(channels))] for _ in range(len(contrast_levels))]
    M1_s2_pre = [[[] for _ in range(len(channels))] for _ in range(len(contrast_levels))]

    M1_s1_post = [[[] for _ in range(len(channels))] for _ in range(len(contrast_levels))]
    M1_s2_post = [[[] for _ in range(len(channels))] for _ in range(len(contrast_levels))]

    ACC_pre = [[] for _ in range(len(contrast_levels))]
    ACC_post = [[] for _ in range(len(contrast_levels))]

    for i_contrast, contrast in enumerate(contrast_levels):
            for i_day, day in enumerate(days):
                if day in pre_tp:
                    if len(ACC_pre[i_contrast]) == 0:
                        ACC_pre[i_contrast] = M1_behavior[i_contrast, i_day]
                    else:
                        ACC_pre[i_contrast] = np.vstack((ACC_pre[i_contrast].reshape(-1, 1), M1_behavior[i_contrast, i_day].reshape(-1, 1)))
                elif day in post_tp:
                    if len(ACC_post[i_contrast]) == 0:
                        ACC_post[i_contrast] = M1_behavior[i_contrast, i_day]
                    else:
                        ACC_post[i_contrast] = np.vstack((ACC_post[i_contrast].reshape(-1, 1), M1_behavior[i_contrast, i_day].reshape(-1, 1)))
            ACC_pre[i_contrast] = np.mean(ACC_pre[i_contrast], axis=0)
            ACC_post[i_contrast] = np.mean(ACC_post[i_contrast], axis=0)
    ACC_pre = np.array(ACC_pre)
    ACC_post = np.array(ACC_post)

    for i_contrast, contrast in enumerate(contrast_levels):
        for i_channel, channel in enumerate(channels):
            for i_day, day in enumerate(days):
                if day in pre_tp:
                    if len(M1_s1_pre[i_contrast][i_channel]) == 0:
                        M1_s1_pre[i_contrast][i_channel] = M1_s1[i_contrast, i_channel, i_day]
                    else:
                        M1_s1_pre[i_contrast][i_channel] = np.hstack(
                            (M1_s1_pre[i_contrast][i_channel], M1_s1[i_contrast, i_channel, i_day]))
                    if len(M1_s2_pre[i_contrast][i_channel]) == 0:
                        M1_s2_pre[i_contrast][i_channel] = M1_s2[i_contrast, i_channel, i_day]
                    else:
                        M1_s2_pre[i_contrast][i_channel] = np.hstack(
                            (M1_s2_pre[i_contrast][i_channel], M1_s2[i_contrast, i_channel, i_day]))
                elif day in post_tp:
                    if len(M1_s1_post[i_contrast][i_channel]) == 0:
                        M1_s1_post[i_contrast][i_channel] = M1_s1[i_contrast, i_channel, i_day]
                    else:
                        M1_s1_post[i_contrast][i_channel] = np.hstack(
                            (M1_s1_post[i_contrast][i_channel], M1_s1[i_contrast, i_channel, i_day]))
                    if len(M1_s2_post[i_contrast][i_channel]) == 0:
                        M1_s2_post[i_contrast][i_channel] = M1_s2[i_contrast, i_channel, i_day]
                    else:
                        M1_s2_post[i_contrast][i_channel] = np.hstack(
                            (M1_s2_post[i_contrast][i_channel], M1_s2[i_contrast, i_channel, i_day]))
        M1_s1_pre[i_contrast] = np.array(M1_s1_pre[i_contrast]).squeeze()
        M1_s1_post[i_contrast] = np.array(M1_s1_post[i_contrast]).squeeze()
        M1_s2_pre[i_contrast] = np.array(M1_s2_pre[i_contrast]).squeeze()
        M1_s2_post[i_contrast] = np.array(M1_s2_post[i_contrast]).squeeze()

    FanoFactor_tp1 = [[] for _ in range(N_contrast)]
    FanoFactor_tp2 = [[] for _ in range(N_contrast)]
    NoiseCorrelation_tp1 = [[] for _ in range(N_contrast)]
    NoiseCorrelation_tp2 = [[] for _ in range(N_contrast)]
    fr_tp1 = [[] for _ in range(N_contrast)]
    fr_tp2 = [[] for _ in range(N_contrast)]

    df_abs_tp1 = np.full([N_contrast], np.nan)
    df_abs_tp2 = np.full([N_contrast], np.nan)
    variance_tp1 = [[] for _ in range(N_contrast)]
    variance_tp2 = [[] for _ in range(N_contrast)]
    LFI_tp1 = np.full([N_contrast], np.nan)
    LFI_tp2 = np.full([N_contrast], np.nan)
    rotation_tp1_tp2 = np.full([N_contrast], np.nan)
    PCA_LFI = np.full([N_contrast, 4], np.nan)
    PCA_rotation = [[] for _ in range(N_contrast)]
    PC_abs_tp1 = [[] for _ in range(N_contrast)]
    PC_abs_tp2 = [[] for _ in range(N_contrast)]
    decoding_tp1 = np.full([N_contrast], np.nan)
    decoding_tp2 = np.full([N_contrast], np.nan)

    for i_contrast, contrast in enumerate(contrast_levels):
        fr_tp1[i_contrast].append(M1_s2_pre[i_contrast].mean(axis=1))
        fr_tp2[i_contrast].append(M1_s2_post[i_contrast].mean(axis=1))
        FanoFactor_tp1[i_contrast].append((M1_s2_pre[i_contrast].var(axis=1) / M1_s2_pre[i_contrast].mean(axis=1))*(twindow[1]-twindow[0])/1000)
        FanoFactor_tp2[i_contrast].append((M1_s2_post[i_contrast].var(axis=1) / M1_s2_post[i_contrast].mean(axis=1))*(twindow[1]-twindow[0])/1000)
        nc_tp1 = np.corrcoef(M1_s2_pre[i_contrast])
        nc_tp2 = np.corrcoef(M1_s2_post[i_contrast])
        NoiseCorrelation_tp1[i_contrast].append(nc_tp1[np.triu_indices(n=nc_tp1.shape[0], k=1)])
        NoiseCorrelation_tp2[i_contrast].append(nc_tp2[np.triu_indices(n=nc_tp2.shape[0], k=1)])

    for i_contrast, contrast in enumerate(contrast_levels):

        decoding_tp1[i_contrast] = Decoding(M1_s1_pre[i_contrast], M1_s2_pre[i_contrast])
        decoding_tp2[i_contrast] = Decoding(M1_s1_post[i_contrast], M1_s2_post[i_contrast])

        params = [(M1_s2_pre[i_contrast], (twindow[1] - twindow[0]) / 1000),
                  (M1_s2_post[i_contrast], (twindow[1] - twindow[0]) / 1000),
                  (M1_s1_pre[i_contrast], (twindow[1] - twindow[0]) / 1000),
                  (M1_s1_post[i_contrast], (twindow[1] - twindow[0]) / 1000)]

        with mp.Pool(4) as pool:
            results = pool.map(mu_cov_estimate, params)
        s2_pre, s2_post, s1_pre, s1_post = results

        df_tp1 = (s2_pre[0] - s1_pre[0]) / (contrast / 100 - 0.3)
        df_tp2 = (s2_post[0] - s1_post[0]) / (contrast / 100 - 0.3)
        df_abs_tp1[i_contrast] = np.linalg.norm(df_tp1, axis=0)
        df_abs_tp2[i_contrast] = np.linalg.norm(df_tp2, axis=0)
        cov_tp1 = (s2_pre[1] + s1_pre[1]) / 2
        cov_tp2 = (s2_post[1] + s1_post[1]) / 2

        variance_tp1[i_contrast] = np.diag(cov_tp1)
        variance_tp2[i_contrast] = np.diag(cov_tp2)
        LFI_tp1[i_contrast] = LFI_calculator(df_tp1, cov_tp1)
        LFI_tp2[i_contrast] = LFI_calculator(df_tp2, cov_tp2)
        rotation_tp1_tp2[i_contrast] = wrap(angle_calculator(df_tp1, df_tp2))
        PCA_LFI[i_contrast], eig_pairs_pre, eig_pairs_post = PCA_LFI_calculator(
            df_tp1, df_tp2, cov_tp1, cov_tp2)
        assert len(eig_pairs_pre) == len(eig_pairs_post)
        for i in range(len(eig_pairs_pre)):
            PCA_rotation[i_contrast].append(wrap(angle_calculator(eig_pairs_pre[i][1], eig_pairs_post[i][1])))
            PC_abs_tp1[i_contrast].append(eig_pairs_pre[i][0])
            PC_abs_tp2[i_contrast].append(eig_pairs_post[i][0])

    Metrics = {'correlation_pre': NoiseCorrelation_tp1, 'correlation_post': NoiseCorrelation_tp2,
               'FanoFactor_pre': FanoFactor_tp1, 'FanoFactor_post': FanoFactor_tp2,
                  'fr_pre': fr_tp1, 'fr_post': fr_tp2,
               'decoding_pre': decoding_tp1, 'decoding_post': decoding_tp2,
               "ACC_pre": ACC_pre, "ACC_post": ACC_post,
               'LFI_pre': LFI_tp1, 'LFI_post': LFI_tp2, 'df_abs_pre': df_abs_tp1, 'df_abs_post': df_abs_tp2,
               'variance_pre': variance_tp1, 'variance_post': variance_tp2, 'rotation': rotation_tp1_tp2,
               'PCA_LFI': PCA_LFI,
               "PCA_rotation": PCA_rotation, "PC_abs_pre": PC_abs_tp1, "PC_abs_post": PC_abs_tp2}

    with open('Fig8_M1_Metrics.pkl', 'wb') as f:
        pickle.dump(Metrics, f)