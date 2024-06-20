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
    decoding_tp1 = [[[None for _ in range(14)] for _ in range(14)]  for _ in range(4)]  # 4 days, 14 twindows, 14 contrast
    decoding_tp2 = [[[None for _ in range(14)] for _ in range(14)]  for _ in range(4)]

    days = [24, 25, 27, 28, 29,
            30, 31, 33, 34, 35,
            36, 37, 38, 39, 40,
            41, 42, 43, 44, 45,
            46, 48, 49]
    contrast_levels = [10, 15, 20, 25, 27, 28, 29, 31, 32, 33, 35, 40, 50, 60]
    N_contrast = len(contrast_levels)
    channels = [1, 2, 3, 4, 5, 6, 8, 10, 24, 35, 37, 39, 40, 41, 49, 50, 52, 53, 54, 56]
    for i, n_day in enumerate([4, 6, 8, 10]):
        pre_tp = days[:n_day]
        post_tp = days[-n_day:]
        for i_window, twindow in enumerate(
                [[30, 130], [130, 230], [230, 330], [330, 430], [430, 530], [30, 230], [130, 330], [230, 430],
                 [330, 530], [30, 330], [130, 430], [230, 530], [30, 430], [130, 530]]):
            M2_s1 = loadmat("M2_clear_" + str(twindow[0]) + "_" + str(twindow[1]) + ".mat")['M2_stimuli1']
            M2_s2 = loadmat("M2_clear_" + str(twindow[0]) + "_" + str(twindow[1]) + ".mat")['M2_stimuli2']
            M2_behavior = loadmat('M2/M2_behavior.mat')['ACC']

            M2_s1_pre = [[[] for _ in range(len(channels))] for _ in range(len(contrast_levels))]
            M2_s2_pre = [[[] for _ in range(len(channels))] for _ in range(len(contrast_levels))]
            ACC_pre = [[] for _ in range(len(contrast_levels))]

            M2_s1_post = [[[] for _ in range(len(channels))] for _ in range(len(contrast_levels))]
            M2_s2_post = [[[] for _ in range(len(channels))] for _ in range(len(contrast_levels))]
            ACC_post = [[] for _ in range(len(contrast_levels))]

            for i_contrast, contrast in enumerate(contrast_levels):
                    for i_day, day in enumerate(days):
                        if day in pre_tp:
                            if len(ACC_pre[i_contrast]) == 0:
                                ACC_pre[i_contrast] = M2_behavior[i_contrast, i_day]
                            else:
                                ACC_pre[i_contrast] = np.vstack((ACC_pre[i_contrast], M2_behavior[i_contrast, i_day]))
                        elif day in post_tp:
                            if len(ACC_post[i_contrast]) == 0:
                                ACC_post[i_contrast] = M2_behavior[i_contrast, i_day]
                            else:
                                ACC_post[i_contrast] = np.vstack((ACC_post[i_contrast], M2_behavior[i_contrast, i_day]))
                    ACC_pre[i_contrast] = np.mean(ACC_pre[i_contrast], axis=0)
                    ACC_post[i_contrast] = np.mean(ACC_post[i_contrast], axis=0)
            ACC_pre = np.array(ACC_pre)
            ACC_post = np.array(ACC_post)

            for i_contrast, contrast in enumerate(contrast_levels):
                for i_channel, channel in enumerate(channels):
                    for i_day, day in enumerate(days):
                        if day in pre_tp:
                            if len(M2_s1_pre[i_contrast][i_channel]) == 0:
                                M2_s1_pre[i_contrast][i_channel] = M2_s1[i_contrast, i_channel, i_day]
                            else:
                                M2_s1_pre[i_contrast][i_channel] = np.hstack(
                                    (M2_s1_pre[i_contrast][i_channel], M2_s1[i_contrast, i_channel, i_day]))
                            if len(M2_s2_pre[i_contrast][i_channel]) == 0:
                                M2_s2_pre[i_contrast][i_channel] = M2_s2[i_contrast, i_channel, i_day]
                            else:
                                M2_s2_pre[i_contrast][i_channel] = np.hstack(
                                    (M2_s2_pre[i_contrast][i_channel], M2_s2[i_contrast, i_channel, i_day]))

                        elif day in post_tp:
                            if len(M2_s1_post[i_contrast][i_channel]) == 0:
                                M2_s1_post[i_contrast][i_channel] = M2_s1[i_contrast, i_channel, i_day]
                            else:
                                M2_s1_post[i_contrast][i_channel] = np.hstack(
                                    (M2_s1_post[i_contrast][i_channel], M2_s1[i_contrast, i_channel, i_day]))
                            if len(M2_s2_post[i_contrast][i_channel]) == 0:
                                M2_s2_post[i_contrast][i_channel] = M2_s2[i_contrast, i_channel, i_day]
                            else:
                                M2_s2_post[i_contrast][i_channel] = np.hstack(
                                    (M2_s2_post[i_contrast][i_channel], M2_s2[i_contrast, i_channel, i_day]))
                M2_s1_pre[i_contrast] = np.array(M2_s1_pre[i_contrast]).squeeze()
                M2_s1_post[i_contrast] = np.array(M2_s1_post[i_contrast]).squeeze()
                M2_s2_pre[i_contrast] = np.array(M2_s2_pre[i_contrast]).squeeze()
                M2_s2_post[i_contrast] = np.array(M2_s2_post[i_contrast]).squeeze()

            for i_contrast, contrast in enumerate(contrast_levels):

                decoding_tp1[i][i_window][i_contrast] = Decoding(M2_s1_pre[i_contrast], M2_s2_pre[i_contrast])
                decoding_tp2[i][i_window][i_contrast] = Decoding(M2_s1_post[i_contrast], M2_s2_post[i_contrast])

    decoding_tp1 = np.array(decoding_tp1)
    decoding_tp2 = np.array(decoding_tp2)
    np.savez("M2_decoding_twindow", decoding_tp1=decoding_tp1, decoding_tp2=decoding_tp2)