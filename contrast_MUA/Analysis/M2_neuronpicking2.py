import numpy as np
from scipy.io import loadmat
from PCA_LFI_calculator import PCA_LFI_calculator
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pymc as pm
import multiprocessing as mp

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
        trace = mean_field.sample(100000)

    mu_estimate = np.mean(trace.posterior['mu'], axis=(0,1))
    cov_estimate = np.mean(trace.posterior['cov'], axis=(0,1))
    return np.array(mu_estimate), np.array(cov_estimate)


M2_neuronpicking = np.load("Neuronpicking.pkl", allow_pickle=True)["M2_neuronpicking"]
M2_s1 = loadmat('M2_clear_130_230.mat')['M2_stimuli1']
M2_s2 = loadmat('M2_clear_130_230.mat')['M2_stimuli2']
twindow = loadmat('M2_clear_130_230.mat')['twindow'].squeeze()

contrast_levels = [10, 15, 20, 25, 27, 28, 29, 31, 32, 33, 35, 40, 50, 60]
N_contrast = len(contrast_levels)
days = [24, 25, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49]
channels = [1, 2, 3, 4, 5, 6, 8, 10, 24, 35, 37, 39, 40, 41, 49, 50, 52, 53, 54, 56]

pre_tp = days[:4]
post_tp = days[-4:]

M2_s1_pre = [[[] for _ in range(len(channels))] for _ in range(len(contrast_levels))]
M2_s2_pre = [[[] for _ in range(len(channels))] for _ in range(len(contrast_levels))]

M2_s1_post = [[[] for _ in range(len(channels))] for _ in range(len(contrast_levels))]
M2_s2_post = [[[] for _ in range(len(channels))] for _ in range(len(contrast_levels))]

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

with open("Fig7_2.pkl", "wb") as f:
    pickle.dump({"M2_s1_pre": M2_s1_pre, "M2_s2_pre": M2_s2_pre,
                 "M2_s1_post": M2_s1_post, "M2_s2_post": M2_s2_post}, f)