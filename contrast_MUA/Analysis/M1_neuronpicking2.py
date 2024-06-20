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


M1_neuronpicking = np.load("Neuronpicking.pkl", allow_pickle=True)["M1_neuronpicking"]
M1_s1 = loadmat('M1_clear_30_130.mat')['M1_stimuli1']
M1_s2 = loadmat('M1_clear_30_130.mat')['M1_stimuli2']
twindow = loadmat('M1_clear_30_130.mat')['twindow'].squeeze()

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

with open("Fig7_1.pkl", "wb") as f:
    pickle.dump({"M1_s1_pre": M1_s1_pre, "M1_s2_pre": M1_s2_pre,
                  "M1_s1_post": M1_s1_post, "M1_s2_post": M1_s2_post}, f)

