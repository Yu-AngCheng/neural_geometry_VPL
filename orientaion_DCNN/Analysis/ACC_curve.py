import numpy as np
import pickle
from scipy.interpolate import interp1d
from scipy.io import savemat

ref_list = [35, 55, 125, 145]
rep_list = np.arange(0, 10, 1)
contrast_levels = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
noise_levels = [0.005, 1, 5, 10, 15, 30, 50, 75]

ACC_pre = [[] for _ in range(len(ref_list))]
ACC_post = [[] for _ in range(len(ref_list))]

for i_ref, ref_orientation in enumerate(ref_list):
    for repetition in rep_list:
        folderspath = "../Results/ref_" + str(ref_orientation) + \
                      "_sf_40_dtheta_1_results_" + str(repetition)
        ACC_temp = np.load(folderspath + "/ACC_pre.npy").mean(axis=(2, 3))
        ACC_pre[i_ref].append(ACC_temp)
        ACC_temp = np.load(folderspath + "/ACC_post.npy").mean(axis=(2, 3))
        ACC_post[i_ref].append(ACC_temp)
ACC_pre = np.array(ACC_pre)
ACC_post = np.array(ACC_post)

with open("../Results/ACC_curve.pkl", 'wb') as g:
    pickle.dump({"ACC_pre": ACC_pre, "ACC_post": ACC_post}, g)
savemat("../Results/ACC_curve.mat", {"ACC_pre": ACC_pre, "ACC_post": ACC_post})
