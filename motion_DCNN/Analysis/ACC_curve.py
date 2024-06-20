import numpy as np
import pickle

ref_list = [45, 135, -45, -135]
coherence_list = [0.0884, 0.1250, 0.1768, 0.2500, 0.3536, 0.5000, 0.7071, 1]
ACC_pre = [[] for _ in range(len(ref_list))]
ACC_post = [[] for _ in range(len(ref_list))]

for i_ref, ref_direction in enumerate(ref_list):
    for repetition in range(10):
        folderspath = "../Results/ref_direction_" + str(ref_direction) + \
              "_target_sep_4_results_" + str(repetition)
        ACC_temp = np.load(folderspath+"/ACC_pre-test.npy")
        ACC_pre[i_ref].append(ACC_temp)
        ACC_temp = np.load(folderspath+"/ACC_post-test.npy")
        ACC_post[i_ref].append(ACC_temp)
ACC_pre = np.array(ACC_pre)
ACC_post = np.array(ACC_post)

with open("../Results/ACC_curve.pkl", 'wb') as g:
    pickle.dump({"ACC_pre":ACC_pre, "ACC_post":ACC_post}, g)