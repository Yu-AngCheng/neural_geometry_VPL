import numpy as np
from numpy.testing import assert_allclose


subjects = 22
regions = 7
trials = 120
smalltr = 30
tps = 2

def LFI_calculator(df, cov):
    temp_LFI = np.squeeze(
        df.reshape(1, -1) @
        np.linalg.inv(cov) @
        df.reshape(-1, 1) /
        df.shape[0])
    return temp_LFI

def PCA_LFI_calculator(df_pre, df_post, cov_pre, cov_post):

    eig_val_pre, eig_vec_pre = np.linalg.eigh(cov_pre)
    eig_pairs_pre = [(np.abs(eig_val_pre[i]), eig_vec_pre[:, i]) for i in range(len(eig_val_pre))]
    eig_pairs_pre.sort(reverse=True)

    eig_val_post, eig_vec_post = np.linalg.eigh(cov_post)
    eig_pairs_post = [(np.abs(eig_val_post[i]), eig_vec_post[:, i]) for i in range(len(eig_val_post))]
    eig_pairs_post.sort(reverse=True)

    assert df_pre.size == df_post.size
    N = df_pre.size

    df_abs_pre = np.linalg.norm(df_pre)
    unit_df_pre = df_pre / np.linalg.norm(df_pre)
    df_abs_post = np.linalg.norm(df_post)
    unit_df_post = df_post / np.linalg.norm(df_post)
    lambda_sum_pre = eig_val_pre.sum()
    lambda_sum_post = eig_val_post.sum()
    unit_eig_val_pre = eig_val_pre / eig_val_pre.sum()
    unit_eig_val_post = eig_val_post / eig_val_post.sum()

    temp0 = df_abs_pre ** 2 / (N * lambda_sum_pre) * (
            unit_df_pre.reshape(1, -1) @
            eig_vec_pre @ np.diag(1 / unit_eig_val_pre) @ eig_vec_pre.T
            @ unit_df_pre.reshape(-1, 1))
    temp1 = df_abs_post ** 2 / (N * lambda_sum_pre) * (
            unit_df_pre.reshape(1, -1) @
            eig_vec_pre @ np.diag(1 / unit_eig_val_pre) @ eig_vec_pre.T
            @ unit_df_pre.reshape(-1, 1))
    temp2 = df_abs_post ** 2 / (N * lambda_sum_post) * (
            unit_df_pre.reshape(1, -1) @
            eig_vec_pre @ np.diag(1 / unit_eig_val_pre) @ eig_vec_pre.T
            @ unit_df_pre.reshape(-1, 1))
    temp3 = df_abs_post ** 2 / (N * lambda_sum_post) * (
            unit_df_post.reshape(1, -1) @
            eig_vec_pre @ np.diag(1 / unit_eig_val_pre) @ eig_vec_pre.T
            @ unit_df_post.reshape(-1, 1))
    temp4 = df_abs_post ** 2 / (N * lambda_sum_post) * (
            unit_df_post.reshape(1, -1) @
            eig_vec_post @ np.diag(1 / unit_eig_val_post) @ eig_vec_post.T
            @ unit_df_post.reshape(-1, 1))

    temp0 = temp0.flatten()
    temp1 = temp1.flatten()
    temp2 = temp2.flatten()
    temp3 = temp3.flatten()
    temp4 = temp4.flatten()

    PCA_LFI = np.hstack(
        (np.log10(temp1) - np.log10(temp0),
         np.log10(temp2) - np.log10(temp0),
         np.log10(temp3) - np.log10(temp0),
         np.log10(temp4) - np.log10(temp0)))

    return PCA_LFI, eig_pairs_pre, eig_pairs_post
