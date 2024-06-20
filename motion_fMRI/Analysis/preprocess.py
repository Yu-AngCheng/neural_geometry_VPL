import scipy.io as io
import numpy as np
from copy import deepcopy


def preprocess():
    fMRI_motion = io.loadmat("motionPLbetar2.mat")['betas']
    subjects = 22
    regions = 7
    runs = 8

    voxel_pre_left_temp = [[[] for _ in range(regions)] for _ in range(subjects)]
    voxel_pre_right_temp = [[[] for _ in range(regions)] for _ in range(subjects)]
    voxel_post_left_temp = [[[] for _ in range(regions)] for _ in range(subjects)]
    voxel_post_right_temp = [[[] for _ in range(regions)] for _ in range(subjects)]
    for i_sub in range(subjects):
        for i_reg in range(regions):
            for i_run in range(runs):

                if i_run == 0 or i_run == 1 or i_run == 2 or i_run == 3:
                    voxel_pre_left_temp[i_sub][i_reg].append(deepcopy(fMRI_motion[i_sub][i_reg][i_run][:, 0:30]))
                    voxel_pre_right_temp[i_sub][i_reg].append(deepcopy(fMRI_motion[i_sub][i_reg][i_run][:, 30:60]))
                elif i_run == 4 or i_run == 5 or i_run == 6 or i_run == 7:
                    voxel_post_left_temp[i_sub][i_reg].append(deepcopy(fMRI_motion[i_sub][i_reg][i_run][:, 0:30]))
                    voxel_post_right_temp[i_sub][i_reg].append(deepcopy(fMRI_motion[i_sub][i_reg][i_run][:, 30:60]))

    voxel_pre_left = [[None for _ in range(regions)] for _ in range(subjects)]
    voxel_pre_right = [[None for _ in range(regions)] for _ in range(subjects)]
    voxel_post_left = [[None for _ in range(regions)] for _ in range(subjects)]
    voxel_post_right = [[None for _ in range(regions)] for _ in range(subjects)]
    for i_sub in range(subjects):
        for i_reg in range(regions):
            voxel_pre_left[i_sub][i_reg] = deepcopy(np.hstack(voxel_pre_left_temp[i_sub][i_reg]))
            voxel_pre_right[i_sub][i_reg] = deepcopy(np.hstack(voxel_pre_right_temp[i_sub][i_reg]))
            voxel_post_left[i_sub][i_reg] = deepcopy(np.hstack(voxel_post_left_temp[i_sub][i_reg]))
            voxel_post_right[i_sub][i_reg] = deepcopy(np.hstack(voxel_post_right_temp[i_sub][i_reg]))

    voxel_pre_left_clear = [[None for _ in range(regions)] for _ in range(subjects)]
    voxel_pre_right_clear = [[None for _ in range(regions)] for _ in range(subjects)]
    voxel_post_left_clear = [[None for _ in range(regions)] for _ in range(subjects)]
    voxel_post_right_clear = [[None for _ in range(regions)] for _ in range(subjects)]
    for i_sub in range(subjects):
        for i_reg in range(regions):

            temp = (np.mean(voxel_pre_left[i_sub][i_reg], axis=1) +
                    np.mean(voxel_pre_right[i_sub][i_reg], axis=1))

            effective = np.sum(np.logical_not(np.isnan(temp)))
            idx = (-temp).argsort()[:np.min([60, effective])]

            voxel_pre_left_clear[i_sub][i_reg] = deepcopy(voxel_pre_left[i_sub][i_reg][idx])
            voxel_pre_right_clear[i_sub][i_reg] = deepcopy(voxel_pre_right[i_sub][i_reg][idx])
            voxel_post_left_clear[i_sub][i_reg] = deepcopy(voxel_post_left[i_sub][i_reg][idx])
            voxel_post_right_clear[i_sub][i_reg] = deepcopy(voxel_post_right[i_sub][i_reg][idx])

    return [[voxel_pre_left_clear, voxel_pre_right_clear], [voxel_post_left_clear, voxel_post_right_clear]]
