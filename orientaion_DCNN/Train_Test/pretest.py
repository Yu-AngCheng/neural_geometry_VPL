import torch
import torch.nn as nn
from tqdm import tqdm
from AlexNet import AlexNet
from copy import deepcopy
from GaborCreater import test_sample_generator
import numpy as np
import pickle

def pretest(ref_angle, sf, target_sep, runs, repetition):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weightspath = "Results/" + "ref_" + str(ref_angle) + "_sf_" + \
                  str(sf) + "_dtheta_" + str(target_sep) + "_results_" + \
                  str(repetition)
    model = AlexNet().to(device)
    model.load_state_dict(torch.load(weightspath + "/pre_model_weights.pth", map_location=device))
    model.eval()
    model.readout=True

    ACC_pre = np.full([11, 8, 2, runs], np.NaN)
    FR_target = [[[[None for _ in range(2)] for _ in range(8)] for _ in range(11)] for _ in range(model.n_layers)]
    FR_ref = [[[[None for _ in range(2)] for _ in range(8)] for _ in range(11)] for _ in range(model.n_layers)]

    with tqdm(total=176) as pbar:
        for contrast in np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]):
            for noise_level in np.array([0.005, 1, 5, 10, 15, 30, 50, 75]):
                for flag in np.array([0, 1]):
                    contrast_idx = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0].index(contrast)
                    noise_idx = [0.005, 1, 5, 10, 15, 30, 50, 75].index(noise_level)
                    with torch.no_grad():
                        input1, input2, label = test_sample_generator(batchsize=runs, p=noise_level,
                                                                      theta=ref_angle, contrast=contrast, flag=flag)
                        input1, input2, label = input1.to(device), input2.to(device), label.to(device)
                        FR, output = model(input1, input2)

                    output_p = model.sigmoid(output.squeeze(dim=1))
                    output_p_corrected = output_p * label + (1 - output_p) * (1 - label)
                    ACC_pre[contrast_idx][noise_idx][flag] = deepcopy(output_p_corrected.cpu().numpy())

                    FR_target_temp = deepcopy(FR['Target'])
                    FR_ref_temp = deepcopy(FR['Reference'])
                    for i_layer in range(model.n_layers):
                        FR_target[i_layer][contrast_idx][noise_idx][flag] = deepcopy(FR_target_temp[i_layer])
                        FR_ref[i_layer][contrast_idx][noise_idx][flag] = deepcopy(FR_ref_temp[i_layer])

                    pbar.update(1)

        for i_layer in range(model.n_layers):
            FR_target[i_layer] = np.array(FR_target[i_layer])
            FR_ref[i_layer] = np.array(FR_ref[i_layer])

    np.save(weightspath + '/ACC_pre.npy', ACC_pre)
    with open(weightspath + "/FR_pre.pkl", 'wb') as f:
        pickle.dump({'FR_target': FR_target, 'FR_ref': FR_ref}, f)