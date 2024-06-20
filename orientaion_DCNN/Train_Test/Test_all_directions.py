import torch
import numpy as np
import pickle
from AlexNet import AlexNet
from tqdm import tqdm
from copy import deepcopy
from GaborCreater import test_sample_generator2
import argparse
import os


def Test_all_directions(ref_angle, sf, target_sep, runs, repetition, prefix, contrast_levels, noise_levels, orientation_levels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    weightspath = "Results/" + "ref_" + str(ref_angle) + "_sf_" + \
                  str(sf) + "_dtheta_" + str(target_sep) + "_results_" + \
                  str(repetition)
    model = AlexNet().to(device)
    model.load_state_dict(torch.load(weightspath + "/" + prefix + "_model_weights.pth", map_location=device))
    model.eval()
    model.readout = True

    FiringRate_Y = [[[[None for _ in range(len(orientation_levels))] for _ in range(len(noise_levels))] for _ in range(len(contrast_levels))] for _ in range(model.n_layers)]

    with tqdm(total=14400) as pbar:
        for i_contrast, contrast in enumerate(contrast_levels):
            for i_noise, noise in enumerate(noise_levels):
                    for i_ori, orientation in enumerate(orientation_levels):
                        with torch.no_grad():
                            input = test_sample_generator2(batchsize=runs, p=noise,
                                                           theta=orientation, contrast=contrast).to(device)
                            FR_temp, _ = model.forward_once(input)
                            for i_layer in range(model.n_layers):
                                FiringRate_Y[i_layer][i_contrast][i_noise][i_ori] = deepcopy(FR_temp[i_layer])
                        pbar.update(1)
    for i_layer in range(model.n_layers):
        FiringRate_Y[i_layer] = np.array(FiringRate_Y[i_layer])
    dataspath = "Results_all_directions/" + "ref_" + str(ref_angle) + "_sf_" + \
                str(sf) + "_dtheta_" + str(target_sep) + "_results_" + \
                str(repetition)
    if os.path.exists(dataspath) is not True:
        os.makedirs(dataspath)

    if prefix == "pre":
        with open(dataspath + "/FiringRate_Pre_all_directions.pkl", 'wb') as f:
            pickle.dump({'FiringRate_Pre_all_directions': FiringRate_Y}, f)
    elif prefix == "post":
        with open(dataspath + "/FiringRate_Post_all_directions.pkl", 'wb') as f:
            pickle.dump({'FiringRate_Post_all_directions': FiringRate_Y}, f)
    else:
        ValueError("Wrong nepoch value!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Denoise simplified')
    parser.add_argument('--ref_angle', type=int)
    parser.add_argument('--prefix', type=str)
    args = parser.parse_args()
    for sf in [40]:
        for target_sep in [1]:
            for repetition in range(1):
                contrast_levels = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
                noise_levels = [0.005, 1, 5, 10, 15, 30, 50, 75]
                orientation_levels = np.arange(0, 180, 1)
                Test_all_directions(ref_angle=args.ref_angle, sf=sf, target_sep=target_sep,
                                    runs=100, repetition=repetition, prefix=args.prefix,
                                    contrast_levels=contrast_levels, noise_levels=noise_levels,
                                    orientation_levels=orientation_levels)