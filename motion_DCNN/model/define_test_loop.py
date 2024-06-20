import torch
import numpy as np
from MotionNN.model.RDM_generator import Sample_generator
from MotionNN.model.FeatureExtractor import FeatureExtractor
import cv2 as cv

def test_loop(model, loss_fn, parameters, runs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Number of layers
    n_layers = 6

    with torch.no_grad():
        feature_extractor = FeatureExtractor(model).to(device)

        FiringRate_X_temp, FiringRate_X = [list() for _ in range(n_layers)], [None for _ in range(n_layers)]
        FiringRate_Y_temp, FiringRate_Y = [list() for _ in range(n_layers)], [None for _ in range(n_layers)]
        correct_temp, loss_temp = [], []
        for run in range(0, runs):
            sample = Sample_generator(*parameters)
            X = sample["target"].to(device)
            Y = sample["reference"].to(device)
            answer = sample["label"].to(device)

            X_extractor, Y_extractor = feature_extractor(X), feature_extractor(Y)
            for i_layer in range(n_layers):
                FiringRate_X_temp[i_layer].append(X_extractor[0][i_layer])
                FiringRate_Y_temp[i_layer].append(Y_extractor[0][i_layer])

            H_X, H_Y = X_extractor[1], Y_extractor[1]
            H_2 = torch.cat((H_X, H_Y), dim=1)
            p = torch.softmax(H_2, dim=1)

            loss_temp.append(loss_fn(H_2, answer).item())
            correct_temp.append(p[torch.arange(p.shape[0]), answer].mean().item())

        for i_layer in range(n_layers):
            FiringRate_X[i_layer] = np.stack(FiringRate_X_temp[i_layer], axis=3)
            FiringRate_Y[i_layer] = np.stack(FiringRate_Y_temp[i_layer], axis=3)

        test_loss = np.array(loss_temp).flatten()
        correct = np.array(correct_temp).flatten()

    print(f"Test Error: \n Accuracy: {(100 * correct.mean()):>0.1f}%, Avg loss: {test_loss.mean():>8f} \n")
    return {'FiringRate_X': FiringRate_X, 'FiringRate_Y': FiringRate_Y,
            'Correct': correct, 'Loss': test_loss}
