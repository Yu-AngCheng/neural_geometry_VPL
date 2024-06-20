import torch
import torch.nn as nn
import numpy as np
import pickle
from MotionNN.model.C3D import C3D
from MotionNN.model.define_test_loop import test_loop

from datetime import datetime


def posttest(ref_direction, target_sep, runs, repetition, nepoch, coherence_levels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weightspath = "../Results/" + "ref_direction_" + str(ref_direction) + \
                  "_target_sep_" + str(target_sep) + "_results_" + \
                  str(repetition)
    model = C3D(pretrained=False).to(device)

    if nepoch is None:
        model.load_state_dict(torch.load(weightspath + "/Post_model_weights.pt", map_location=device)['model_state_dict'])
    elif isinstance(nepoch, int):
        model.load_state_dict(torch.load(weightspath + "/Post-train_"+str(nepoch)+"_model_weights.pt", map_location=device)['model_state_dict'])
    else:
        ValueError("Wrong nepoch value!")

    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    ACC = np.full([len(coherence_levels), runs], np.NaN)
    Test_Loss = np.full([len(coherence_levels), runs], np.NaN)
    FiringRate_X = [None for _ in range(len(coherence_levels))]
    FiringRate_Y = [None for _ in range(len(coherence_levels))]

    tic = datetime.now()
    for coherence_idx in range(len(coherence_levels)):
        coherence = coherence_levels[coherence_idx]
        parameters = (ref_direction, target_sep, coherence)
        temp = test_loop(model, loss_fn, parameters, runs)
        FiringRate_X[coherence_idx] = temp['FiringRate_X']
        FiringRate_Y[coherence_idx] = temp['FiringRate_Y']
        ACC[coherence_idx, :] = temp['Correct']
        Test_Loss[coherence_idx, :] = temp['Loss']
    toc = datetime.now()
    print('Elapsed time: %f seconds' % (toc - tic).total_seconds())

    if nepoch is None:
        np.save(weightspath + '/ACC_post-test.npy', ACC)
        np.save(weightspath + '/Loss_post-test.npy', Test_Loss)
        with open(weightspath + "/FiringRate_Post.pkl", 'wb') as f:
            pickle.dump({'FiringRate_X': FiringRate_X, 'FiringRate_Y': FiringRate_Y}, f, pickle.HIGHEST_PROTOCOL)
    elif isinstance(nepoch, int):
        np.save(weightspath + "/ACC_post-train_"+str(nepoch)+".npy", ACC)
        np.save(weightspath + "/Loss_post-train_" + str(nepoch) + ".npy", Test_Loss)
        with open(weightspath + "/FiringRate_Post-train_"+str(nepoch)+".pkl", 'wb') as f:
            pickle.dump({'FiringRate_X': FiringRate_X, 'FiringRate_Y': FiringRate_Y}, f, pickle.HIGHEST_PROTOCOL)
    else:
        ValueError("Wrong nepoch value!")
    print("Done!")
