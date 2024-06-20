import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

from MotionNN.model.C3D import C3D
from MotionNN.model.define_train_loop import train_loop

from tqdm import tqdm


def posttrain(ref_direction, target_sep, lr, epochs, repetition, recorded_epoch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weightspath = "../Results/" + "ref_direction_" + str(ref_direction) + \
                  "_target_sep_" + str(target_sep) + "_results_" + \
                  str(repetition)
    model = C3D(pretrained=False).to(device)
    model.load_state_dict(torch.load(weightspath + "/Pre_model_weights.pt", map_location=device)['model_state_dict'])

    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            m.weight.requires_grad = True
            m.bias.requires_grad = True
            m.weight.grad = None
            m.bias.grad = None
        elif isinstance(m, nn.Linear):
            m.weight.requires_grad = True
            m.bias.requires_grad = True
            m.weight.grad = None
            m.bias.grad = None

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    coherence_levels = np.array([0.0884, 0.1250, 0.1768, 0.2500, 0.3536, 0.5000, 0.7071, 1])

    for t in tqdm(range(epochs)):
        coherence_seq = np.random.permutation(coherence_levels)
        for coherence in coherence_seq:
            parameters = (ref_direction, target_sep, coherence)
            train_loop(model, loss_fn, parameters, optimizer)

        if t in recorded_epoch:
            state = {'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()}
            torch.save(state, weightspath + "/Post-train_"+str(t + 1)+"_model_weights.pt")

    state = {'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict()}
    torch.save(state, weightspath + "/Post_model_weights.pt")
    print("Done!")
