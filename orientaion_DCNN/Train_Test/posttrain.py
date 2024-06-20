import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from AlexNet import AlexNet
from GaborCreater import training_sample_generator

def posttrain(ref_angle, sf, target_sep, learning_rate, epochs, repetition):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weightspath = "Results/" + "ref_" + str(ref_angle) + "_sf_" + \
                  str(sf) + "_dtheta_" + str(target_sep) + "_results_" + \
                  str(repetition)
    model = AlexNet().to(device)
    model.load_state_dict(torch.load(weightspath + "/pre_model_weights.pth", map_location=device))
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
        if isinstance(m, nn.Linear):
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for i_epoch in tqdm(range(epochs)):

        for noise_level in np.random.permutation(np.array([0.005, 1, 5, 10, 15, 30, 50, 75])):

            for contrast in np.random.permutation(np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])):

                input1, input2, label = training_sample_generator(p=noise_level, theta=ref_angle, contrast=contrast)
                input1, input2, label = input1.to(device), input2.to(device), label.to(device)
                output = model(input1, input2).squeeze(dim=1)
                loss = loss_fn(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # if i_epoch in np.array([1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])-1:
        if i_epoch in np.array([1, 10, 20, 50, 100, 200, 300, 400, 500]) - 1:
            torch.save(model.state_dict(), weightspath + "/" + "VPL_"+str(i_epoch)+"_model_weights.pth")

    torch.save(model.state_dict(), weightspath + "/post_model_weights.pth")