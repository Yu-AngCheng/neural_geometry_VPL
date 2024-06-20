import torch
import torch.nn as nn
from tqdm import tqdm
from AlexNet import AlexNet
from GaborCreater import training_sample_generator
import os

def pretrain(ref_angle, sf, target_sep, learning_rate, epochs, repetition):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AlexNet().to(device)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
        if isinstance(m, nn.Linear):
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    weightspath = "Results/" + "ref_" + str(ref_angle) + "_sf_" + \
                  str(sf) + "_dtheta_" + str(target_sep) + "_results_" +\
                  str(repetition)
    if os.path.exists(weightspath) is not True:
        os.makedirs(weightspath)

    for i_epoch in tqdm(range(epochs)):

        for noise_level in [None]:

            for contrast in [1]:

                input1, input2, label = training_sample_generator(p=noise_level, theta=ref_angle, contrast=contrast)
                input1, input2, label = input1.to(device), input2.to(device), label.to(device)
                output = model(input1, input2).squeeze(dim=1)
                loss = loss_fn(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    torch.save(model.state_dict(), weightspath + "/pre_model_weights.pth")
