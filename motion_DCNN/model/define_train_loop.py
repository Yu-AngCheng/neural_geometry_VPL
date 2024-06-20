import torch
from model.RDM_generator import Sample_generator


def train_loop(model, loss_fn, parameters, optimizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    sample = Sample_generator(*parameters)
    X = sample["target"].to(device)
    Y = sample["reference"].to(device)
    answer = sample["label"].to(device)

    H_X = model(X)
    H_Y = model(Y)
    H_2 = torch.cat((H_X, H_Y), dim=1)
    loss = loss_fn(H_2, answer)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
