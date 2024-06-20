import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):

    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model

    def forward(self, x):
        self.model.eval()
        outputs = list()

        x = self.model.relu(self.model.conv1(x))
        outputs.append(x.mean(axis=[3, 4]).cpu().numpy())
        x = self.model.pool1(x)

        x = self.model.relu(self.model.conv2(x))
        outputs.append(x.mean(axis=[3, 4]).cpu().numpy())
        x = self.model.pool2(x)

        x = self.model.relu(self.model.conv3a(x))
        outputs.append(x.mean(axis=[3, 4]).cpu().numpy())
        x = self.model.relu(self.model.conv3b(x))
        outputs.append(x.mean(axis=[3, 4]).cpu().numpy())
        x = self.model.pool3(x)

        x = self.model.relu(self.model.conv4a(x))
        outputs.append(x.mean(axis=[3, 4]).cpu().numpy())
        x = self.model.relu(self.model.conv4b(x))
        outputs.append(x.mean(axis=[3, 4]).cpu().numpy())
        x = self.model.pool4(x)

        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return outputs, x
