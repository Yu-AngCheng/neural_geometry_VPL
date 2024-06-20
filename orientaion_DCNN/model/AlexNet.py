import torch
import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self) -> None:
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, groups=2)
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, groups=2)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, groups=2)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=1)
        self.sigmoid = nn.Sigmoid()

        self.__init_weight()
        self.__load_pretrained_weights()

        self.n_layers = 5
        self.readout = False

    def forward_once(self, x: torch.Tensor):

        l1 = self.lrn1(self.relu(self.conv1(x)))
        l2 = self.lrn2(self.relu(self.conv2(self.pool1(l1))))
        l3 = self.relu(self.conv3(self.pool2(l2)))
        l4 = self.relu(self.conv4(l3))
        l5 = self.relu(self.conv5(l4))
        l5_flat = torch.flatten(self.pool3(l5), 1)
        output = self.fc1(l5_flat)

        if self.readout:
            return [l1.mean(axis=(2, 3)).cpu().numpy(), l2.mean(axis=(2, 3)).cpu().numpy(),
                    l3.mean(axis=(2, 3)).cpu().numpy(), l4.mean(axis=(2, 3)).cpu().numpy(),
                    l5.mean(axis=(2, 3)).cpu().numpy()], output
        else:
            return output

    def forward(self, input1, input2):
        if self.readout:

            input1_fr, output1 = self.forward_once(input1)
            input2_fr, output2 = self.forward_once(input2)
            output = output1 - output2
            return {"Target": input1_fr, "Reference": input2_fr}, output

        else:

            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            output = output1 - output2

            return output

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight.data, 0)
                nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight.data, 0)
                nn.init.constant_(m.bias.data, 0)

    def __load_pretrained_weights(self):
        pretrained_dict = torch.load('bvlc_alexnet.caffemodel.pt')
        model_dict = self.state_dict()
        model_dict["conv1.weight"] = torch.squeeze(pretrained_dict["conv1.weight"])
        model_dict["conv1.bias"] = torch.squeeze(pretrained_dict["conv1.bias"])
        model_dict["conv2.weight"] = torch.squeeze(pretrained_dict["conv2.weight"])
        model_dict["conv2.bias"] = torch.squeeze(pretrained_dict["conv2.bias"])
        model_dict["conv3.weight"] = torch.squeeze(pretrained_dict["conv3.weight"])
        model_dict["conv3.bias"] = torch.squeeze(pretrained_dict["conv3.bias"])
        model_dict["conv4.weight"] = torch.squeeze(pretrained_dict["conv4.weight"])
        model_dict["conv4.bias"] = torch.squeeze(pretrained_dict["conv4.bias"])
        model_dict["conv5.weight"] = torch.squeeze(pretrained_dict["conv5.weight"])
        model_dict["conv5.bias"] = torch.squeeze(pretrained_dict["conv5.bias"])
        self.load_state_dict(model_dict)