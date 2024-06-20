import torch
import torch.nn as nn


class C3D(nn.Module):

    def __init__(self, pretrained=True) -> None:
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc = nn.Linear(50176, 1)


        self.relu = nn.ReLU()

        self.__init_weight()
        if pretrained is True:
            self.__load_pretrained_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.005)
                nn.init.constant_(m.bias, val=0)

    def __load_pretrained_weights(self):
        pretrained_dict = torch.load('../model/c3d_sports1m_pretrain_20201016-dcc47ddc.pth')
        model_dict = self.state_dict()
        model_dict["conv1.weight"] = torch.squeeze(pretrained_dict["conv1a.conv.weight"])
        model_dict["conv1.bias"] = torch.squeeze(pretrained_dict["conv1a.conv.bias"])
        model_dict["conv2.weight"] = torch.squeeze(pretrained_dict["conv2a.conv.weight"])
        model_dict["conv2.bias"] = torch.squeeze(pretrained_dict["conv2a.conv.bias"])
        model_dict["conv3a.weight"] = torch.squeeze(pretrained_dict["conv3a.conv.weight"])
        model_dict["conv3a.bias"] = torch.squeeze(pretrained_dict["conv3a.conv.bias"])
        model_dict["conv3b.weight"] = torch.squeeze(pretrained_dict["conv3b.conv.weight"])
        model_dict["conv3b.bias"] = torch.squeeze(pretrained_dict["conv3b.conv.bias"])
        model_dict["conv4a.weight"] = torch.squeeze(pretrained_dict["conv4a.conv.weight"])
        model_dict["conv4a.bias"] = torch.squeeze(pretrained_dict["conv4a.conv.bias"])
        model_dict["conv4b.weight"] = torch.squeeze(pretrained_dict["conv4b.conv.weight"])
        model_dict["conv4b.bias"] = torch.squeeze(pretrained_dict["conv4b.conv.bias"])

        self.load_state_dict(model_dict, strict=False)


if __name__ == "__main__":
    from torchvision.models.feature_extraction import get_graph_node_names
    model = C3D(pretrained = True)
    nodes, _ = get_graph_node_names(model)
    print(nodes)
