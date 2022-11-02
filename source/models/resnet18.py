import torch.nn
import torchvision


class ResnetRegression(torch.nn.Module):
    def __init__(self, n_fc_layers=1, n_neurons=64):
        super(ResnetRegression, self).__init__()
        self.backbone = torchvision.models.resnet.resnet18(
            weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT)

        self.backbone.fc = torch.nn.Linear(in_features=self.backbone.fc.in_features,
                                           out_features=n_neurons * (n_fc_layers - 1) + 1)
        # if n_fc_layers > 1, add extra layers
        additional_layers = []
        for i in range(n_fc_layers - 1):
            additional_layers.append(torch.nn.ReLU())
            additional_layers.append(torch.nn.Linear(in_features=n_neurons * (n_fc_layers - 1 - i) + 1,
                                             out_features=n_neurons * (n_fc_layers - 2 - i) + 1))
        additional_layers.append(torch.nn.Sigmoid())
        self.sequential = torch.nn.Sequential(*additional_layers)

    def forward(self, x):
        out = self.backbone(x)
        return self.sequential(out)
