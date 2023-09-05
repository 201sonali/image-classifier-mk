import torch
import torch.nn.functional as F
import torch.nn as nn


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        # mean(-log(softmax(input)_label))
        # @input:  torch.Tensor((B,C))
        # @target: torch.Tensor((B,), dtype=torch.int64)
        # @return:  torch.Tensor((,))
        return F.cross_entropy(input, target)

        # raise NotImplementedError('ClassificationLoss.forward')


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        #  define the linear model and all layers
        self.linear = nn.Linear(3*64*64, 6)

        # raise NotImplementedError('LinearClassifier.__init__')

    def forward(self, x):
        # @x: torch.Tensor((B,3,64,64))
        # @return: torch.Tensor((B,6))

        # flatten x
        x = x.view(x.size(0), -1)

        # pass through linear model
        x = self.linear(x)
        return x

        # raise NotImplementedError('LinearClassifier.forward')


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        input_size = 3 * 64 * 64  # dimensions of data
        hidden_layer_size = 250  # tips for choosing this?
        output_layer_size = 6  # number of labels

        # layers of MLP
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_layer_size)
        # raise NotImplementedError('MLPClassifier.__init__')

    def forward(self, x):
        # @x: torch.Tensor((B,3,64,64))
        # @return: torch.Tensor((B,6))

        # flatten x
        x = x.view(x.size(0), -1)

        # pass through MLP
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

        # raise NotImplementedError('MLPClassifier.forward')


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
