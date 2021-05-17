import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyLinearDenseNet(nn.Sequential):
    """
    An implementation of the Fully Linear DenseNet (FL-DenseNet).

    Parameters
    ----------
    in_features : int
        number of input features
    out_classes : int
        number of output classes
    growth_rate : int
        growth rate of the DenseNet
    bottleneck_multiplier : int
        size multiplier of the bottleneck layer,
        the size of the bottleneck layer is bottleneck_multiplier * out_features
    drop_rate : float
        probability of an element to be zeroed in dropout function
    block_config : array-like
        number of dense layer in each dense block

    Attributes
    ----------

    Examples
    --------

    """

    # todo: docString
    def __init__(self, in_features, out_classes,
                 growth_rate, bottleneck_multiplier,
                 drop_rate, block_config):
        super(FullyLinearDenseNet, self).__init__()
        self.in_features = in_features

        inter_features = in_features
        for i, n_layers in enumerate(block_config):
            # dense block
            block = _DenseBlock(n_layers, inter_features, growth_rate,
                                bottleneck_multiplier, drop_rate)
            self.add_module('dense_block%d' % (i + 1), block)
            inter_features += n_layers * growth_rate

            # transition layer
            if i < (len(block_config) - 1):
                compression = 3  # compression number
                trans = _Transition(inter_features,
                                    inter_features // compression)
                self.add_module('transition%d' % (i + 1), trans)
                inter_features = inter_features // compression
            else:
                trans = _Transition(inter_features, out_classes)
                self.add_module('global_transition', trans)

    def forward(self, x):
        x = x.reshape(-1, self.in_features)
        out = super(FullyLinearDenseNet, self).forward(x)
        out = F.log_softmax(out, dim=1)
        return out

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0, std=1)
            nn.init.normal_(m.bias.data, mean=0, std=1)


class _DenseLayer(nn.Sequential):
    def __init__(self, in_features, growth_rate, bottleneck_multiplier,
                 drop_rate):
        """
        Dense layer of the FL-DenseNet.

        Parameters
        ----------
        in_features : int
            number of the input features
        growth_rate : int
            growth rate of the DenseNet
        bottleneck_multiplier : int
            size multiplier of the bottleneck layer,
            the size of the bottleneck layer is
            bottleneck_multiplier * out_features
        drop_rate : float
            probability of an element to be zeroed in dropout function

        """
        super(_DenseLayer, self).__init__()
        inter_features = bottleneck_multiplier * growth_rate

        self.add_module('batch_norm1', nn.BatchNorm1d(in_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('linear1', nn.Linear(
            in_features, inter_features))

        self.add_module('batch_norm2', nn.BatchNorm1d(inter_features))
        self.add_module('relu2', nn. ReLU(inplace=True))
        self.add_module('linear2', nn.Linear(inter_features, growth_rate))

        self.drop_rate = drop_rate

    def forward(self, x):
        out = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat((x, out), dim=1)


class _DenseBlock(nn.Sequential):
    def __init__(self, n_layers, in_features, growth_rate,
                 bottleneck_multiplier, drop_rate):
        """
        Dense block of the FL-DenseNet.

        Parameters
        ----------
        n_layers : int
            number of dense layers in each dense block
        in_features : int
            number of the input features
        growth_rate : int
            growth rate of the DenseNet
        bottleneck_multiplier : int
            size multiplier of the bottleneck layer,
            the size of the bottleneck layer is
            bottleneck_multiplier * out_features
        drop_rate : float
            probability of an element to be zeroed in dropout function

        """
        super(_DenseBlock, self).__init__()
        for i in range(n_layers):
            layer = _DenseLayer(
                (in_features + i * growth_rate), growth_rate,
                bottleneck_multiplier, drop_rate)
            self.add_module('dense_layer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, in_features, out_features):
        """
        Transition layer of the FL-DenseNet.

        Parameters
        ----------
        in_features : int
            number of the input features
        out_features : int
            number of the output features

        """
        super(_Transition, self).__init__()
        self.add_module('batch_norm', nn.BatchNorm1d(in_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('linear', nn.Linear(in_features, out_features))
