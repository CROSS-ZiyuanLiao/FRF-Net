import torch
import torch.nn as nn
import torch.nn.functional as F


class Trainer(nn.Module):
    """
    The trainer of the frf-net.

    Parameters
    ----------
    configs : Configs (see configs.py)
    model : FullyLinearDenseNet (see fl_densenet.py)

    """

    def __init__(self, configs, model):
        super(Trainer, self).__init__()
        self._configs = configs
        self._model = model
        self._optimizer = None

    def train_step(self, target, data):
        """Take a train step."""
        self._optimizer.zero_grad()
        output = self._model(data)
        if self._configs.multi_label > 1:
            loss = F.binary_cross_entropy_with_logits(output, target)
            with torch.no_grad():
                output = F.sigmoid(output)
        else:
            loss = F.cross_entropy(output, target)
            with torch.no_grad():
                output = F.log_softmax(output, dim=1)
        loss.backward()
        self._optimizer.step()
        return output, loss

    def adjust_lr(self):
        """Adjust learning rate by multiplying lr_decay_rate in configs."""
        configs = self._configs
        old_lr = configs.lr
        configs.lr *= configs.lr_decay_rate
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = configs.lr
        print('========== *** lr {0} change to lr {1} *** ==========\n'
              .format(old_lr, configs.lr))

    def get_optimizer(self):
        """Get optimizer for the trainer."""
        configs = self._configs
        model = self._model

        if configs.optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=configs.lr,
                                        momentum=0.9,
                                        weight_decay=configs.weight_decay)
        elif configs.optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=configs.lr,
                                         weight_decay=configs.weight_decay)

        self._optimizer = optimizer

    def save(self):
        # TODO: save model
        pass

    def load(self):
        # TODO: load model
        pass
