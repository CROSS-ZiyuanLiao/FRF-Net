import os

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

    _PARAM_FOLDER = 'param'
    _SUFFIX = '.pt'
    _LOAD_KEYWORDS = ('n_test_samples',
                      'test_batch_size',
                      'n_test_workers',
                      'test_pin_memory',
                      'top_k',
                      'plot_every')

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
                output = torch.sigmoid(output)
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
        """Save model."""
        save_dict = {
            'model': self._model.state_dict(),
            'configs': self._configs.state_dict,
            'optimizer': self._optimizer.state_dict()
        }

        save_name = self._configs.kind + self._SUFFIX
        save_path = os.path.join(self._PARAM_FOLDER, save_name)

        if not os.path.exists(self._PARAM_FOLDER):
            os.mkdir(self._PARAM_FOLDER)

        torch.save(save_dict, save_path)

    def load(self, param_path, test_data_dir, kwargs):
        """
        Load model.

        Parameters
        ----------
        param_path : str
            Path of the file that contains model parameters,
            configs and optimizer
        test_data_dir : str
            Path of test data
        kwargs : dict
            Keyword-argument pairs,
            available keyword are listed in self._LOAD_KEYWORDS,
            for more details, see Configs in config.py

        """
        load_dict = torch.load(param_path)

        configs_dict = load_dict['configs']
        configs_dict['test_data_dir'] = test_data_dir
        configs_dict['test_data_name'] = None

        for key in self._LOAD_KEYWORDS:
            if key in kwargs:
                configs_dict[key] = kwargs[key]

        self._model.load_state_dict(load_dict['model'])
        self._configs.parse_params(configs_dict)
        self._optimizer.load_state_dict(load_dict['optimizer'])
