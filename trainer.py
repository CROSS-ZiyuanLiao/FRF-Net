import torch
import torch.nn as nn
import torch.nn.functional as F


class Trainer(nn.Module):
    # TODO: write DocStrings
    def __init__(self, configs, model):
        super(Trainer, self).__init__()
        self._configs = configs
        self._model = model
        self._optimizer = None

    def train_step(self, target, data):
        self._optimizer.zero_grad()
        output = self._model(data)
        if self._configs.multi_label > 1:
            # todo: use other loss function?
            loss_func = nn.BCELoss()
            loss = loss_func(output, target)
        else:
            loss = F.nll_loss(output, target)
        loss.backward()
        self._optimizer.step()
        return output, loss

    def adjust_lr(self):
        configs = self._configs
        old_lr = configs.lr
        configs.lr *= configs.lr_decay_rate
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = configs.lr
        print('========== *** lr {0} change to lr {1} *** ==========\n'
              .format(old_lr, configs.lr))

    def get_optimizer(self):
        configs = self._configs
        model = self._model

        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=configs.lr,
                                    momentum=0.9,
                                    weight_decay=configs.weight_decay)

        if configs.optimizer_type == 'SGD':
            pass
        if configs.optimizer_type == 'Adam':
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
