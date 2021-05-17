import logging
import time
import torch

from pprint import pformat
from torch.utils.data.dataloader import DataLoader

from configs import Configs
from fl_densenet import FullyLinearDenseNet
from frf_dataset import FRFDataset
from trainer import Trainer


class FRFNet(object):
    """Not now."""

    # todo: docStrings
    def __init__(self, **kwargs):
        print('Constructing model\n')

        self.configs = Configs(kwargs)
        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.trainer = None
        self.train_data_loader = None
        self.test_data_loader = None

        # model construction
        self._build_model()
        self._build_dataset()
        self._build_trainer()
        self._build_data_loader()

        # move the model to GPU and then build optimizer
        if torch.cuda.is_available():
            self.trainer.cuda()
        self.trainer.get_optimizer()

        print('Construction complete\n')

    def _build_model(self):
        configs = self.configs
        model = FullyLinearDenseNet(
            configs.in_features,
            configs.out_classes,
            configs.growth_rate,
            configs.bottleneck_multiplier,
            configs.drop_rate,
            configs.block_config
        )
        model.apply(model.init_weights)
        self.model = model

    def _build_dataset(self):
        self.train_dataset = FRFDataset(self.configs, 'train')
        self.test_dataset = FRFDataset(self.configs, 'test')

    def _build_trainer(self):
        self.trainer = Trainer(self.configs, self.model)

    def _build_data_loader(self):
        configs = self.configs
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=configs.train_batch_size,
            num_workers=configs.n_train_workers,
            pin_memory=configs.train_pin_memory
        )
        self.test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=configs.test_batch_size,
            num_workers=configs.n_test_workers,
            pin_memory=configs.test_pin_memory
        )

    def main_worker(self):
        configs = self.configs

        # define criterion for evaluation
        criterion = torch.nn.CrossEntropyLoss().cuda()

        # training
        for epoch in range(configs.n_epoch):
            if (epoch + 1) in configs.lr_adjust_at:
                self.trainer.adjust_lr()
            self.train(epoch)

        # validation
        self.validate(criterion)

    def train(self, epoch):
        print('epoch {}'.format(epoch + 1))
        configs = self.configs
        batch_size = configs.train_batch_size

        # initialize meters
        batch_time_meter = _AverageMeter()  # batch elapsed time
        data_time_meter = _AverageMeter()  # data loading time
        loss_meter = _AverageMeter()
        top_meter = [_AverageMeter() for _ in configs.top_k]

        # switch to train mode
        self.model.train()

        # start loss value
        old_loss = 100

        # load data and train
        start_time = time.time()
        for i, (target, data) in enumerate(self.train_data_loader):
            # load data
            target, data = target.cuda(), data.cuda()

            # measure data loading time
            data_time_meter.update(time.time() - start_time)

            # take a train step
            output, loss = self.trainer.train_step(target, data)

            # update loss
            loss_meter.update(loss.item(), batch_size)
            if loss_meter.value < old_loss:
                old_loss = loss_meter.value
                print('==== iter *{}* ====, '
                      '========== *** loss update: {} *** ==========\n'
                      .format((i + 1), old_loss))

            # evaluate accuracy
            if configs.multi_label > 1:
                # TODO add multi-label support
                pass
            else:
                acc, _, _ = self._cal_accuracy(
                    output, target, top_k=configs.top_k
                )
            for ii in range(len(configs.top_k)):
                top_meter[ii].update(acc[ii], batch_size)

            # update batch elapsed time
            batch_time_meter.update(time.time() - start_time)

            # plot
            if (i + 1) % configs.plot_every == 0:
                # write python console message
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time: {3:.3f} ({4:.3f})\t'
                      'Data: {5:.3f} ({6:.3f})\t'
                      'Loss: {7:.4f} ({8:.4f})\t'
                      .format((epoch + 1), (i + 1), len(self.train_data_loader),
                              batch_time_meter.value, batch_time_meter.average,
                              data_time_meter.value, data_time_meter.average,
                              loss_meter.value, loss_meter.average),
                      end='')
                for ii in range(len(configs.top_k)):
                    print('Acc@{0} {1:.3f} ({2:.3f})\t'
                          .format(configs.top_k[ii],
                                  top_meter[ii].value,
                                  top_meter[ii].average),
                          end='')
                print('\n')

                # write log message
                logger = logging.getLogger('log')
                log_str = [' train-----* ===Epoch: [{0}][{1}/{2}]\t'.format(
                    (epoch + 1), (i + 1), len(self.train_data_loader))]
                for ii in range(len(configs.top_k)):
                    log_str.append(
                        ' Acc@{0} {1:.3f}'.format(configs.top_k[ii],
                                                  top_meter[ii].average)
                    )
                logger.info(''.join(log_str))

            # update start time for the next load
            start_time = time.time()

    @torch.no_grad()
    def validate(self, criterion, output_flag=True):
        print('validating')
        configs = self.configs
        batch_size = configs.train_batch_size

        # initialize meters
        batch_time_meter = _AverageMeter()  # batch elapsed time
        loss_meter = _AverageMeter()
        top_meter = [_AverageMeter() for _ in configs.top_k]

        # switch to evaluation mode
        self.model.eval()

        # output file
        out_path = configs.predict_path
        with open(out_path, 'w') as out_file:

            # load data and evaluate
            start_time = time.time()
            for i, (target, data) in enumerate(self.test_data_loader):
                # load data
                target, data = target.cuda(), data.cuda()

                # validation
                output = self.model(data)
                loss = criterion(output, target)

                # update loss
                loss_meter.update(loss.item(), batch_size)

                # evaluate accuracy
                if configs.multi_label > 1:
                    # TODO add multi-label support
                    pass
                else:
                    acc, pred, out_val = self._cal_accuracy(
                        output, target, top_k=configs.top_k
                    )
                for ii in range(len(configs.top_k)):
                    top_meter[ii].update(acc[ii], batch_size)

                # write details of prediction for each sample
                if output_flag:
                    pred = pred.tolist()
                    out_val = out_val.tolist()
                    target = target.tolist()
                    for ii, item in enumerate(pred):
                        out_file.writelines(str(item))
                        out_file.writelines(',')
                        out_file.writelines(str(out_val[ii]))
                        out_file.writelines(',')
                        out_file.writelines(
                            str(configs.label_list[target[ii]]))
                        out_file.writelines('\n')

                # update batch elapsed time
                batch_time_meter.update(time.time() - start_time)

                # plot
                if (i + 1) % configs.plot_every == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time: {2:.3f} ({3:.3f})\t'
                          'Loss: {4:.4f} ({5:.4f})\t'
                          .format((i + 1), len(self.test_data_loader),
                                  batch_time_meter.value,
                                  batch_time_meter.average,
                                  loss_meter.value, loss_meter.average),
                          end='')
                    for ii in range(len(configs.top_k)):
                        print('Acc@{0} {1:.3f} ({2:.3f})\t'
                              .format(configs.top_k[ii],
                                      top_meter[ii].value,
                                      top_meter[ii].average),
                              end='')
                    print('\n')

            # summary
            logger = logging.getLogger('log')

            if configs.multi_label > 1:
                # TODO add multi-label support
                pass
            else:
                # write python console message
                # todo do not concatenate str during iteration
                acc_str = ' *'
                for ii in range(len(configs.top_k)):
                    ii_str = ' Acc@{0} {1:.3f}'.format(configs.top_k[ii],
                                                       top_meter[ii].average)
                    acc_str += ii_str
                print(acc_str)

                # write log message
                log_str = ' validate-----*'
                for ii in range(len(configs.top_k)):
                    ii_str = ' Acc@{0} {1:.3f}'.format(configs.top_k[ii],
                                                       top_meter[ii].average)
                    log_str += ii_str
                log_str += ' Loss {:.4f}\n'.format(loss_meter.average)
                logger.info(log_str)

            if output_flag:
                if configs.multi_label > 1:
                    # TODO add multi-label support
                    pass
                else:
                    out_file.writelines('\n')
                    acc_str += ' Loss {:.4f}\n'.format(loss_meter.average)
                    out_file.writelines(acc_str)
                out_file.writelines('\n====== *** user config *** ======\n')
                out_file.writelines(pformat(configs.state_dict))
                out_file.writelines('\n========== *** end *** ==========\n')

    @staticmethod
    def _cal_accuracy(output, target, top_k=(1,)):
        """Calculate the accuracy over the top k predictions."""
        with torch.no_grad():
            max_k = max(top_k)
            batch_size = output.size(0)

            out_val, pred = output.topk(max_k, dim=1)
            correct = pred.eq(target.reshape([-1, 1]).expand_as(pred))

            result = []
            for k in top_k:
                n_correct_k = correct[:, :k].reshape(-1).sum()
                result.append(n_correct_k / batch_size * 100)

        return result, pred, out_val


class _AverageMeter(object):
    """
    Meters the current and overall average value.

    Attributes
    ----------
    value : float
        the current value
    sum : float
        the sum of the historical value
    count : int
        the number of the historical value
    average : float
        the average of the historical value

    Examples
    --------
    am = _AverageMeter()

    """

    # todo: rewrite docString
    def __init__(self):
        self.value = 0.
        self.sum = 0.
        self.count = 0
        self.average = 0.

    def reset(self):
        """Reset the Average Meter."""
        self.__init__()

    def update(self, value, increment=1):
        """
        Update the average meter.

        Parameters
        ----------
        value : float
            the current average value
        increment : int
            increment of count value

        """
        self.value = value
        self.sum += value * increment
        self.count += increment
        self.average = self.sum / self.count
