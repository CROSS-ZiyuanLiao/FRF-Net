import logging
import time
import torch
import torch.nn.functional as F

from pprint import pformat
from torch.utils.data.dataloader import DataLoader

from configs import Configs
from fl_densenet import FullyLinearDenseNet
from frf_dataset import FRFDataset
from trainer import Trainer


class FRFNet(object):
    """
    Extract features in FRF with FL-DenseNet.

    This should be instantiated like this:

        frf_net = FRFNet(
            data_dir='%MyDataDir%'
            in_features=640,
            out_classes=33,
            label_list=['P1', 'P2', ..., 'P33']
            kw1=arg1,
            kw2=arg2,
            ...
        )

        keywords in_features and out_classes are mandatory

        you can omit data_dir if you are going to load a pre-trained model

        label_list should contains and only contains all possible labels,
        if it is something just like
        [str(x + 1) for x in range(self.out_classes)],
        you can omit this keyword,
        otherwise you should explicitly define it

        see Configs in configs.py for more available keywords.

    Attributes
    ----------
    configs : Configs (see configs.py)
    model : FullyLinearDenseNet (see fl_densenet.py)
    train_dataset : FRFDataset (see frf_dataset.py)
    test_dataset : FRFDataset (see frf_dataset.py)
    train_data_loader : DataLoader (in torch.utils.data.dataloader)
    test_data_loader : DataLoader (in torch.utils.data.dataloader)
    trainer : Trainer (see trainer.py)

    """

    def __init__(self, **kwargs):
        print('Constructing model\n')

        self.configs = Configs(kwargs)
        configs = self.configs

        # build model
        model = FullyLinearDenseNet(
            configs.in_features,
            configs.out_classes,
            growth_rate=configs.growth_rate,
            bottleneck_multiplier=configs.bottleneck_multiplier,
            compression_factor=configs.compression_factor,
            drop_rate=configs.drop_rate,
            block_config=configs.block_config
        )
        model.apply(model.init_weights)
        self.model = model

        # build trainer
        self.trainer = Trainer(configs, model)

        # data set
        self.train_dataset = None
        self.test_dataset = None
        self.train_data_loader = None
        self.test_data_loader = None

        # move the model to GPU and then build optimizer
        if torch.cuda.is_available():
            self.trainer.cuda()
        self.trainer.get_optimizer()

    def _create_meters(self):
        self._batch_time_meter = _AverageMeter()  # batch elapsed time
        self._data_time_meter = _AverageMeter()  # data loading time
        self._loss_meter = _AverageMeter()  # loss value
        self._top_k_meters = [_AverageMeter() for _ in self.configs.top_k]
        self._hamming_meter = _AverageMeter()  # hamming accuracy
        self._tp_meter = _SumMeter()  # true positive
        self._fp_meter = _SumMeter()  # false positive
        self._fn_meter = _SumMeter()  # false negative

    def _build_train_data_loader(self):
        configs = self.configs
        self.train_dataset = FRFDataset(configs, 'train')
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=configs.train_batch_size,
            num_workers=configs.n_train_workers,
            pin_memory=configs.train_pin_memory
        )

    def _build_test_data_loader(self):
        configs = self.configs
        self.test_dataset = FRFDataset(configs, 'test')
        self.test_data_loader = DataLoader(
            self.test_dataset,
            batch_size=configs.test_batch_size,
            num_workers=configs.n_test_workers,
            pin_memory=configs.test_pin_memory
        )

    @staticmethod
    def _cal_accuracy(output, target, top_k=(1,)):
        """Calculate the accuracy over the top k predictions."""
        with torch.no_grad():
            max_k = max(top_k)
            batch_size = output.size(0)

            out_val, pred = output.topk(max_k, dim=1)
            correct = pred.eq(target.reshape([-1, 1]).expand_as(pred))

            acc = []
            for k in top_k:
                n_correct_k = correct[:, :k].reshape(-1).sum()
                acc.append(n_correct_k / batch_size * 100)

        return acc, pred, out_val

    @staticmethod
    def _cal_multi_accuracy(output, target, threshold=0.5):
        """Calculate the accuracy of multi-label prediction."""
        with torch.no_grad():
            batch_size = output.size(0)
            multi_label_size = output.size(1)

            target = target.tolist()

            # hamming acc, true positive, false positive, false negative
            acc, tp, fp, fn = 0, 0, 0, 0

            acc_list = []
            pred = []
            for i, output_i in enumerate(output):
                acc_i = 0
                pred_i = [0. for _ in range(multi_label_size)]
                for j, output_ij in enumerate(output_i):
                    if output_ij > threshold:
                        pred_i[j] = 1.
                        if target[i][j] == 1.:
                            tp += 1
                            acc_i += 1
                            acc += 1
                        else:
                            fp += 1
                    else:
                        if target[i][j] == 0.:
                            acc_i += 1
                            acc += 1
                        else:
                            fn += 1
                acc_list.append(acc_i / multi_label_size)
                pred.append(pred_i)

            acc = acc / (batch_size * multi_label_size)

        return acc, acc_list, pred, tp, fp, fn

    def _evaluate(self, batch_size, output, target):
        configs = self.configs

        acc, pred, out_val = self._cal_accuracy(
            output, target, top_k=configs.top_k
        )
        for ii in range(len(configs.top_k)):
            self._top_k_meters[ii].update(acc[ii], batch_size)

        return pred.tolist(), out_val.tolist()

    def _evaluate_multi(self, batch_size, output, target):
        acc, acc_list, pred, tp, fp, fn = self._cal_multi_accuracy(
            output, target, threshold=self.configs.rounding_threshold
        )

        self._hamming_meter.update(acc, batch_size)
        self._tp_meter.update(tp)
        self._fp_meter.update(fp)
        self._fn_meter.update(fn)

        return pred, acc_list

    def _console_plot(self, epoch, i):
        """Plot in console, epoch is -1 for validation."""
        if epoch == -1:
            msg_str = [('Test: [{0}/{1}]\t'
                        'Time: {2:.3f} ({3:.3f})\t'
                        'Loss: {4:.4f} ({5:.4f})\t'
                        .format((i + 1), len(self.test_data_loader),
                                self._batch_time_meter.value,
                                self._batch_time_meter.average,
                                self._loss_meter.value,
                                self._loss_meter.average))]
        else:
            msg_str = [('Epoch: [{0}][{1}/{2}]\t'
                        'Time: {3:.3f} ({4:.3f})\t'
                        'Data: {5:.3f} ({6:.3f})\t'
                        'Loss: {7:.4f} ({8:.4f})\t'
                        .format((epoch + 1), (i + 1),
                                len(self.train_data_loader),
                                self._batch_time_meter.value,
                                self._batch_time_meter.average,
                                self._data_time_meter.value,
                                self._data_time_meter.average,
                                self._loss_meter.value,
                                self._loss_meter.average))]
        if self.configs.multi_label > 1:
            tp = self._tp_meter.value
            fp = self._fp_meter.value
            fn = self._fn_meter.value
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            msg_str.append(
                'hamming_acc: {0:.3f} ({1:.3f})\t'
                'precision: {2:.3f}\t'
                'recall: {3:.3f}\t'.format(
                    self._hamming_meter.value, self._hamming_meter.average,
                    precision,
                    recall
                )
            )
        else:
            for ii, v in enumerate(self.configs.top_k):
                msg_str.append('Acc@{0} {1:.3f} ({2:.3f})\t'.format(
                        v,
                        self._top_k_meters[ii].value,
                        self._top_k_meters[ii].average
                ))
        print(''.join(msg_str))
        print('\n')

    def _train_logging_plot(self, epoch, i):
        logger = logging.getLogger('log')
        log_str = [' train-----* ===Epoch: [{0}][{1}/{2}]\t'.format(
            (epoch + 1), (i + 1), len(self.train_data_loader))]
        if self.configs.multi_label > 1:
            tp = self._tp_meter.value
            fp = self._fp_meter.value
            fn = self._fn_meter.value
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            log_str.append(
                'hamming_acc: {0:.3f}\t'
                'precision: {1:.3f}\t'
                'recall: {2:.3f}\t'.format(self._hamming_meter.value,
                                           precision, recall)
            )
        else:
            for ii, v in enumerate(self.configs.top_k):
                log_str.append(' Acc@{0} {1:.3f}'.format(
                    v,
                    self._top_k_meters[ii].average
                ))
        logger.info(''.join(log_str))

    def train(self, epoch):
        """
        Train the model.

        Parameters
        ----------
        epoch : int
            The current epoch number.

        """
        print('epoch {}'.format(epoch + 1))
        configs = self.configs
        batch_size = configs.train_batch_size

        # create meters
        self._create_meters()

        # switch to train mode
        self.model.train()

        # start loss value
        old_loss = 100

        # load data and train
        start_time = time.time()
        for i, (target, data) in enumerate(self.train_data_loader):
            # load data to GPU
            if torch.cuda.is_available():
                target, data = target.cuda(), data.cuda()

            # measure data loading time
            self._data_time_meter.update(time.time() - start_time)

            # take a train step
            output, loss = self.trainer.train_step(target, data)

            # update loss
            loss_meter = self._loss_meter
            loss_meter.update(loss.item(), batch_size)
            if loss_meter.value < old_loss:
                old_loss = loss_meter.value
                print('==== iter *{}* ====, '
                      '========== *** loss update: {} *** ==========\n'
                      .format((i + 1), old_loss))

            # evaluate accuracy
            if configs.multi_label > 1:
                self._evaluate_multi(batch_size, output, target)
            else:
                self._evaluate(batch_size, output, target)

            # update batch elapsed time
            self._batch_time_meter.update(time.time() - start_time)

            # plot
            if (i + 1) % configs.plot_every == 0:
                self._console_plot(epoch, i)
                self._train_logging_plot(epoch, i)

            # update start time for the next load
            start_time = time.time()

    def _validate_write_summary(self, out_file):
        configs = self.configs
        logger = logging.getLogger('log')

        msg_str = [' *']
        log_str = [' validate-----*']

        if configs.multi_label > 1:
            tp = self._tp_meter.sum
            fp = self._fp_meter.sum
            fn = self._fn_meter.sum
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            some_str = (' hamming_acc: {0:.3f}'
                        ' precision: {1:.3f}'
                        ' recall: {2:.3f}'.format(self._hamming_meter.average,
                                                  precision, recall))
            msg_str.append(some_str)
            msg_str.append(' Loss {:.4f}\n'.format(
                self._loss_meter.average
            ))
            log_str.append(some_str)
        else:
            for ii, v in enumerate(configs.top_k):
                ii_str = ' Acc@{0} {1:.3f}'.format(
                    v,
                    self._top_k_meters[ii].average
                )
                msg_str.append(ii_str)
                log_str.append(ii_str)
            msg_str.append(' Loss {:.4f}\n'.format(
                self._loss_meter.average
            ))

        # write messages
        print(''.join(msg_str))

        out_file.writelines('\n')
        out_file.writelines(msg_str)

        logger.info(''.join(msg_str))

        # write user config
        out_file.writelines('\n====== *** user config *** ======\n')
        out_file.writelines(pformat(configs.state_dict))
        out_file.writelines('\n========== *** end *** ==========\n')

    @torch.no_grad()
    def validate(self, criterion):
        """Validate the model."""
        print('validating')
        configs = self.configs
        batch_size = configs.train_batch_size

        # create meters
        self._create_meters()

        # switch to evaluation mode
        self.model.eval()

        # output file
        out_path = configs.predict_path
        with open(out_path, 'w') as out_file:

            # load data and evaluate
            start_time = time.time()
            for i, (target, data) in enumerate(self.test_data_loader):
                # load data
                if torch.cuda.is_available():
                    target, data = target.cuda(), data.cuda()

                # validation
                output = self.model(data)
                loss = criterion(output, target)

                # update loss
                self._loss_meter.update(loss.item(), batch_size)

                # evaluate accuracy
                target_list = target.tolist()
                if self.configs.multi_label > 1:
                    output = torch.sigmoid(output)
                    pred, acc_list = self._evaluate_multi(
                        batch_size, output, target
                    )

                    # write details of prediction for each sample
                    for ii, item in enumerate(pred):
                        lines = ['prediction: ', str(item), ', ',
                                 'target: ', str(target_list[ii]), ', ',
                                 'hamming_acc: ', str(acc_list[ii]),
                                 '\n']
                        out_file.writelines(lines)

                else:
                    output = F.log_softmax(output, dim=1)
                    pred, out_val = self._evaluate(batch_size, output, target)

                    # write details of prediction for each sample
                    for ii, item in enumerate(pred):
                        lines = ['output value: ', str(out_val[ii]), ', '
                                 'prediction: ', str(item), ', ',
                                 'target: ', str(target_list[ii]),
                                 '\n'
                                 ]
                        out_file.writelines(lines)

                # update batch elapsed time
                self._batch_time_meter.update(time.time() - start_time)

                # plot
                if (i + 1) % configs.plot_every == 0:
                    self._console_plot(-1, i)

            # summary
            self._validate_write_summary(out_file)

    def _get_criterion(self):
        if self.configs.multi_label > 1:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        return criterion

    def _write_msgs(self):
        configs = self.configs
        configs.create_logger()
        configs.write_log_msg()
        configs.write_console_msg()

    def train_and_validate(self):
        """Train and validate."""
        self._write_msgs()

        self._build_train_data_loader()
        self._build_test_data_loader()
        print('Construction complete\n')

        # define criterion for evaluation
        criterion = self._get_criterion()

        # training
        for epoch in range(self.configs.n_epoch):
            if (epoch + 1) in self.configs.lr_adjust_at:
                self.trainer.adjust_lr()
            self.train(epoch)

        # validation
        self.validate(criterion)

    def load_and_validate(self, param_path, test_data_dir, **kwargs):
        """
        Load model and validate.

        Parameters
        ----------
        param_path : str
            Path of the file that contains model parameters and configs
        test_data_dir : str
            Path of test data
        **kwargs
            These keyword arguments can change some test-related configs
            before validation, for more details,
            see Trainer.load in trainer.py

        """
        self.trainer.load(param_path, test_data_dir, kwargs)
        self._write_msgs()

        self._build_test_data_loader()
        print('Construction complete\n')

        # validation
        criterion = self._get_criterion()
        self.validate(criterion)

    def save_model(self):
        """Save current model and configs."""
        self.trainer.save()


class _AverageMeter(object):
    """
    Meters the current and overall average value.

    Attributes
    ----------
    value : float
        The current average value
    average : float
        The overall average value

    """

    def __init__(self):
        self.value = 0.
        self._sum = 0.
        self._count = 0
        self.average = 0.

    def reset(self):
        """Reset the average meter."""
        self.__init__()

    def update(self, value, increment=1):
        """
        Update the average meter.

        Parameters
        ----------
        value : float
            The current average value
        increment : int
            Increment of count value

        """
        self.value = value
        self._sum += value * increment
        self._count += increment
        self.average = self._sum / self._count


class _SumMeter(object):
    """
    Meters the current value and sum value.

    Attributes
    ----------
    value : float
        The current value
    sum : float
        The sum of all values

    """

    def __init__(self):
        self.value = 0.
        self.sum = 0.

    def reset(self):
        """Reset the sum meter."""
        self.__init__()

    def update(self, value):
        """
        Update the sum meter.

        Parameters
        ----------
        value : float
            the current value

        """
        self.value = value
        self.sum += value
