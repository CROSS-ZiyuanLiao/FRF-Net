import logging
import os

from pprint import pprint, pformat


class Configs(object):
    """
    Configs used in frf-net.

    This should be instantiated like this:

        cfg = Config(kwarg_dict)
            with kwarg_dict, a dict of attribute keywords
            and corresponding arguments, the available attributes
            are listed as below

    Attributes
    ----------
    data_dir : str
        Directory of the training dataset (and test dataset)
    test_data_dir : str
        Test dataset directory, use data_dir if not specified
    test_data_name : str
        Test data name, automatically get form test_data_dir if not specified

    n_train_samples : NoneType / int
        Number of training samples, use all samples if None
    n_test_samples : NoneType / int
        Number of test samples, use all samples if None

    multi_label : int
        Maximum number of leak pipe label(s) in each sample

    in_features : int
        Number of input features of the fully linear DenseNet
    out_classes : int
        Number of output features of the fully linear DenseNet

    growth_rate : int
        Growth rate of dense blocks, use 128 by default
    bottleneck_multiplier : int
        Bottleneck multiplier in every dense blocks, use 4 by default
    drop_rate : float
        Drop rate in every dense blocks, use 0.5 by default
    block_config : tuple
        Layers in each dense block, use (8, 16, 16, 12) by default

    kind : str

    optimizer_type : str
        Name of the optimizer used, now 'SGD' and 'Adam' are supported
    lr : float
        Learning rate of the optimizer,
    weight_decay : float
        Weight decay (L2 penalty) of the optimizer
    lr_decay_rate : float
        Learning rate decay factor

    label_list : list
        List of pipe labels

    train_batch_size : int
        Mini-batch size for training set, use 128 by default
    test_batch_size : int
        Mini-batch size for test set, use 64 by default
    n_train_workers : int
        Number of workers for data loading during training
    n_test_workers : int
        Number of workers for data loading during testing
    train_pin_memory : bool
        Whether use pin-memory during train data loading, use True by default,
        if seeing freeze or swap used a lot, use False
    test_pin_memory : bool
        Whether use pin-memory during test data loading, use True by default,
        if seeing freeze or swap used a lot, use False

    """

    # todo: docStrings
    _OPTIMIZER_LIST = ('SGD', 'Adam')

    def __init__(self, kwargs={}):
        # data directory
        self.data_dir = ''

        # specify this if test data in another directory is used
        self.test_data_dir = None
        self.test_data_name = None

        # number of samples, use all samples if remains 'default'
        self.n_train_samples = None
        self.n_test_samples = None

        # number of labels
        self.multi_label = 1

        # architecture of the FL-DenseNet
        self.in_features = 800
        self.out_classes = 33
        self.growth_rate = 128   # 128 by default
        self.bottleneck_multiplier = 4  # 4 by default
        # todo self.compression_factor = 3
        self.drop_rate = 0.5  # 0.5 by default
        self.block_config = (8, 16, 16, 12)  # (8, 16, 16, 12) by default

        # kind
        self.kind = None

        # optimizer
        self.optimizer_type = 'SGD'
        self.lr = 0.6  # learning rate, 0.6 by default
        self.weight_decay = 5E-5  # 5E-5 by default
        self.lr_decay_rate = 0.33  # 0.33 by default

        # label list
        self.label_list = None

        # data loader settings
        self.train_batch_size = 128  # 128 by default
        self.test_batch_size = 64  # 64 by default
        self.n_train_workers = 8  # may use CPU number
        self.n_test_workers = 8  # may use CPU number
        self.train_pin_memory = True  # should use True if do not freeze
        self.test_pin_memory = True  # should use True if do not freeze

        # training epoch
        self.n_epoch = 120  # 120 by default
        self.lr_adjust_at = (
            20, 40, 60, 80, 100)  # (20, 40, 60, 80, 100) by default

        # output
        self.logging_name = 'log_'  # do not change
        self.predict_path = '_predict.txt'  # do not change

        self.top_k = (1, 3, 5)  # (1, 2, 3, 4, 5) by default
        self.plot_every = 30  # console output during each epoch process

        # parse inputs
        self._parse_params(kwargs)

    def _state_dict(self):
        return {k: getattr(self, k) for (k, _)
                in self.__dict__.items() if not k.startswith('_')}

    @property
    def state_dict(self):
        return self._state_dict()

    def _parse_params(self, kwargs):
        for (k, v) in kwargs.items():
            if k not in self.state_dict:
                raise ValueError('Unknown Option: \'--{}\''.format(k))
            setattr(self, k, v)

        if self.optimizer_type not in self._OPTIMIZER_LIST:
            raise ValueError('Unknown Optimizer: \'{}\''
                             .format(self.optimizer_type))

        # use data directory if test data directory not specified
        if self.test_data_dir is None:
            self.test_data_dir = self.data_dir

        # get test data name from test data directory
        if self.test_data_name is None:
            sep = os.sep
            tokens = self.test_data_dir.split(sep=sep)
            if tokens[-1] == '':
                self.test_data_name = tokens[-2]
            else:
                self.test_data_name = tokens[-1]

        if self.kind is None:
            sep = os.sep
            tokens = self.data_dir.split(sep=sep)
            if tokens[-1] == '':
                self.kind = tokens[-2]
            else:
                self.kind = tokens[-1]

        if self.label_list is None:
            self.label_list = [(x + 1) for x in range(self.out_classes)]

        log_folder = 'log'
        pred_folder = 'pred'

        if not os.path.exists(log_folder):
            os.mkdir(log_folder)

        if not os.path.exists(pred_folder):
            os.mkdir(pred_folder)

        # arrange logging file name
        self.logging_name = '{0}{1}_{2}_lr_{3}'.format(
            self.logging_name,
            self.optimizer_type,
            self.kind,
            self.lr
        )

        if self.multi_label > 1:
            # TODO add multi-label support
            pass

        if self.data_dir != self.test_data_dir:
            self.logging_name = '{0}_TestOn_{1}'.format(
                self.logging_name,
                self.test_data_name
            )

        if self.n_train_samples is not None:
            self.logging_name = '{0}_num_train_{1:d}'.format(
                self.logging_name,
                int(self.n_train_samples)
            )

        if self.n_test_samples is not None:
            self.logging_name = '{0}_num_test_{1:d}'.format(
                self.logging_name,
                int(self.n_test_samples)
            )

        self.predict_path = self.logging_name + self.predict_path
        self.predict_path = os.path.join(pred_folder, self.predict_path)

        self.logging_name = '{}.log'.format(self.logging_name)
        logging_path = os.path.join(log_folder, self.logging_name)

        # write console message
        print('====== *** user config *** ======')
        pprint(self.state_dict)
        print('========== *** end *** ==========')

        # set logger
        logger = logging.getLogger('log')
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(logging_path, mode='a')
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt='%(asctime)s - %(filename)s[line:%(lineno)d]'
                ' - %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # write log message
        logger.info('====== *** user config *** ======')
        logger.info(pformat(self.state_dict))
        logger.info('========== *** end *** ==========')
