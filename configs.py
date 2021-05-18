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

    n_train_samples : int
        Number of training samples, use all samples if -1
    n_test_samples : int
        Number of test samples, use all samples if -1

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
    compression_factor : int
        compression factor of interval transition layers, use 3 by default
    drop_rate : float
        Drop rate in every dense blocks, use 0.5 by default
    block_config : tuple (of int)
        Layers in each dense block, use (8, 16, 16, 12) by default

    kind : str
        The identifier used in output (i.e. log and predict) file names,
        automatically get form data_dir if not specified

    optimizer_type : str
        Name of the optimizer used, now 'SGD' and 'Adam' are supported,
        use 'SGD' by default
    lr : float
        Start learning rate of the optimizer, use 0.6 by default
    weight_decay : float
        Weight decay (L2 penalty) of the optimizer, use 5E-5 by default
    lr_decay_rate : float
        Learning rate decay factor, use 0.33 by default

    label_list : list
        List of pipe labels

    train_batch_size : int
        Mini-batch size for training set, use 128 by default
    test_batch_size : int
        Mini-batch size for test set, use 64 by default
    n_train_workers : int
        Number of workers for data loading during training,
        may use number of available CPU
    n_test_workers : int
        Number of workers for data loading during testing,
        may use number of available CPU
    train_pin_memory : bool
        Whether use pin-memory during train data loading, use True by default,
        if seeing freeze or swap used a lot, use False
    test_pin_memory : bool
        Whether use pin-memory during test data loading, use True by default,
        if seeing freeze or swap used a lot, use False

    n_epoch : int
        Total epoch number, use 120 by default
    lr_adjust_at : list (of int)
        Epochs where the learning rate are changed by multiplying
        lr_decay_rate, use (20, 40, 60, 80, 100) by default

    logging_path : str
        Logging file path
    predict_path : str
        Predict file path
    top_k : tuple (of int)
        Top k accuracy will be calculated for each k in top_k,
        use (1, 2, 3, 4, 5) by default
    plot_every : int
        Interval of plotting training state in each epoch,
        use 30 by default

    """

    _OPTIMIZER_LIST = ('SGD', 'Adam')
    _LOG_PREFIX = 'log_'
    _PRED_SUFFIX = '_predict.txt'
    _LOG_FOLDER = 'log'
    _PRED_FOLDER = 'pred'

    def __init__(self, kwargs):
        # data directory
        self.data_dir = ''

        # specify this if test data in another directory is used
        self.test_data_dir = None
        self.test_data_name = None

        # number of samples, use all samples if remains 'default'
        self.n_train_samples = -1
        self.n_test_samples = -1

        # number of labels
        self.multi_label = 1

        # architecture of the FL-DenseNet
        self.in_features = 800
        self.out_classes = 33
        self.growth_rate = 128
        self.bottleneck_multiplier = 4
        self.compression_factor = 3
        self.drop_rate = 0.5
        self.block_config = (8, 16, 16, 12)

        # kind
        self.kind = None

        # optimizer
        self.optimizer_type = 'SGD'
        self.lr = 0.6
        self.weight_decay = 5E-5
        self.lr_decay_rate = 0.33

        # label list
        self.label_list = None

        # data loader settings
        self.train_batch_size = 128
        self.test_batch_size = 64
        self.n_train_workers = 8
        self.n_test_workers = 8
        self.train_pin_memory = True
        self.test_pin_memory = True

        # training epoch
        self.n_epoch = 120
        self.lr_adjust_at = (20, 40, 60, 80, 100)

        # output
        self.logging_path = None
        self.predict_path = None

        self.top_k = (1, 2, 3, 4, 5)
        self.plot_every = 30

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

        # automatically get test data name from test data directory
        if self.test_data_name is None:
            sep = os.sep
            tokens = self.test_data_dir.split(sep=sep)
            if tokens[-1] == '':
                self.test_data_name = tokens[-2]
            else:
                self.test_data_name = tokens[-1]

        # automatically get file name identifier from data directory
        if self.kind is None:
            sep = os.sep
            tokens = self.data_dir.split(sep=sep)
            if tokens[-1] == '':
                self.kind = tokens[-2]
            else:
                self.kind = tokens[-1]

        # automatically generate pipe label list if not specified
        if self.label_list is None:
            self.label_list = [(x + 1) for x in range(self.out_classes)]

        # create log folder
        if not os.path.exists(self._LOG_FOLDER):
            os.mkdir(self._LOG_FOLDER)

        # create pred folder
        if not os.path.exists(self._PRED_FOLDER):
            os.mkdir(self._PRED_FOLDER)

        # arrange logging and predict file path
        logging_name = '{0}{1}_{2}_lr_{3}'.format(self._LOG_PREFIX,
                                                  self.optimizer_type,
                                                  self.kind,
                                                  self.lr)
        if self.multi_label > 1:
            logging_name = '{0}_multi_label_{1:d}'.format(logging_name,
                                                          self.multi_label)
        if self.data_dir != self.test_data_dir:
            logging_name = '{0}_TestOn_{1}'.format(logging_name,
                                                   self.test_data_name)
        if self.n_train_samples is not None:
            logging_name = '{0}_num_train_{1:d}'.format(
                logging_name,
                int(self.n_train_samples)
            )
        if self.n_test_samples is not None:
            logging_name = '{0}_num_test_{1:d}'.format(
                logging_name,
                int(self.n_test_samples)
            )

        predict_name = logging_name + self._PRED_SUFFIX
        self.predict_path = os.path.join(self._PRED_FOLDER, predict_name)

        logging_name = '{}.log'.format(logging_name)
        self.logging_path = os.path.join(self._LOG_FOLDER, logging_name)

        # set logger
        logger = logging.getLogger('log')
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(self.logging_path, mode='a')
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt='%(asctime)s - %(filename)s[line:%(lineno)d] '
                '- %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # write log message
        logger.info('====== *** user config *** ======')
        logger.info(pformat(self.state_dict))
        logger.info('========== *** end *** ==========')

        # write console message
        print('====== *** user config *** ======')
        pprint(self.state_dict)
        print('========== *** end *** ==========')
