import logging
import os

from pprint import pprint, pformat


class Configs(object):
    """
    Configs used in frf-net.

    This should be instantiated like this:

        cfg = Config(kwarg_dict)
            with kwarg_dict, a dict of attribute keywords
            and corresponding arguments,

            the available attributes are listed as below.

    Attributes
    ----------
    data_dir : str (Optional)
        Directory of the training dataset (and test dataset)
    data_name : str (Optional)
        Automatically get form test_data_dir if not specified
    kind : str (Optional)
        The identifier used in output (i.e. log and predict) file names,
        automatically get form data_dir if not specified

    test_data_dir : str (Optional)
        Test dataset directory, use data_dir if not specified
    test_data_name : str (Optional)
        Test data name, automatically get form test_data_dir if not specified

    n_train_samples : int (Optional)
        Number of training samples, use all samples if 'all'
    n_test_samples : int (Optional)
        Number of test samples, use all samples if 'all'

    multi_label : int (Optional)
        Maximum number of leak pipe label(s) in each sample
    rounding_threshold : float (Optional)
        Threshold for rounding output in multi-label task

    in_features : int
        Number of input features of the fully linear DenseNet
    out_classes : int
        Number of output features of the fully linear DenseNet

    growth_rate : int (Optional)
        Growth rate of dense blocks, use 128 by default
    bottleneck_multiplier : int (Optional)
        Bottleneck multiplier in every dense blocks, use 4 by default
    compression_factor : int (Optional)
        compression factor of interval transition layers, use 3 by default
    drop_rate : float (Optional)
        Drop rate in every dense blocks, use 0.5 by default
    block_config : tuple (of int) (Optional)
        Layers in each dense block, use (8, 16, 16, 12) by default

    optimizer_type : str (Optional)
        Name of the optimizer used, now 'SGD' and 'Adam' are supported,
        use 'SGD' by default
    initial_lr : float (Optional)
        Start learning rate of the optimizer, use 0.6 by default
    weight_decay : float (Optional)
        Weight decay (L2 penalty) of the optimizer, use 5E-5 by default
    lr_decay_rate : float (Optional)
        Learning rate decay factor, use 0.33 by default

    label_list : str list (Optional)
        List of pipe labels

    train_batch_size : int (Optional)
        Mini-batch size for training set, use 128 by default
    test_batch_size : int (Optional)
        Mini-batch size for test set, use 64 by default
    n_train_workers : int (Optional)
        Number of workers for data loading during training,
        may use number of available CPU
    n_test_workers : int (Optional)
        Number of workers for data loading during testing,
        may use number of available CPU
    train_pin_memory : bool (Optional)
        Whether use pin-memory during train data loading, use True by default,
        if seeing freeze or swap used a lot, use False
    test_pin_memory : bool (Optional)
        Whether use pin-memory during test data loading, use True by default,
        if seeing freeze or swap used a lot, use False

    n_epoch : int (Optional)
        Total epoch number, use 120 by default
    lr_adjust_at : list (of int) (Optional)
        Epochs where the learning rate are changed by multiplying
        lr_decay_rate, use (20, 40, 60, 80, 100) by default

    logging_path : str (Optional)
        Logging file path
    predict_path : str (Optional)
        Predict file path

    top_k : tuple (of int) (Optional)
        Top k accuracy will be calculated for each k in top_k,
        use (1, 2, 3, 4, 5) by default
    plot_every : int (Optional)
        Interval of plotting training state in each epoch,
        use 30 by default

    """

    _OPTIMIZER_LIST = ('SGD', 'Adam')
    _LOG_PREFIX = 'log_'
    _PRED_SUFFIX = '_predict.txt'
    _LOG_FOLDER = 'log'
    _PRED_FOLDER = 'pred'
    _MANDATORY_KEYWORDS = ('in_features', 'out_classes')

    def __init__(self, kwargs):
        # data directory
        self.data_dir = 'default'
        self.data_name = None
        self.kind = None

        # specify this if test data in another directory is used
        self.test_data_dir = None
        self.test_data_name = None

        # number of samples, use all samples if remains 0
        self.n_train_samples = 'all'
        self.n_test_samples = 'all'

        # number of labels
        self.multi_label = 1
        self.rounding_threshold = 0.5

        # architecture of the FL-DenseNet
        self.in_features = 800
        self.out_classes = 33
        self.growth_rate = 128
        self.bottleneck_multiplier = 4
        self.compression_factor = 3
        self.drop_rate = 0.5
        self.block_config = (8, 16, 16, 12)

        # optimizer
        self.optimizer_type = 'SGD'
        self.initial_lr = 0.6
        self.weight_decay = 5E-5
        self.lr_decay_rate = 0.33

        # label list
        self.label_list = None

        # data loader settings
        self.train_batch_size = 128
        self.test_batch_size = 64
        self.n_train_workers = 0
        self.n_test_workers = 0
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

        self._lr = 0.  # current learning rate

        # check existence of mandatory arguments
        for kw in self._MANDATORY_KEYWORDS:
            if kw not in kwargs:
                raise ValueError('Option \'{}\' is mandatory'.format(kw))

        # parse inputs
        self.parse_params(kwargs)

    def _state_dict(self):
        return {k: getattr(self, k) for (k, _)
                in self.__dict__.items() if not k.startswith('_')}

    @property
    def state_dict(self):
        return self._state_dict()

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        self._lr = value

    def parse_params(self, kwargs):
        """Parse keyword-argument pairs for Configs."""
        # parse all input arguments
        for (k, v) in kwargs.items():
            if k not in self.state_dict:
                raise ValueError('Unknown Option: \'--{}\''.format(k))
            setattr(self, k, v)

        # check validity of optimizer_type
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
                kind = tokens[-2]
            else:
                kind = tokens[-1]

            self.data_name = kind

            kind = '{0}_{1}_lr_{2}_nEpoch_{3}'.format(
                self.optimizer_type,
                kind, self.initial_lr, self.n_epoch
            )

            if self.multi_label > 1:
                kind = '{0}_multi_label_{1:d}'.format(kind,
                                                      self.multi_label)

            if self.n_train_samples != 'all':
                kind = '{0}_num_train_{1:d}'.format(
                    kind,
                    int(self.n_train_samples)
                )

            self.kind = kind

        # automatically generate pipe label list if not specified
        if self.label_list is None:
            self.label_list = [str(x + 1) for x in range(self.out_classes)]

        # disable option not used
        if self.multi_label > 1:
            self.top_k = ()
        else:
            self.rounding_threshold = 'NaN'

        # set start learning rate
        self.lr = self.initial_lr

        # create log folder
        if not os.path.exists(self._LOG_FOLDER):
            os.mkdir(self._LOG_FOLDER)

        # create pred folder
        if not os.path.exists(self._PRED_FOLDER):
            os.mkdir(self._PRED_FOLDER)

        # arrange logging and predict file path
        logging_name = self._LOG_PREFIX + self.kind

        if self.data_name != self.test_data_name:
            logging_name = '{0}_TestBy_{1}'.format(logging_name,
                                                   self.test_data_name)

        if self.n_test_samples != 'all':
            logging_name = '{0}_num_test_{1:d}'.format(
                logging_name,
                int(self.n_test_samples)
            )

        predict_name = logging_name + self._PRED_SUFFIX
        if self.predict_path is None:
            self.predict_path = os.path.join(self._PRED_FOLDER, predict_name)

        logging_name = '{}.log'.format(logging_name)
        if self.logging_path is None:
            self.logging_path = os.path.join(self._LOG_FOLDER, logging_name)

    def create_logger(self):
        logger = logging.getLogger('log')
        logger.setLevel(logging.INFO)

        if len(logger.handlers) != 0:
            logger.handlers = []  # reset handler list

        handler = logging.FileHandler(self.logging_path, mode='a')
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt='%(asctime)s - %(filename)s[line:%(lineno)d] '
                '- %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def write_log_msg(self):
        """Write log messages."""
        logger = logging.getLogger('log')

        logger.info('====== *** user config *** ======')
        logger.info(pformat(self.state_dict))
        logger.info('========== *** end *** ==========\n')

    def write_console_msg(self):
        """write console messages."""
        print('====== *** user config *** ======')
        pprint(self.state_dict)
        print('========== *** end *** ==========\n')
