from frf_net import FRFNet

if __name__ == '__main__':
    in_features = 640
    out_classes = 12
    # in_features = 800
    # out_classes = 33

    train = 0

    if train:
        f_net = FRFNet(
            # # Windows
            # data_dir='D:\\191220_SpeedAndFric_R1S1\\',

            # # MacOS
            # data_dir='/Users/Cross/Downloads/DATA/191220_MultipleLeaks',
            # data_dir='/Users/Cross/Downloads/DATA/191220_SpeedAndFric_R1S1',

            # # Linux
            # data_dir='/opt/data/private/Lab_hypothetical/dataset/lab_4_0005_f_R1/',
            # data_dir='/opt/data/private/Paper1/191220_MultipleLeaks',
            data_dir='/opt/data/private/Paper1/200115_LeakScale_M5',
            # data_dir='nonsense',

            # test_data_dir='/opt/data/private/Lab_hypothetical/dataset/lab_4_0005_f_R2/',

            in_features=in_features,
            out_classes=out_classes,
            n_train_samples=1280,  # for debugging
            # n_test_samples=640,  # for debugging

            # n_train_workers=16,  # comment this line for debugging or on Windows
            # n_test_workers=16,  # comment this line for debugging or on Windows
            # train_pin_memory=False,
            # test_pin_memory=False,

            # multi_label=3,

            n_epoch=2,  # for debugging
            lr_adjust_at=(1, 2),  # for debugging
            # plot_every=1,  # for debugging
        )
        f_net.train_and_validate()
        f_net.save_model()

    else:
        f_net = FRFNet(
            in_features=in_features,
            out_classes=out_classes
        )
        f_net.load_and_validate(
            # param_path='param/SGD_191220_MultipleLeaks_lr_0.6_nEpoch_120_multi_label_3.pt',
            param_path='param/SGD_200115_LeakScale_M5_lr_0.6_nEpoch_2_num_train_1280.pt',
            # test_data_dir='/opt/data/private/Paper1/191220_MultipleLeaks',
            # test_data_dir='/opt/data/private/Paper1/210421_GA_compare_origin',
            test_data_dir='/opt/data/private/Paper1/200115_LeakScale_M5',
            test_data_name='otherName',
            # n_test_workers=16,
            # n_test_samples=5000,
            top_k=(1, 2)
        )
