from frf_net import FRFNet

if __name__ == '__main__':
    # in_features = 640
    # out_classes = 12
    in_features = 800
    out_classes = 33

    f_net = FRFNet(
        data_dir='/opt/data/private/Lab_hypothetical/dataset/lab_4_0005_f_R1/',
        test_data_dir='/opt/data/private/Lab_hypothetical/dataset/lab_4_0005_f_R2/',
        # data_dir='/opt/data/private/Paper1/191220_MultipleLeaks',
        in_features=in_features,
        out_classes=out_classes,
        n_train_workers=0,  # 0 for debugging
        n_test_workers=0,  # 0 for debugging
        # optimizer_type='SGD',
        # multi_label=3,
        n_epoch=2,
        lr_adjust_at=(1,),
    )
    f_net.main_worker()
