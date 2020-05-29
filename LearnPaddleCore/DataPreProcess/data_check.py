def data_check(imgs, labels):
    assert len(imgs) == len(labels), \
        'Length of train data should be same as length of label data'

