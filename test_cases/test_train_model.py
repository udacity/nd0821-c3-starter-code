import os


def test_train_files():
    assert os.path.isfile(os.path.join('model', 'encoder_dtc.pkl'))
    assert os.path.isfile(os.path.join('model', 'lb_dtc.pkl'))
    assert os.path.isfile(os.path.join('model', 'model_dtc.pkl'))