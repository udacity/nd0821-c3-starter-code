import os


def test_train_files():
    assert os.path.isfile(os.path.join('.', '.', 'starter','model', 'encoder_dtc.pkl'))
    assert os.path.isfile(os.path.join('.', '.', 'starter', 'model', 'lb_dtc.pkl'))
    assert os.path.isfile(os.path.join('.', '.', 'starter', 'model', 'model_dtc.pkl'))