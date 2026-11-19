import numpy as np
import pandas as pd
import pytest

from starter.ml.data import clean_data, process_data

CATS = ["color", "shape"]


@pytest.fixture
def arrays():
    rng = np.random.RandomState(0)
    X = rng.rand(100, 5)
    y = (X[:, 0] > 0.5).astype(int)
    return X, y


@pytest.fixture
def df():
    rng = np.random.RandomState(1)
    n = 200
    d = pd.DataFrame(
        {
            "num": rng.rand(n),
            "color": rng.choice(["red", "green", "blue"], n),
            "shape": rng.choice(["sq", "circle"], n),
        }
    )
    d["salary"] = np.where(d["num"] > 0.5, ">50K", "<=50K")
    return d


def test_clean_data_fixes_common_problems():
    raw = pd.DataFrame(
        {
            " age": [30, 30, 40, 50],
            "workclass": [" Private", " Private", "?", " Gov"],
            "salary": [" <=50K.", " <=50K.", ">50K", ">50K"],
        }
    )
    out = clean_data(raw)
    assert list(out.columns) == [
        "age",
        "workclass",
        "salary",
    ]  # names stripped
    assert "?" not in out["workclass"].values  # missing row dropped
    assert len(out) == 2  # duplicate + '?' row removed
    assert set(out["salary"]) == {"<=50K", ">50K"}  # label normalised


def test_clean_data_unknown_strategy_keeps_rows():
    raw = pd.DataFrame(
        {"workclass": ["Private", "?"], "salary": ["<=50K", ">50K"]}
    )
    out = clean_data(raw, missing="unknown")
    assert len(out) == 2 and "Unknown" in out["workclass"].values


def test_clean_data_is_idempotent(df):
    once = clean_data(df)
    pd.testing.assert_frame_equal(once, clean_data(once))


def test_process_data_train_then_inference_shapes(df):
    X, y, enc, lb = process_data(df, CATS, label="salary", training=True)
    assert X.shape[0] == len(df) and y.shape == (len(df),)
    # 1 numeric + 3 colors + 2 shapes
    assert X.shape[1] == 1 + 3 + 2
    X2, y2, _, _ = process_data(
        df, CATS, label="salary", training=False, encoder=enc, lb=lb
    )
    assert X2.shape == X.shape
    np.testing.assert_array_equal(y, y2)


def test_process_data_without_label_returns_empty_y(df):
    _, _, enc, lb = process_data(df, CATS, label="salary", training=True)
    X, y, _, _ = process_data(
        df.drop(columns="salary"),
        CATS,
        label=None,
        training=False,
        encoder=enc,
        lb=lb,
    )
    assert y.size == 0 and X.shape[0] == len(df)
