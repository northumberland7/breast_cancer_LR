"""对 core.py 的测试。"""

import numpy as np
import pandas as pd
import pytest

import core


# ---------- 正常流程 ----------

def test_load_data_shape():
    """数据集应该是 569 行 30 列，357 个良性。"""
    X, y = core.load_data()
    assert X.shape == (569, 30)
    assert len(y) == 569
    assert int(y.sum()) == 357


def test_split_is_reproducible():
    """同一个 random_state 切两次，结果必须完全一样。"""
    X, y = core.load_data()
    first = core.split_data(X, y)
    second = core.split_data(X, y)
    assert list(first[0].index) == list(second[0].index)


def test_model_can_predict_and_score():
    """训练、预测、评估整条链路能走通。"""
    X, y = core.load_data()
    X_train, X_test, y_train, y_test = core.split_data(X, y)
    model, y_pred = core.train_and_predict(X_train, y_train, X_test)
    assert set(np.unique(y_pred)).issubset({0, 1})
    acc = core.evaluate(y_test, y_pred)
    assert 0.9 < acc <= 1.0


# ---------- 异常情况 ----------

def test_empty_data_raises():
    """空数据要报错。"""
    X = pd.DataFrame(columns=['a', 'b'])
    y = pd.Series(dtype=int)
    with pytest.raises(ValueError, match='数据为空'):
        core.check_data(X, y)


def test_length_mismatch_raises():
    """特征 5 行、标签 3 行，要报错。"""
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 2, 3, 4, 5]})
    y = pd.Series([0, 1, 0])
    with pytest.raises(ValueError, match='行数不一致'):
        core.check_data(X, y)


def test_invalid_label_raises():
    """标签里出现 2，要报错。"""
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3]})
    y = pd.Series([0, 1, 2])
    with pytest.raises(ValueError, match='只能是 0 或 1'):
        core.check_data(X, y)


def test_non_positive_C_raises():
    """C 小于等于 0，要报错。"""
    with pytest.raises(ValueError, match='C 必须大于 0'):
        core.build_model(C=0)


def test_evaluate_length_mismatch_raises():
    """真实标签 4 个、预测 3 个，要报错。"""
    with pytest.raises(ValueError, match='长度不一致'):
        core.evaluate(np.array([0, 1, 0, 1]), np.array([0, 1]))