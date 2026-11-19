#这个文件用来执行异常处理和测试
"""可测试的函数：数据校验、切分、建模、评估。"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42


def load_data():
    """加载数据，返回特征表和标签。"""
    df = load_breast_cancer(as_frame=True).frame
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y


def check_data(X, y):
    """检查数据是否可用，有问题就抛出说明原因的异常。"""
    if len(X) == 0 or len(y) == 0:
        raise ValueError('数据为空，无法训练')

    if len(X) != len(y):
        raise ValueError(
            f'特征和标签的行数不一致：特征 {len(X)} 行，标签 {len(y)} 行'
        )

    labels = set(y.unique()) #把标签去重后转成一个集合
    if not labels.issubset({0, 1}): #判断集合是否是 {0, 1} 的子集，即是不是有除了 0 和 1 之外的标签
        raise ValueError(f'标签只能是 0 或 1，实际出现了：{sorted(labels)}')


def split_data(X, y):
    """按 80/20 切分，先检查再切。"""
    check_data(X, y)
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )


def build_model(C=1.0):
    """建流水线，先检查 C 再建。"""
    if C <= 0:
        raise ValueError(f'C 必须大于 0，收到的是 {C}')
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=C)),
    ])


def train_and_predict(X_train, y_train, X_test, C=1.0):
    """训练并预测，返回模型和预测标签。"""
    check_data(X_train, y_train)
    model = build_model(C)
    model.fit(X_train, y_train)
    return model, model.predict(X_test)


def evaluate(y_true, y_pred):
    """算准确率，先检查长度是否一致。"""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f'真实标签和预测标签长度不一致：{len(y_true)} 对 {len(y_pred)}'
        )
    return accuracy_score(y_true, y_pred)