# 乳腺癌数据集逻辑回归实践

## 数据集

本项目唯一使用的数据集是 `sklearn.datasets.load_breast_cancer`，
随 scikit-learn 一并提供，不需要额外下载。

- 样本数：569
- 特征数：30 个数值特征
- 类别：212 个恶性（malignant，标签 0）、357 个良性（benign，标签 1）
- 缺失值：无

## 运行方式

1. 安装依赖：`pip install -r requirements.txt`
2. 运行主程序：双击 `run.bat`，或在项目文件夹里执行 `python main.py`
3. 运行测试：`python -m pytest tests -v`

## 可复现性

所有涉及随机的步骤（数据划分、交叉验证、网格搜索）都固定 `random_state=42`，
在 `main.py` 顶部以常量 `RANDOM_STATE` 统一定义，各函数一律引用这个常量，
不在函数内部另写数字。

## 目录说明

- `main.py`：主程序
- `figures/`：程序生成的图片
- `tests/`：pytest 测试
- `requirements.txt`：依赖清单
- `run.bat`：一键运行脚本

## 进度

- [x] 第 1 节 工程环境与代码规范
- [x] 第 2 节 数据加载与探索
- [ ] 第 3 节 数据预处理 Pipeline
- [ ] 第 4 节 模型训练与预测
- [ ] 第 5 节 模型评估
- [ ] 第 6 节 可视化分析
- [ ] 第 7 节 正则化与超参数分析
- [ ] 第 8 节 交叉验证与网格搜索
- [ ] 第 9 节 异常处理与测试复现
- [ ] 第 10 节 报告与答辩
