"""第11周上机实践,乳腺癌数据集的逻辑回归实践。

唯一数据集:sklearn.datasets.load_breast_cancer
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import load_breast_cancer

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

#数据加载和预处理
data = load_breast_cancer(as_frame=True)
#调 sklearn 自带的函数把数据拿到手。as_frame=True 让它直接整理成表格。不加这个参数交回的是两个裸数组，得自己建表、自己起列名

df = data.frame

print('形状:', df.shape)
print('特征数:', len(data.feature_names))
print(df.head())
print(df['target'].value_counts())
print('缺失值总数:', df.isna().sum().sum())

#画柱状图
import matplotlib.pyplot as plt

counts = df['target'].value_counts().sort_index()
#统计每个类别的样本数，并按类别顺序排序。
plt.figure()
plt.bar(['malignant (0)', 'benign (1)'], counts.values)
#柱状图。第一个参数是每根柱子底下写什么，第二个参数是柱子多高。

plt.title('Class distribution')
plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.savefig('figures/class_distribution.png')

#新开一个画布，展示特征直方图
cols = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
plt.figure()
for i, col in enumerate(cols, start=1): #cols是一个列表，里面是四个特征的名字。enumerate()函数可以把列表变成一个可迭代对象，start=1表示索引从1开始。
    plt.subplot(2, 2, i)
    #把画布切成 2 行 2 列共 4 块，这一轮画在第 i 块。之后的 plt.xxx 都画在这一块里，直到下一次 subplot 换地方。
    
    plt.hist(df.loc[df['target'] == 0, col], label='malignant')
    #loc[] 是 pandas 的索引器，左边是行，右边是列。df['target'] == 0 是一个布尔数组，表示哪些行的 target 列是 0。df.loc[df['target'] == 0, col] 就是取出这些行的 col 列的值。
    plt.hist(df.loc[df['target'] == 1, col], label='benign')
   #同样，去索引出 target 列是 1 的行的 col 列的值。
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Number of samples')
    plt.legend()

plt.tight_layout()  # 自动调整四块子图的间距，避免上面的标题压住下面的坐标轴标签
plt.savefig('figures/feature_histograms.png')
plt.show()

#切分数据，标准化，pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

X = df.drop(columns=['target'])  #.drop()函数可以删除指定的列，columns参数指定要删除的列名。这里删除了'target'列，剩下的就是特征数据。
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
#stratify=y：切的时候保持比例。原数据 357 良性 212 恶性，切完训练集是 285 良性 170 恶性

pipe = Pipeline([('scaler', StandardScaler()),('clf', LogisticRegression()),])
#pipeline()函数可以把多个步骤串起来，形成一个整体。这里有两个步骤：第一步是标准化，第二步是逻辑回归分类器。每个步骤都是一个二元组，第一个元素是步骤的名字，第二个元素是具体的处理器。
pipe.fit(X_train, y_train)
#用训练集算出均值和标准差，然后把训练集标准化,并进行逻辑回归的训练

print('训练集:', X_train.shape, ' 测试集:', X_test.shape)
print('训练集标签:', y_train.value_counts().to_dict())

scaler = pipe.named_steps['scaler']
print('scaler 学到的均值(前3个):', scaler.mean_[:3].round(4))
print('训练集均值(前3个):      ', X_train.mean().values[:3].round(4))

#系数和截距
clf = pipe.named_steps['clf']

print('系数形状:', clf.coef_.shape)
print('截距:', clf.intercept_)

print('30 个特征的系数:')
""" zip 是 Python 的内置函数，接收两个（或更多）可以逐个取值的对象，按位置把它们一一配对，第 1 个对第 1 个，第 2 个对第 2 个，这样排下去。
它返回的不是列表，是一个「取的时候才生成」的对象（类型就是 zip），
所以直接打印看不到内容，要用 list() 转一下，或者放进 for 里循环。"""
for name, w in zip(X.columns, clf.coef_[0]):
    print(f'{name:<22} {w:+.4f}')   #展示30个特征对应的系数
    #{name:<22}：名字占 22 个字符宽，靠左对齐（< 是左对齐，22 是宽度）。作用是让后面那列数字对齐成一竖，不然名字有长有短，输出会参差不齐
    #{w:+.4f}：+ 表示正数也显示加号，负号本来就显示；.4f 是保留 4 位小数
    
#接下来看训练好的模型的预测情况
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test) 
#这一步是预测每个样本属于每个类别的概率，返回一个二维数组，第一列是属于类别 0（恶性）的概率，第二列是属于类别 1（良性）的概率。
#predict_proba函数是在预测每个样本属于每个类别的概率，返回一个二维数组，第一列是属于类别 0（恶性）的概率，第二列是属于类别 1（良性）的概率。
print('预测标签前 10 个:', y_pred[:10])
print('标签数组的形状:', y_pred.shape)
print('概率矩阵形状:', y_prob.shape)
print('前 5 行概率:')
print(y_prob[:5].round(4))
print('概率最小值:', round(y_prob.min(), 4), ' 最大值:', round(y_prob.max(), 4))

#调整决策阈值，即不一定是大于0.5的时候才判定为良性，可以根据实际情况调整阈值。
threshold = 0.9
y_pred_t = (y_prob[:, 1] >= threshold).astype(int)
'''y_prob[:, 1]  是从那个概率表里，取所有行的第 1 列。  
   >= threshold  是逐个和 0.9 比，得到一个长度 114 的 True/False 数组。
   astype(int)    是把 True/False 转成 1/0,得到一个长度 114 的 0/1 数组。
'''

print('阈值:', threshold)
print('判为良性的个数:', int(y_pred_t.sum()))
print('和默认阈值相比，改判了', int((y_pred_t != y_pred).sum()), '个样本')

#模型评估-混淆矩阵（准确率、精确率、召回率、F1分数）等
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print('准确率:', round(accuracy_score(y_test, y_pred), 4))

cm = confusion_matrix(y_test, y_pred)
#交回一张 2 行 2 列的表,行是真实类别，列是预测类别,顺序都是[0, 1]，即第一行是恶性，第二行是良性。第一列是预测为恶性，第二列是预测为良性。
print('混淆矩阵:')
print(cm)
print(classification_report(y_test, y_pred, digits=4))
''' classification_report()函数会返回一个字符串，
里面包含了每个类别的精确率、召回率(tpr)、F1分数和支持度（support，即每个类别的样本数）
以及宏平均（macro avg）和加权平均（weighted avg）的指标。digits=4表示保留 4 位小数。
精确率是针对预测结果而言的，即预测这类里有几个是真的属于这一类，召回率是针对真实结果而言的，即真实这类里有几个被预测对了。'''

#模型评估-ROC曲线和AUC
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1]) #fpr:假阳性率，tpr:真阳性率也就是召回率，thresholds:阈值
'''roc_curve函数有两个参数，第一个传的是真实标签，第二个传的是预测为阳性的概率。返回三个数组
   这个函数将阈值从高到低调整，计算每个阈值下的假阳性率和真阳性率，从而绘制出 ROC 曲线。'''

roc_auc = auc(fpr, tpr)
#auc计算 ROC 曲线下面积（Area Under Curve, AUC），传入假阳性率和真阳性率即可。
print('AUC:', round(roc_auc, 4))

#将上面那些数据画图
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.title('ROC curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.savefig('figures/roc_curve.png')
plt.show()

#可视化分析，挑两个特征重新训练模型，画出决策边界并标出错判的样本
X2 = df[['mean radius', 'mean texture']] 
#得到一个新的特征矩阵，只有两列，分别是 mean radius 和 mean texture

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

pipe2 = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
pipe2.fit(X2_train, y2_train)

print('二维模型 训练集准确率:', round(pipe2.score(X2_train, y2_train), 4))
print('二维模型 测试集准确率:', round(pipe2.score(X2_test, y2_test), 4))
#score等价于先predict再accuracy_score，直接返回准确率。

y2_pred = pipe2.predict(X2_test)
wrong = y2_pred != y2_test
print('二维模型 测试集误分个数:', int(wrong.sum()))

from sklearn.inspection import DecisionBoundaryDisplay

plt.figure()
DecisionBoundaryDisplay.from_estimator(pipe2, X2_test,
                                       response_method='predict', alpha=0.25)
#这个函数用来画决策边界。传入训练好的模型和测试集特征矩阵，response_method='predict'表示用预测标签来决定边界，alpha=0.25表示边界的透明度。

plt.scatter(X2_test.loc[y2_test == 0, 'mean radius'],
            X2_test.loc[y2_test == 0, 'mean texture'],
            label='malignant', marker='o')
#第一批散点是显示每个恶性样本的 mean radius 和 mean texture，marker='o'表示用圆圈表示。

plt.scatter(X2_test.loc[y2_test == 1, 'mean radius'],
            X2_test.loc[y2_test == 1, 'mean texture'],
            label='benign', marker='^')
#第二批散点是显示每个良性样本的 mean radius 和 mean texture，marker='^'表示用三角形表示。

plt.scatter(X2_test.loc[wrong, 'mean radius'],
            X2_test.loc[wrong, 'mean texture'],
            label='misclassified', marker='x', color='black')
#第三批散点是显示所有错判样本的 mean radius 和 mean texture，marker='x'表示用叉号表示，color='black'表示用黑色。
plt.title('Decision boundary (mean radius vs mean texture)')
plt.xlabel('mean radius')
plt.ylabel('mean texture')
plt.legend()
plt.savefig('figures/decision_boundary.png')
plt.show()

#较不同正则化强度 C 和决策阈值对准确率、F1 的影响；分析过拟合与欠拟合
from sklearn.metrics import f1_score

Cs = [0.001, 0.01, 0.1, 1, 10, 100]

for C in Cs:
    p = Pipeline([('scaler', StandardScaler()),
                  ('clf', LogisticRegression(C=C))])  #每轮新建一个回归模型
    p.fit(X_train, y_train)
    yp = p.predict(X_test)  #得到预测标签
    print(f'C={C:<7} 训练={p.score(X_train, y_train):.4f} ' #得到训练集的准确率
          f'测试={p.score(X_test, y_test):.4f} '  #测试集的准确率
          f'F1={f1_score(y_test, yp):.4f}')     #测试集的 F1 分数
#这里预期输出，当 C 很小的时候，正则化很强，模型会欠拟合，训练集和测试集的准确率都不高；当 C 很大的时候，正则化很弱，模型可能会过拟合，训练集准确率很高，但测试集准确率下降.

#接下来针对 C=1 的模型，调整决策阈值，观察准确率和 F1 分数的变化,
# 为什么是C=1呢？因为前面输出的结果显示 C=1 时训练集和测试集的准确率都比较高，说明模型拟合得比较好，所以选择这个模型来调整阈值.
pipe_best = Pipeline([('scaler', StandardScaler()),
                      ('clf', LogisticRegression(C=1))])
pipe_best.fit(X_train, y_train)
proba = pipe_best.predict_proba(X_test)[:, 1] #选择所有行的第 1 列，即预测为良性的概率

for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    yp = (proba >= t).astype(int)   #在每一轮中，当前的t和预测概率比较，如果大的话，得到真值，得到一个布尔数组，再转成整数数组，这是在得到当前阈值下的预测标签（为什么要我们手动比较？因为predict不接受阈值参数，只能用predict_proba得到概率，再自己比较）
    print(f'阈值={t:<5} 准确率={accuracy_score(y_test, yp):.4f} '
          f'F1={f1_score(y_test, yp):.4f}')

#交叉验证和网格搜索
#先是5折交叉验证，使用 cross_val_score 函数来评估模型在不同的训练/验证集上的表现。然后使用 GridSearchCV 来进行网格搜索，寻找最佳的正则化参数 C。
from sklearn.model_selection import cross_val_score, GridSearchCV

scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy') #倒数第二个参数是交叉验证的折数，最后一个参数是评分指标，这里用准确率。
print('5 折得分:', scores.round(4))  #返回一个长度为 5 的数组，表示每一轮的准确率得分。
print('平均:', round(scores.mean(), 4), ' 标准差:', round(scores.std(), 4))

#再是网格搜索，把所有想试的组合都列出来让sklearn自己去循环
param_grid = {
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__l1_ratio': [0.0, 1.0],
    'clf__solver': ['liblinear', 'saga'],
}  #这里的字典键是参数名，值是候选的取值，它会进行笛卡尔积组合，尝试每一种组合。
#注意参数名是 'clf__C' 这样的形式，因为这里是对流水线中的clf那一步也就是逻辑回归分类器进行调参，'clf' 是流水线中逻辑回归的名字，'C' 是逻辑回归的参数名，中间用双下划线连接。
#这里的solver是算法，只写这两种算法是因为这两种算法都支持 l1 和 l2 正则化，其他算法不支持 l1 正则化，所以不写。
base = Pipeline([('scaler', StandardScaler()),
                 ('clf', LogisticRegression(max_iter=20000))]) #max_iter是迭代次数的上限

grid = GridSearchCV(base, param_grid, cv=5, scoring='accuracy')
#参数是：第一个是模型，第二个是参数网格，第三个是交叉验证的折数，第四个是评分指标，这里用准确率。
grid.fit(X_train, y_train)

print('最佳参数:', grid.best_params_) #这个返回的是一个字典，里面是最佳参数的组合。
print('最佳交叉验证得分:', round(grid.best_score_, 4)) #返回的是最佳参数组合下的平均交叉验证得分。
print('测试集得分:', round(grid.score(X_test, y_test), 4)) #返回的是最佳参数组合下的测试集得分。

#这里会发现它给的最佳参数组合是 C=0.1, penalty='l2', solver='liblinear'，这个组合在交叉验证中得分最高。测试集得分也比较高，说明模型泛化能力不错.但其实，C=1的模型交叉验证得分一样，这时候它会给最先出现的那一个。

#接下来画C和验证得分的关系图，看看不同的C值对模型性能的影响。
Cs = [0.001, 0.01, 0.1, 1, 10, 100]
means = []

for C in Cs:
    p = Pipeline([('scaler', StandardScaler()),
                  ('clf', LogisticRegression(C=C, solver='liblinear'))])
    s = cross_val_score(p, X_train, y_train, cv=5, scoring='accuracy')
    means.append(s.mean())

plt.figure()
plt.plot(Cs, means, marker='o') #横坐标是 C 的取值，纵坐标是对应的平均交叉验证得分，marker='o'表示每个点用圆圈标记。
plt.xscale('log')  #这是在把横轴设成对数刻度，因为C的取值跨度很大，使用对数刻度可以更好地展示趋势。
plt.title('5-fold CV accuracy vs C')
plt.xlabel('C (log scale)')
plt.ylabel('Mean CV accuracy')
plt.savefig('figures/cv_vs_C.png')
plt.show() 




