import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from multiprocessing import Pool
import time as ti
import pingouin as pg
# 标准化器
Stand_X = StandardScaler()


# 自定义的交叉验证单个折叠函数
def cross_validation_fold(train_index, test_index, X_known, y_known, unknown_split,x):
    # 训练集和测试集划分
    X_train, y_train = X_known.iloc[train_index], y_known.iloc[train_index]
    X_test_known, y_test_known = X_known.iloc[test_index], y_known.iloc[test_index]

    # 合并已知类和未知类的测试集
    X_test = pd.concat([X_test_known, unknown_split.iloc[:, :13]], ignore_index=True)
    y_test = pd.concat([y_test_known, unknown_split.iloc[:, 13]], ignore_index=True)

    # 数据标准化
    X_train = Stand_X.fit_transform(X_train)
    X_test = Stand_X.transform(X_test)

    # 固定 C 值的 SVM 模型
    model = SVC(C=15, kernel='rbf', decision_function_shape='ovr', probability=True)

    model.fit(X_train, y_train)

    threshold = 0.8
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    max_prob = np.max(y_prob, axis=1)
    unknown_samples = np.max(y_prob, axis=1)
    y_pred[max_prob < threshold] = 51


    # 预测并计算准确率
    cm = confusion_matrix(y_test, y_pred)

    diam = np.trace(cm)
    sum = cm.sum()

    TN = cm[-1, -1]
    TP = diam - TN

    FP = cm[-1, :].sum() - TN
    FN = cm[:, -1].sum() - TN
    # FN=sum-TN-TP-FP

    accuracy = diam / sum
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    y=[accuracy, precision, recall, f1]
    # print("准确",accuracy)
    # print("精确",precision)
    # print("召回",recall)
    # print("F1",f1)

    x.append(y)
    return x


# 并行化交叉验证
def parallel_cross_validation(X_known, y_known, unknown_splits, n_splits=5):
    # 交叉验证折叠
    kf = KFold(n_splits=n_splits, shuffle=True,random_state=42)

    # 设置多进程
    t0=ti.time()
    x=[]
    with Pool() as pool:
        y = pool.starmap(
            cross_validation_fold,
            [(train_index, test_index, X_known, y_known, unknown_splits[i],x)
             for i, (train_index, test_index) in enumerate(kf.split(X_known))]
        )
    t1 = ti.time()
    time = t1 - t0
    print(time)
    # 返回平均准确率
    return y


def original():


    known_data = pd.read_csv('new_gait_dataset/avova.csv')
    unknown_data = pd.read_csv('new_gait_dataset/avova_unknow.csv')
    unknown_splits = np.array_split(unknown_data, 10)
    X_known = known_data.iloc[:, :17]
    y_known = known_data.iloc[:, 17]
    x = parallel_cross_validation(X_known, y_known, unknown_splits, n_splits=5)
    z = np.array(x).reshape(5, 4)
    meandata = np.mean(z, axis=0)
    print(z[:, 0])

    print("交叉验证得分:", meandata)

def only_feature():
    known_data = pd.read_csv('new_gait_dataset/only_feature.csv')
    unknown_data = pd.read_csv('new_gait_dataset/only_feature_vaild.csv')
    unknown_splits = np.array_split(unknown_data, 6)
    X_known = known_data.iloc[:, :13]
    y_known = known_data.iloc[:, 13]
    x = parallel_cross_validation(X_known, y_known, unknown_splits, n_splits=5)
    z = np.array(x).reshape(5, 4)
    meandata = np.mean(z, axis=0)
    print(z[:, 0])

    print("交叉验证得分:", meandata)

    return z[:, 0]

def only_sample():
    known_data = pd.read_csv('new_gait_dataset/only_sample_original_gait_dataset.csv')
    unknown_data = pd.read_csv('new_gait_dataset/only_sample_original_vaild_gait_dataset.csv')
    unknown_splits = np.array_split(unknown_data, 6)
    X_known = known_data.iloc[:, :17]
    y_known = known_data.iloc[:, 17]

    x = parallel_cross_validation(X_known, y_known, unknown_splits, n_splits=5)
    z = np.array(x).reshape(5, 4)
    meandata = np.mean(z, axis=0)
    print(z[:, 0])

    print("交叉验证得分:", meandata)

    return z[:, 0]

def both():
    known_data = pd.read_csv('new_gait_dataset/both_original_dataset.csv')
    unknown_data = pd.read_csv('new_gait_dataset/both_original_vaild_dataset.csv')
    unknown_splits = np.array_split(unknown_data, 6)
    X_known = known_data.iloc[:, :13]
    y_known = known_data.iloc[:, 13]
    x = parallel_cross_validation(X_known, y_known, unknown_splits, n_splits=5)
    z = np.array(x).reshape(5, 4)
    meandata = np.mean(z, axis=0)
    print(z[:, 0])

    print("交叉验证得分:", meandata)

    return z[:, 0]



if __name__ == "__main__":


    # original()


    original_acc=[0.95915033 ,0.98315525, 0.98315525 ,0.93122957 ,0.98227307]
    # only_feature=only_feature()
    #
    #
    # t_stat, p_value = stats.ttest_ind( original_acc,   only_feature)
    # p_value = pg.ttest(original_acc,   only_feature)
    #
    # print(p_value['p-val'])

    # only_sample=only_sample()
    # t_stat, p_value = stats.ttest_ind(original_acc, only_sample)
    # p_value = pg.ttest(original_acc, only_sample)
    #
    # print(p_value['p-val'])


    both=both()
    t_stat, p_value = stats.ttest_ind( original_acc,   both)
    p_value = pg.ttest(original_acc,   both)
    print(p_value['p-val'])
